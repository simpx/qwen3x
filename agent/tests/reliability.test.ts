import { expect, test } from "bun:test";
import { appendFileSync, chmodSync, existsSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Session } from "../session.ts";
import { bash } from "../tools/bash.ts";
import { Agent } from "../agent.ts";
import { answer, command, config, event, records, run, sandbox, serve, tool, until } from "./support.ts";

test("real SIGKILL recovery before effects, after effects and after durable tool result", async () => {
  for (const phase of ["before", "after", "recorded"]) {
    const box = sandbox(), file = join(box.cwd, "session.jsonl");
    let resumed = false;
    const server = serve(body => {
      if (resumed) {
        const result = body.messages.findLast((m: any) => m.role === "tool");
        expect(result?.content).toContain(phase === "recorded" ? "exit_code=0" : "unknown");
        return answer("Recovered without replay");
      }
      if (body.messages.at(-1).role === "tool") return new Response(new ReadableStream({ start() {} }),
        { headers: { "content-type": "text/event-stream" } });
      return tool(`echo $$ > tool.pid; ${phase !== "before" ? "printf 'effect\\n' >> effects;" : ""}
        touch ready; ${phase === "recorded" ? "true" : "while test ! -f release; do sleep 0.05; done"}`);
    });
    const child = Bun.spawn([...command(), "--base-url", server.url, "-m", "test", "-o", file, "-p", "perform task"],
      { cwd: box.cwd, stdin: "ignore", stdout: "ignore", stderr: "ignore" });
    try {
      await until(() => existsSync(join(box.cwd, "ready")), "tool process readiness");
      if (phase === "recorded") await until(() => records(file).some(e => e.message?.role === "tool"), "durable tool result");
      child.kill("SIGKILL"); await child.exited;
      // SIGKILL cannot execute cleanup. Observe and explicitly clean the test's orphan group.
      const pid = Number(readFileSync(join(box.cwd, "tool.pid"), "utf8"));
      let orphan = false;
      try { process.kill(-pid, 0); orphan = true; process.kill(-pid, "SIGKILL"); } catch {}
      if (phase !== "recorded") expect(orphan).toBe(true);
      const before = existsSync(join(box.cwd, "effects")) ? readFileSync(join(box.cwd, "effects"), "utf8") : "";
      resumed = true;
      const result = await run(["--base-url", server.url, "-m", "test", "-r", file, "-p", "Inspect and continue"], box.cwd);
      expect(result.code).toBe(0);
      expect(existsSync(file + ".lock")).toBe(false);
      expect(existsSync(join(box.cwd, "effects")) ? readFileSync(join(box.cwd, "effects"), "utf8") : "").toBe(before);
    } finally {
      child.kill();
      if (existsSync(join(box.cwd, "tool.pid"))) try { process.kill(-Number(readFileSync(join(box.cwd, "tool.pid"), "utf8")), "SIGKILL"); } catch {}
      server.close(); box.close();
    }
  }
}, 15000);

test("actual concurrent resumptions have exactly one writer, including stale lock takeover", async () => {
  const box = sandbox(), file = join(box.cwd, "session.jsonl");
  Session.open(box.cwd, undefined, file).close();
  const server = serve(() => new Response(new ReadableStream({ start() {} }), { headers: { "content-type": "text/event-stream" } }));
  try {
    for (let round = 0; round < 5; round++) {
      // Obtain a known exited PID rather than guessing an unused PID.
      const dead = Bun.spawn(["true"]); await dead.exited;
      writeFileSync(file + ".lock", `${dead.pid}\n`);
      const children = Array.from({ length: 8 }, (_, index) => Bun.spawn([...command(), "--base-url", server.url,
        "-m", "test", "-r", file, "-p", `contender ${round}:${index}`, "--request-timeout", "3"],
        { cwd: box.cwd, stdin: "ignore", stdout: "ignore", stderr: "ignore" }));
      try {
        await until(() => children.filter(p => p.exitCode !== null).length >= 7, "losing session writers", 2000);
        const live = children.filter(p => p.exitCode === null);
        expect(live).toHaveLength(1);
        await until(() => records(file).some(e => e.message?.role === "user" && e.message.content.startsWith(`contender ${round}:`)), "winning writer's first record");
        expect(records(file).filter(e => e.message?.role === "user" && e.message.content.startsWith(`contender ${round}:`))).toHaveLength(1);
      } finally {
        for (const child of children) if (child.exitCode === null) child.kill("SIGTERM");
        await Promise.all(children.map(child => child.exited));
      }
    }
  } finally { server.close(); box.close(); }
}, 15000);

test("every truncated UTF-8 journal tail is preserved byte-for-byte", () => {
  const box = sandbox(), base = join(box.cwd, "base.jsonl");
  Session.open(box.cwd, undefined, base).close();
  const original = readFileSync(base);
  const tail = Buffer.from(JSON.stringify({ type: "message", message: { role: "user", content: "你好🌲" } }));
  try {
    for (let i = 1; i < tail.length; i++) {
      const file = join(box.cwd, `cut-${i}.jsonl`);
      writeFileSync(file, Buffer.concat([original, tail.subarray(0, i)]));
      Session.open(undefined, file).close();
      const recovered = records(file).find(e => e.type === "recovered_tail");
      expect(recovered).toBeDefined();
      const bytes = recovered.bytes_base64 ? Buffer.from(recovered.bytes_base64, "base64") : Buffer.from(recovered.raw);
      expect(bytes.equals(tail.subarray(0, i))).toBe(true);
    }
  } finally { box.close(); }
});

test("interrupted stale-lock recovery refuses a second recovery without touching the journal", async () => {
  const box = sandbox(), file = join(box.cwd, "session.jsonl");
  Session.open(box.cwd, undefined, file).close();
  const original = readFileSync(file), dead = Bun.spawn(["true"]); await dead.exited;
  writeFileSync(file + ".lock", `${dead.pid}\n`); writeFileSync(file + ".lock.reclaim", "");
  try {
    expect(() => Session.open(undefined, file)).toThrow("recovery is busy");
    expect(readFileSync(file).equals(original)).toBe(true);
    expect(readFileSync(file + ".lock", "utf8")).toBe(`${dead.pid}\n`);
    unlinkSync(file + ".lock.reclaim");
    Session.open(undefined, file).close();
    expect(existsSync(file + ".lock")).toBe(false);
  } finally { box.close(); }
});

test("TERM-ignoring descendants and inherited pipes are killed after timeout", async () => {
  const box = sandbox(), cfg = config(box.cwd); cfg.commandTimeout = 100;
  try {
    const start = performance.now();
    const result = await bash({ command: "trap '' TERM; (trap '' TERM; while :; do sleep 1; done) & echo $! > child.pid; wait" },
      cfg, new AbortController().signal, () => {});
    expect(result).toContain("timed out");
    expect(performance.now() - start).toBeLessThan(2000);
    const pid = Number(readFileSync(join(box.cwd, "child.pid"), "utf8"));
    // A zombie is exited, not a running descendant (reaping is the host init's job).
    if (process.platform === "linux" && existsSync(`/proc/${pid}/stat`))
      expect(readFileSync(`/proc/${pid}/stat`, "utf8").split(") ")[1]?.[0]).toBe("Z");
    else expect(() => process.kill(pid, 0)).toThrow();
  } finally { box.close(); }
});

test("cancelling the second tool preserves the first result and never starts the third", async () => {
  const box = sandbox(), cfg = config(box.cwd), file = join(box.cwd, "session.jsonl");
  const controller = new AbortController();
  const calls = ["printf first > first", "touch second; sleep 20", "touch third"].map((command, index) =>
    ({ index, id: `call-${index}`, type: "function", function: { name: "bash", arguments: JSON.stringify({ command }) } }));
  const server = serve(() => new Response(event({ tool_calls: calls }) + event({}, "tool_calls") + "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } }));
  cfg.baseUrl = server.url; cfg.commandTimeout = 30000;
  const session = Session.open(box.cwd, undefined, file);
  try {
    const result = new Agent(cfg, session, () => {}).run("three commands", controller.signal).catch(error => error);
    await until(() => existsSync(join(box.cwd, "second")), "second command marker");
    controller.abort(new Error("cancel second"));
    expect(await result).toBeInstanceOf(Error);
    expect(readFileSync(join(box.cwd, "first"), "utf8")).toBe("first");
    expect(existsSync(join(box.cwd, "third"))).toBe(false);
    session.close();
    const resumed = Session.open(undefined, file);
    try {
      const results = resumed.messages.filter(message => message.role === "tool");
      expect(results.map(message => message.tool_call_id)).toEqual(["call-0", "call-1", "call-2"]);
      expect(results[0]!.content).toContain("exit_code=0");
      expect(results[2]!.content).toContain("Tool interrupted");
      expect(records(file).filter(record => record.type === "tool_start").map(record => record.id)).toEqual(["call-0", "call-1"]);
    } finally { resumed.close(); }
  } finally { controller.abort(); session.close(); server.close(); box.close(); }
});

test("large stdout and stderr drain without deadlock and preserve bounded output", async () => {
  const box = sandbox(), cfg = config(box.cwd); cfg.commandTimeout = 5000; cfg.maxOutput = 4096;
  let displayed = 0;
  try {
    const result = await bash({ command: "(head -c 8000000 /dev/zero | tr '\\0' a) & (head -c 8000000 /dev/zero | tr '\\0' b >&2) & wait" },
      cfg, new AbortController().signal, (_kind, text) => { displayed += Buffer.byteLength(text); });
    expect(result).toContain("exit_code=0"); expect(result).toContain("15995904 output bytes omitted");
    expect(displayed).toBe(4096); expect(result.length).toBeLessThan(4200);
  } finally { box.close(); }
});

test("read-only session output fails before bash has side effects", async () => {
  const box = sandbox(), folder = join(box.cwd, "readonly");
  await Bun.write(join(folder, "existing"), "x"); chmodSync(folder, 0o500);
  const server = serve(() => tool("touch should-not-run"));
  try {
    const result = await run(["--base-url", server.url, "-m", "test", "-o", join(folder, "log"), "-p", "task"], box.cwd);
    expect(result.code).toBe(1); expect(existsSync(join(box.cwd, "should-not-run"))).toBe(false);
  } finally { chmodSync(folder, 0o700); server.close(); box.close(); }
});

test("PTY paste stays one unsubmitted task until Enter", async () => {
  const box = sandbox(); let received: string[] = [];
  const server = serve(body => { received.push(body.messages.at(-1).content); return answer(); });
  let output = "";
  const child = Bun.spawn([...command(), "--base-url", server.url, "-m", "test"], { cwd: box.cwd,
    terminal: { data(_term, bytes) { output += Buffer.from(bytes).toString(); } } });
  try {
    await until(() => output.includes("q3x> "), "prompt");
    child.terminal!.write("\x1b[200~第一行\nsecond line\x1b[201~");
    await Bun.sleep(100);
    expect(received).toHaveLength(0);
    child.terminal!.write("\r");
    await until(() => received.length > 0, "pasted task");
    expect(received).toEqual(["第一行\nsecond line"]);
  } finally { child.kill("SIGTERM"); await child.exited; child.terminal?.close(); server.close(); box.close(); }
});

test("20 PTY cancellations occur after actual bash starts and allow the next task", async () => {
  const box = sandbox(), file = join(box.cwd, "log.jsonl");
  const server = serve(body => {
    const last = body.messages.at(-1);
    return last.role === "user" && last.content.startsWith("hang-") ?
      tool(`touch ${last.content}; sleep 10`, last.content) : answer("NEXT-TASK-OK");
  });
  let output = "";
  const child = Bun.spawn([...command(), "--base-url", server.url, "-m", "test", "-o", file], { cwd: box.cwd,
    terminal: { data(_term, bytes) { output += Buffer.from(bytes).toString(); } } });
  try {
    await until(() => output.includes("q3x> "), "prompt");
    for (let i = 0; i < 20; i++) {
      child.terminal!.write(`hang-${i}\r`);
      await until(() => existsSync(join(box.cwd, `hang-${i}`)), "bash marker file");
      let start = output.length; child.terminal!.write("\x03");
      await until(() => output.slice(start).includes("q3x> "), "prompt after cancellation");
      start = output.length; child.terminal!.write(`next-${i}\r`);
      await until(() => output.slice(start).includes("NEXT-TASK-OK") && output.slice(start).includes("q3x> "), "new successful task");
    }
    child.terminal!.write("/exit\r"); expect(await child.exited).toBe(0);
    expect(records(file).filter(e => e.type === "turn_end" && e.status === "cancelled")).toHaveLength(20);
    expect(records(file).filter(e => e.type === "turn_end" && e.status === "completed")).toHaveLength(20);
  } finally { child.kill(); child.terminal?.close(); server.close(); box.close(); }
}, 20000);
