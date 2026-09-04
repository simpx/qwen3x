import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Agent, estimate } from "../agent.ts";
import { Session } from "../session.ts";
import { answer, config, event, records, sandbox, serve, tool, until } from "./support.ts";

test("thinking sessions reserve a useful summary budget without changing task thinking", async () => {
  const box = sandbox(), cfg = config(box.cwd); cfg.thinking = true; cfg.maxTokens = 4096;
  let summaries = 0, tasks = 0;
  const server = serve(body => {
    if (!body.tools) {
      summaries++;
      expect(body.chat_template_kwargs.enable_thinking).toBe(false);
      expect(body.max_tokens).toBe(4096);
      return answer("Only change a.py. b.py is out of scope. a.py already fixed; run its tests.");
    }
    tasks++; expect(body.chat_template_kwargs.enable_thinking).toBe(true); return answer("Done");
  }); cfg.baseUrl = server.url;
  const session = Session.open(box.cwd);
  try {
    session.add({ role: "user", content: "Only change a.py; leave b.py alone." });
    session.add({ role: "assistant", content: "evidence ".repeat(2000) });
    const agent = new Agent(cfg, session, () => {});
    await agent.compact(new AbortController().signal);
    await agent.run("Continue verification", new AbortController().signal);
    expect(summaries).toBeGreaterThan(0); expect(tasks).toBe(1); expect(cfg.thinking).toBe(true);
  } finally { session.close(); server.close(); box.close(); }
});

test("mid-summary failure or cancellation never commits a partial replacement", async () => {
  for (const cancel of [false, true]) {
    const box = sandbox(), cfg = config(box.cwd); cfg.context = 4096; cfg.maxTokens = 512;
    const file = join(box.cwd, "log.jsonl"), session = Session.open(box.cwd, undefined, file);
    const controller = new AbortController(); let calls = 0;
    const server = serve(() => {
      if (++calls === 1) return answer("first segment summarized");
      if (cancel) { controller.abort(new Error("cancel summary")); return answer(); }
      return new Response("summary failed", { status: 500 });
    }); cfg.baseUrl = server.url;
    session.add({ role: "user", content: "task" });
    session.add({ role: "assistant", content: "大量中文日志🌲路径/a/b/c\n".repeat(1000) });
    const before = structuredClone(session.messages), raw = readFileSync(file);
    try {
      await expect(new Agent(cfg, session, () => {}).compact(controller.signal)).rejects.toThrow();
      expect(calls).toBe(2); expect(session.messages).toEqual(before); expect(readFileSync(file).equals(raw)).toBe(true);
    } finally { session.close(); server.close(); box.close(); }
  }
});

test("empty, longer and truncated summaries retain the previous context", async () => {
  for (const kind of ["empty", "longer", "truncated"]) {
    const box = sandbox(), cfg = config(box.cwd);
    const session = Session.open(box.cwd); session.add({ role: "user", content: "brief task" });
    session.add({ role: "assistant", content: "old evidence ".repeat(100) });
    const before = structuredClone(session.messages);
    let called = false;
    const server = serve(() => { called = true; return kind === "truncated" ? new Response(event({ content: "partial" }) + event({}, "length"),
      { headers: { "content-type": "text/event-stream" } }) : answer(kind === "empty" ? "" : "long summary ".repeat(1000)); });
    cfg.baseUrl = server.url;
    try {
      await expect(new Agent(cfg, session, () => {}).compact(new AbortController().signal)).rejects.toThrow();
      expect(called).toBe(true);
      expect(session.messages).toEqual(before);
    } finally { session.close(); server.close(); box.close(); }
  }
});

test("large immutable instructions and a single huge task stop before requests", async () => {
  const box = sandbox(), cfg = config(box.cwd); cfg.context = 2048; cfg.maxTokens = 512;
  let calls = 0; const server = serve(() => { calls++; return answer(); }); cfg.baseUrl = server.url;
  const session = Session.open(box.cwd);
  try {
    await expect(new Agent(cfg, session, () => {}).run("huge task ".repeat(2000), new AbortController().signal)).rejects.toThrow("Context budget");
    expect(calls).toBe(0);
    session.messages = [{ role: "system", content: "immutable instruction ".repeat(2000) }];
    await expect(new Agent(cfg, session, () => {}).run("small task", new AbortController().signal)).rejects.toThrow("Context budget");
    expect(calls).toBe(0);
  } finally { session.close(); server.close(); box.close(); }
});

test("five compactions followed by reload preserve system and valid tool-message ordering", async () => {
  const box = sandbox(), cfg = config(box.cwd), file = join(box.cwd, "long.jsonl");
  const server = serve(body => body.tools ? tool("true") : answer("Verified a.py; leave b.py unchanged. Next: run tests.")); cfg.baseUrl = server.url;
  let session = Session.open(box.cwd, undefined, file);
  const system = structuredClone(session.messages[0]);
  try {
    for (let i = 0; i < 5; i++) {
      session.add({ role: "user", content: `Task ${i}: leave b.py unchanged` });
      session.add({ role: "assistant", content: "checked ".repeat(1200), tool_calls: [{ id: `call-${i}`, type: "function",
        function: { name: "bash", arguments: '{"command":"true"}' } }] });
      session.add({ role: "tool", tool_call_id: `call-${i}`, content: "exit_code=0" });
      const before = estimate(session.messages);
      await new Agent(cfg, session, () => {}).compact(new AbortController().signal);
      expect(estimate(session.messages)).toBeLessThan(before);
      expect(session.messages[0]).toEqual(system);
      const effective = structuredClone(session.messages); session.close(); session = Session.open(undefined, file);
      expect(session.messages).toEqual(effective);
    }
    expect(records(file).filter(e => e.type === "compact")).toHaveLength(5);
    expect(records(file).filter(e => e.message?.role === "tool")).toHaveLength(5);
  } finally { session.close(); server.close(); box.close(); }
});

test("100-turn session repeatedly auto-compacts and resumes with complete audit history", async () => {
  const box = sandbox(), cfg = config(box.cwd), file = join(box.cwd, "soak.jsonl");
  cfg.context = 4096; cfg.maxTokens = 512;
  let summaries = 0, requests = 0;
  const server = serve(body => {
    if (!body.tools) { summaries++; return answer("Earlier tasks were completed. Continue with the latest user request."); }
    const last = body.messages.at(-1);
    if (last.role === "tool") return answer("Verified: " + "evidence ".repeat(200));
    requests++;
    return tool("true", `call-${requests}`);
  }); cfg.baseUrl = server.url;
  let session = Session.open(box.cwd, undefined, file);
  try {
    for (let i = 0; i < 100; i++) {
      await new Agent(cfg, session, () => {}).run(`Task ${i}: verify the project`, new AbortController().signal);
      if (i % 10 === 9) { session.close(); session = Session.open(undefined, file); }
    }
    expect(requests).toBe(100); expect(summaries).toBeGreaterThan(10);
    const log = records(file);
    expect(log.filter(e => e.message?.role === "user")).toHaveLength(100);
    expect(log.filter(e => e.message?.role === "tool")).toHaveLength(100);
    expect(log.filter(e => e.type === "turn_end" && e.status === "completed")).toHaveLength(100);
  } finally { session.close(); server.close(); box.close(); }
}, 15000);
