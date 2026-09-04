import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync, appendFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, estimate } from "../agent.ts";
import { complete, sse } from "../client.ts";
import { options, terminalText } from "../main.ts";
import { Session } from "../session.ts";
import { bash, bashTool } from "../tools/bash.ts";
import { type Config, type Message } from "../types.ts";

const cleanups: (() => void)[] = [];
afterEach(() => { while (cleanups.length) cleanups.pop()!(); });
function directory() {
  const path = mkdtempSync(join(tmpdir(), "q3x-test-"));
  cleanups.push(() => rmSync(path, { recursive: true, force: true }));
  return path;
}
function session(cwd: string, resume?: string, output?: string) {
  const value = Session.open(cwd, resume, output);
  cleanups.push(() => value.close());
  return value;
}
function config(cwd = directory()): Config {
  return { cwd, baseUrl: "", model: "test-model", context: 32768, maxTokens: 2048, maxTurns: 8,
    requestTimeout: 3000, commandTimeout: 2000, maxOutput: 4096, autoCompact: true };
}
function stream(events: unknown[], done = true): Response {
  const data = events.map(event => `data: ${JSON.stringify(event)}\r\n\r\n`).join("") + (done ? "data: [DONE]\n\n" : "");
  const bytes = new TextEncoder().encode(data);
  let offset = 0;
  return new Response(new ReadableStream({ pull(controller) {
    if (offset === bytes.length) { controller.close(); return; }
    // Single-byte chunks exercise UTF-8 and SSE framing across reads.
    controller.enqueue(bytes.subarray(offset, ++offset));
  } }), { headers: { "content-type": "text/event-stream" } });
}
function chunk(delta: unknown, finish: string | null = null) {
  return { choices: [{ index: 0, delta, finish_reason: finish }] };
}
function answer(text = "Done") {
  return stream([chunk({ content: text }), chunk({}, "stop"),
    { choices: [], usage: { prompt_tokens: 200, completion_tokens: 10 } }]);
}
function tool(command: string, id = "call_1") {
  const args = JSON.stringify({ command });
  return stream([
    chunk({ reasoning_content: "检查一下。" }),
    chunk({ tool_calls: [{ index: 0, id, type: "function", function: { name: "ba", arguments: args.slice(0, 5) } }] }),
    chunk({ tool_calls: [{ index: 0, function: { name: "sh", arguments: args.slice(5) } }] }),
    chunk({}, "tool_calls"),
  ]);
}
function server(handler: (body: any, request: Request) => Response | Promise<Response>) {
  const value = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
    if (new URL(request.url).pathname.endsWith("/models")) return Response.json({ data: [{ id: "discovered" }] });
    return handler(await request.json(), request);
  } });
  cleanups.push(() => { value.stop(true); });
  return `http://127.0.0.1:${value.port}/v1`;
}

test("fragmented streaming reasoning, tool arguments, usage, real bash and final answer", async () => {
  const cfg = config(); cfg.model = ""; cfg.apiKey = "secret-for-test";
  writeFileSync(join(cfg.cwd, "AGENTS.md"), "Project rule: preserve marker.txt.");
  let requests = 0;
  cfg.baseUrl = server((body, request) => {
    expect(request.headers.get("authorization")).toBe("Bearer secret-for-test");
    expect(body.model).toBe("discovered");
    expect(body.stream).toBe(true);
    expect(body.messages[0].content).toContain("preserve marker.txt");
    expect(body.tools.map((t: any) => t.function.name)).toEqual(["bash"]);
    if (++requests === 1) return tool("printf '你好' > marker.txt; cat marker.txt");
    expect(body.messages.at(-1).content).toContain("exit_code=0\n你好");
    expect(body.messages.at(-2).reasoning_content).toBe("检查一下。");
    return answer("已完成。");
  });
  const file = join(cfg.cwd, "session.jsonl");
  const history = session(cfg.cwd, undefined, file);
  let displayed = "";
  await new Agent(cfg, history, (_kind, text) => { displayed += text; }).run("create marker", new AbortController().signal);
  expect(requests).toBe(2);
  expect(displayed).toContain("检查一下。");
  expect(displayed).toContain("已完成。");
  expect(readFileSync(join(cfg.cwd, "marker.txt"), "utf8")).toBe("你好");
  expect(history.messages.map(m => m.role)).toEqual(["system", "user", "assistant", "tool", "assistant"]);
  expect(readFileSync(file, "utf8")).not.toContain("secret-for-test");
});

test("SSE handles comments and multiline data; rejects incomplete final events", async () => {
  const body = new Response(": ping\n\ndata: {\ndata: \"ok\":true}\n\n").body!;
  const result: string[] = [];
  for await (const item of sse(body)) result.push(item);
  expect(JSON.parse(result[0]!)).toEqual({ ok: true });
  const cr: string[] = [];
  for await (const item of sse(new Response("data: bare CR\r\r").body!)) cr.push(item);
  expect(cr).toEqual(["bare CR"]);
  await expect(async () => { for await (const _ of sse(new Response("data: {}\n").body!)) {} }).toThrow("Incomplete SSE");
});

test("truncated or unfinished tool streams never execute", async () => {
  for (const finish of [null, "length", "content_filter"]) {
    const cfg = config();
    cfg.baseUrl = server(() => stream([chunk({ tool_calls: [{ index: 0, id: "danger", type: "function",
      function: { name: "bash", arguments: '{"command":"touch should-not-exist"}' } }] }),
      ...(finish ? [chunk({}, finish)] : [])]));
    const history = session(cfg.cwd);
    await expect(new Agent(cfg, history, () => {}).run("test", new AbortController().signal)).rejects.toThrow();
    expect(existsSync(join(cfg.cwd, "should-not-exist"))).toBe(false);
    expect(history.messages.map(m => m.role)).toEqual(["system", "user"]);
  }
});

test("multiple tool calls execute in order; malformed arguments become tool results", async () => {
  const cfg = config(); let count = 0;
  cfg.baseUrl = server(body => {
    if (++count === 1) return stream([chunk({ tool_calls: [
      { index: 1, id: "b", type: "function", function: { name: "bash", arguments: "bad json" } },
      { index: 0, id: "a", type: "function", function: { name: "bash", arguments: '{"command":"printf ok"}' } },
    ] }), chunk({}, "tool_calls")]);
    expect(body.messages.at(-2).tool_call_id).toBe("a");
    expect(body.messages.at(-2).content).toContain("ok");
    expect(body.messages.at(-1).tool_call_id).toBe("b");
    expect(body.messages.at(-1).content).toContain("Tool error");
    return answer();
  });
  await new Agent(cfg, session(cfg.cwd), () => {}).run("test", new AbortController().signal);
  expect(count).toBe(2);
});

test("HTTP failures, wrong content type, and duplicate call IDs fail clearly", async () => {
  const cfg = config();
  for (const response of [
    () => new Response("no key", { status: 401 }),
    () => Response.json({ choices: [] }),
    () => stream([chunk({ tool_calls: [0, 1].map(index => ({ index, id: "same", type: "function",
      function: { name: "bash", arguments: "{}" } })) }), chunk({}, "tool_calls")]),
  ]) {
    cfg.baseUrl = server(response);
    await expect(complete(cfg, [], [], new AbortController().signal, () => {})).rejects.toThrow();
  }
});

test("request cancellation preserves the user task and partial response in JSONL", async () => {
  const cfg = config(); const controller = new AbortController();
  cfg.baseUrl = server(() => new Response(new ReadableStream({ start(c) {
    c.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk({ content: "partial" }))}\n\n`));
  } }), { headers: { "content-type": "text/event-stream" } }));
  const file = join(cfg.cwd, "cancel.jsonl");
  const history = session(cfg.cwd, undefined, file);
  await expect(new Agent(cfg, history, (kind, text) => {
    if (kind === "text" && text === "partial") controller.abort(new Error("test cancellation"));
  }).run("task", controller.signal)).rejects.toThrow();
  expect(history.messages.at(-1)?.role).toBe("user");
  expect(readFileSync(file, "utf8")).toContain('"partial":"partial"');
});

test("request deadline stops a stalled SSE stream", async () => {
  const cfg = config(); cfg.requestTimeout = 60;
  cfg.baseUrl = server(() => new Response(new ReadableStream({ start() {} }),
    { headers: { "content-type": "text/event-stream" } }));
  await expect(complete(cfg, [], [], new AbortController().signal, () => {})).rejects.toThrow();
});

test("bash timeout and cancellation stop descendants; output is bounded", async () => {
  const cfg = config(); cfg.commandTimeout = 70; cfg.maxOutput = 64;
  const result = await bash({ command: "printf '%01000d' 0; sleep 0.4; touch escaped" }, cfg,
    new AbortController().signal, () => {});
  expect(result).toContain("timed out"); expect(result).toContain("output bytes omitted");
  expect(result.length).toBeLessThan(200);
  await Bun.sleep(450);
  expect(existsSync(join(cfg.cwd, "escaped"))).toBe(false);
  const controller = new AbortController(); cfg.commandTimeout = 2000;
  const run = bash({ command: "(sleep 0.4; touch child-escaped) & wait" }, cfg, controller.signal, () => {});
  setTimeout(() => controller.abort(), 60);
  expect(await run).toContain("cancelled");
  await Bun.sleep(450);
  expect(existsSync(join(cfg.cwd, "child-escaped"))).toBe(false);
});

test("bash commands use fresh shells and fixed cwd", async () => {
  const cfg = config();
  await bash({ command: "cd /; export Q3X_TEST_VALUE=yes" }, cfg, new AbortController().signal, () => {});
  const result = await bash({ command: 'pwd; printf "%s" "${Q3X_TEST_VALUE-unset}"' }, cfg, new AbortController().signal, () => {});
  expect(result).toContain(cfg.cwd); expect(result).toContain("unset");
  cfg.maxOutput = 4;
  const unicode = await bash({ command: "printf '你好世界'" }, cfg, new AbortController().signal, () => {});
  expect(unicode).toContain("\n你\n[9 output bytes omitted]");
  expect(unicode).not.toContain("�");
});

test("resume appends, fork preserves source, output never overwrites, lock prevents concurrent writes", () => {
  const cwd = directory(), file = join(cwd, "log.jsonl"), fork = join(cwd, "fork.jsonl");
  const first = session(cwd, undefined, file);
  first.add({ role: "user", content: "original task" });
  expect(() => Session.open(cwd, file)).toThrow("in use");
  first.close();
  const source = readFileSync(file, "utf8");
  expect(() => Session.open(cwd, undefined, file)).toThrow();
  const second = session(cwd, file, fork);
  second.add({ role: "assistant", content: "forked answer" }); second.close();
  expect(readFileSync(file, "utf8")).toBe(source);
  const resumed = session(cwd, file);
  resumed.add({ role: "assistant", content: "answer" }); resumed.close();
  expect(readFileSync(file, "utf8").startsWith(source)).toBe(true);
  expect(session(cwd, file).messages.at(-1)?.content).toBe("answer");
});

test("resume repairs unresolved tool calls without repeating side effects and preserves torn tail", () => {
  const cwd = directory(), file = join(cwd, "log.jsonl");
  const first = session(cwd, undefined, file);
  first.add({ role: "user", content: "task" });
  first.add({ role: "assistant", content: null, tool_calls: [{ id: "pending", type: "function",
    function: { name: "bash", arguments: '{"command":"touch never-replay"}' } }] });
  first.record({ type: "tool_start", id: "pending" }); first.close();
  appendFileSync(file, '{"type":"message","unfinished');
  const restored = session(cwd, file);
  expect(restored.messages.at(-1)?.role).toBe("tool");
  expect(restored.messages.at(-1)?.content).toContain("outcome is unknown");
  expect(existsSync(join(cwd, "never-replay"))).toBe(false);
  restored.close();
  const events = readFileSync(file, "utf8").trim().split("\n").map(line => JSON.parse(line));
  expect(events.find(event => event.type === "recovered_tail").raw).toBe('{"type":"message","unfinished');
  expect(session(cwd, file).messages.filter(m => m.role === "tool")).toHaveLength(1);
});

test("invalid interior JSONL or different cwd is rejected without modifying the file", () => {
  const cwd = directory(), file = join(cwd, "bad.jsonl");
  writeFileSync(file, '{bad}\n{}\n');
  expect(() => Session.open(cwd, file)).toThrow("line 1");
  expect(readFileSync(file, "utf8")).toBe('{bad}\n{}\n');
  rmSync(file);
  session(cwd, undefined, file).close();
  expect(() => Session.open(directory(), file)).toThrow("belongs to");
});

test("compaction persists raw history, preserves system and pending user task, resumes compacted context", async () => {
  const cfg = config(); const file = join(cfg.cwd, "log.jsonl");
  cfg.baseUrl = server(body => {
    expect(body.tools).toBeUndefined();
    return answer("Task: inspect files. File a.ts was checked; next run tests.");
  });
  const history = session(cfg.cwd, undefined, file);
  history.add({ role: "user", content: "inspect files" });
  history.add({ role: "assistant", content: "evidence ".repeat(1000) });
  history.add({ role: "user", content: "now run tests" });
  const original = readFileSync(file, "utf8"), before = estimate(history.messages);
  await new Agent(cfg, history, () => {}).compact(new AbortController().signal);
  expect(estimate(history.messages)).toBeLessThan(before);
  expect(history.messages.at(-1)?.content).toBe("now run tests");
  expect(readFileSync(file, "utf8").startsWith(original)).toBe(true);
  const effective = structuredClone(history.messages); history.close();
  expect(session(cfg.cwd, file).messages).toEqual(effective);
});

test("automatic compaction triggers before a model request; manual mode enforces budget", async () => {
  for (const auto of [true, false]) {
    const cfg = config(); cfg.context = 4096; cfg.maxTokens = 512; cfg.autoCompact = auto;
    let summaries = 0, responses = 0;
    cfg.baseUrl = server(body => {
      if (!body.tools) { summaries++; return answer("Earlier task completed. Next: verify tests."); }
      responses++; return answer("verified");
    });
    const history = session(cfg.cwd);
    history.add({ role: "user", content: "inspect" });
    history.add({ role: "assistant", content: "large evidence ".repeat(1000) });
    const agent = new Agent(cfg, history, () => {});
    if (auto) {
      await agent.run("verify tests", new AbortController().signal);
      expect(summaries).toBeGreaterThan(1); expect(responses).toBe(1);
    } else {
      await expect(agent.run("verify tests", new AbortController().signal)).rejects.toThrow("Context budget");
      expect(responses).toBe(0);
    }
  }
});

test("failed compaction leaves effective messages unchanged; turn limit is resumable", async () => {
  const cfg = config(); cfg.baseUrl = server(() => new Response("failed", { status: 503 }));
  const history = session(cfg.cwd); history.add({ role: "user", content: "task" });
  history.add({ role: "assistant", content: "earlier evidence ".repeat(100) });
  const original = structuredClone(history.messages);
  await expect(new Agent(cfg, history, () => {}).compact(new AbortController().signal)).rejects.toThrow("503");
  expect(history.messages).toEqual(original);
  cfg.maxTurns = 1; cfg.baseUrl = server(() => tool("true"));
  await expect(new Agent(cfg, history, () => {}).run(undefined, new AbortController().signal)).rejects.toThrow("max-turns");
  expect(history.messages.at(-1)?.role).toBe("tool");
});

test("server context rejection triggers at most one compact-and-retry", async () => {
  const cfg = config(); let requests = 0, summaries = 0;
  cfg.baseUrl = server(body => {
    if (!body.tools) { summaries++; return answer("Earlier work done; continue the task."); }
    requests++;
    return Response.json({ error: { message: "context length exceeded" } }, { status: 400 });
  });
  const history = session(cfg.cwd);
  history.add({ role: "user", content: "earlier task" });
  history.add({ role: "assistant", content: "evidence ".repeat(1000) });
  await expect(new Agent(cfg, history, () => {}).run("next", new AbortController().signal)).rejects.toThrow("context length exceeded");
  expect(requests).toBe(2); expect(summaries).toBe(1);
});

test("PTY interaction: multiline input, manual compact, cancellation, status and exit", async () => {
  const cfg = config(); let received = "";
  cfg.baseUrl = server(body => {
    if (!body.tools) return answer("The first task was completed successfully.");
    received = body.messages.at(-1).content;
    if (received === "hang") return tool("touch running-marker; sleep 10");
    return answer("Some verified evidence. ".repeat(300));
  });
  const command = process.env.Q3X_TEST_BINARY ? [process.env.Q3X_TEST_BINARY] : [process.execPath, join(import.meta.dir, "..", "main.ts")];
  let output = "";
  const child = Bun.spawn([...command, "--base-url", cfg.baseUrl, "--model", "test", "-o", join(cfg.cwd, "pty.jsonl")], {
    cwd: cfg.cwd, terminal: { cols: 100, rows: 30, data(_terminal, data) { output += Buffer.from(data).toString(); } },
  });
  cleanups.push(() => { child.kill(); child.terminal?.close(); });
  const waitFor = async (text: string, start = 0) => {
    const deadline = Date.now() + 3000;
    while (!output.slice(start).includes(text)) {
      if (Date.now() > deadline) throw new Error(`PTY missing ${text}: ${output.slice(-1200)}`);
      await Bun.sleep(10);
    }
  };
  await waitFor("q3x> ");
  let start = output.length;
  child.terminal!.write("first\\\nsecond\n");
  await waitFor("Some verified evidence", start);
  await waitFor("q3x> ", start);
  expect(received).toBe("first\nsecond");
  start = output.length; child.terminal!.write("/compact\n");
  await waitFor("Context compacted:", start); await waitFor("q3x> ", start);
  start = output.length; child.terminal!.write("hang\n");
  const deadline = Date.now() + 3000;
  while (!existsSync(join(cfg.cwd, "running-marker"))) {
    if (Date.now() > deadline) throw new Error("bash did not start");
    await Bun.sleep(10);
  }
  start = output.length; child.terminal!.write("\x03");
  await waitFor("q3x> ", start);
  expect(output.slice(start)).toContain("Cancelled");
  start = output.length; child.terminal!.write("/status\n");
  await waitFor("Context:", start); await waitFor("q3x> ", start);
  child.terminal!.write("/exit\n");
  expect(await child.exited).toBe(0);
  child.terminal!.close();
  expect(existsSync(join(cfg.cwd, "pty.jsonl.lock"))).toBe(false);
}, 10000);

test("help has no config side effects; CLI validation and terminal control filtering", () => {
  expect(options(["--help", "--context", "nonsense"])).toHaveProperty("text");
  expect(() => options(["--max-turns", "0"])).toThrow();
  expect(() => options(["--base-url", "https://user:pass@example.com/v1"])).toThrow();
  expect(() => options(["--context", "2048"])).toThrow();
  expect(terminalText("hello\x1b]52;c;secret\x07\n")).toBe("hello␛]52;c;secret\n");
});

test("standalone or source CLI performs a task, resumes and forks with correct exit status", async () => {
  const cfg = config(); cfg.baseUrl = server(() => answer("CLI passed"));
  const exe = process.env.Q3X_TEST_BINARY;
  const command = exe ? [exe] : [process.execPath, join(import.meta.dir, "..", "main.ts")];
  const file = join(cfg.cwd, "cli.jsonl");
  const run = async (args: string[]) => {
    const child = Bun.spawn([...command, "--base-url", cfg.baseUrl, "--model", "test", ...args],
      { cwd: cfg.cwd, stdin: "ignore", stdout: "pipe", stderr: "pipe" });
    const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
    return { stdout, stderr, code };
  };
  expect((await run(["--help"])).code).toBe(0);
  const result = await run(["-p", "task", "-o", file]);
  expect(result.code).toBe(0); expect(result.stdout).toContain("CLI passed");
  expect((await run(["-p", "again", "-o", file])).code).toBe(1);
  expect((await run(["-p", "continue", "-r", file])).code).toBe(0);
  const resumed = Session.open(undefined, file); expect(resumed.messages.filter(m => m.role === "user")).toHaveLength(2); resumed.close();
  expect((await run(["-p", "fork", "-r", file, "-o", join(cfg.cwd, "fork.jsonl")])).code).toBe(0);
});
