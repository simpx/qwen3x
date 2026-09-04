import { expect, test } from "bun:test";
import { complete, discoverModel, sse } from "../client.ts";
import { options } from "../main.ts";
import { answer, config, event, sandbox, serve } from "./support.ts";

test("32 deterministic wire fragmentations reconstruct identical interleaved tool calls and UTF-8", async () => {
  const box = sandbox(), cfg = config(box.cwd);
  const args = JSON.stringify({ command: "printf '你好\\n\"quote\" 🌲'" });
  const wire = ": heartbeat\r\n\r\n" + event({ role: "assistant", content: null }) +
    event({ reasoning_content: "想一想🌲" }) + event({ content: "正在执行。" }) +
    event({ tool_calls: [
      { index: 1, id: "second", type: "function", function: { name: "ba", arguments: args.slice(0, 8) } },
      { index: 0, id: "first", type: "function", function: { name: "bash", arguments: '{"command":' } },
    ] }) + event({ tool_calls: [
      { index: 0, function: { arguments: '"true"}' } },
      { index: 1, function: { name: "sh", arguments: args.slice(8) } },
    ] }) + event({}, "tool_calls") + "data: [DONE]\n\n";
  let seed = 0;
  const server = serve(() => {
    const bytes = Buffer.from(wire); let offset = 0, state = ++seed;
    return new Response(new ReadableStream({ pull(controller) {
      if (offset === bytes.length) { controller.close(); return; }
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const end = Math.min(offset + 1 + state % 47, bytes.length);
      controller.enqueue(bytes.subarray(offset, end)); offset = end;
    } }), { headers: { "content-type": "text/event-stream" } });
  });
  cfg.baseUrl = server.url;
  try {
    for (let i = 0; i < 32; i++) {
      const result = await complete(cfg, [], [], new AbortController().signal, () => {});
      expect(result.usage).toBeUndefined();
      expect(result.message).toEqual({ role: "assistant", content: "正在执行。", reasoning_content: "想一想🌲",
        tool_calls: [
          { id: "first", type: "function", function: { name: "bash", arguments: '{"command":"true"}' } },
          { id: "second", type: "function", function: { name: "bash", arguments: args } },
        ] });
    }
  } finally { server.close(); box.close(); }
});

test("malformed streaming payloads are rejected without returning executable tools", async () => {
  const box = sandbox(), cfg = config(box.cwd);
  const call = { index: 0, id: "a", type: "function", function: { name: "bash", arguments: "{}" } };
  const wires = [
    "data: not-json\n\n", "data: null\n\n", "data: {\"error\":\"failed\"}\n\n",
    event({ tool_calls: [{ ...call, index: -1 }] }),
    event({ tool_calls: [{ ...call, index: 128 }] }),
    event({ tool_calls: [{ ...call, id: 10 }] }),
    event({ tool_calls: [{ ...call, type: "custom" }] }),
    event({ tool_calls: [{ ...call, function: { name: "bash", arguments: {} } }] }),
    event({ tool_calls: [{ ...call, id: "" }] }) + event({}, "tool_calls"),
    event({}, "stop") + event({ content: "late data" }),
    event({}, "stop") + event({}, "stop"),
    event({ tool_calls: [call] }) + event({}, "length"),
    event({ content: "partial" }) + "data: [DONE]\n\n",
  ];
  let index = 0;
  const server = serve(() => new Response(wires[index++]!, { headers: { "content-type": "text/event-stream" } }));
  cfg.baseUrl = server.url;
  try { for (const _ of wires) await expect(complete(cfg, [], [], new AbortController().signal, () => {})).rejects.toThrow(); }
  finally { server.close(); box.close(); }
});

test("SSE line and event caps stop oversized responses", async () => {
  for (const wire of ["data: " + "x".repeat(4 * 1024 * 1024 + 1),
    Array.from({ length: 5 }, () => "data: " + "x".repeat(1024 * 1024) + "\n").join("")]) {
    await expect(async () => { for await (const _ of sse(new Response(wire).body!)) {} }).toThrow("exceeds");
  }
});

test("model discovery rejects empty/ambiguous/broken IDs and honors auth", async () => {
  const box = sandbox(), cfg = config(box.cwd); cfg.apiKey = "fake-key";
  const bodies = [{ data: [] }, { data: [{ id: "a" }, { id: "b" }] }, { data: [{ id: "" }] }, { data: [{ id: 9 }] }, { data: [{ id: "only" }] }];
  let index = 0;
  const server = serve((_body, request) => {
    expect(request.headers.get("authorization")).toBe("Bearer fake-key");
    return Response.json(bodies[index++]);
  }); cfg.baseUrl = server.url;
  try {
    for (let i = 0; i < bodies.length - 1; i++) await expect(discoverModel(cfg, new AbortController().signal)).rejects.toThrow("specify --model");
    expect(await discoverModel(cfg, new AbortController().signal)).toBe("only");
  } finally { server.close(); box.close(); }
});

test("HTTP 401/429/500 and connection refusal fail once without an implicit retry", async () => {
  const box = sandbox(), cfg = config(box.cwd); let calls = 0, status = 401;
  const server = serve(() => { calls++; return new Response(`failure-${status}`, { status }); }); cfg.baseUrl = server.url;
  try {
    for (status of [401, 429, 500]) await expect(complete(cfg, [], [], new AbortController().signal, () => {})).rejects.toThrow(`HTTP ${status}`);
    expect(calls).toBe(3);
    server.close();
    await expect(complete(cfg, [], [], new AbortController().signal, () => {})).rejects.toThrow();
  } finally { server.close(); box.close(); }
});

test("configuration precedence is CLI, Q3X, OpenAI, then defaults", () => {
  const env = { Q3X_API_KEY: "q3x", OPENAI_API_KEY: "openai", Q3X_MODEL: "model", Q3X_BASE_URL: "http://localhost:123/v1" };
  const a = options([], env).config!;
  expect(a.apiKey).toBe("q3x"); expect(a.baseUrl).toBe(env.Q3X_BASE_URL); expect(a.model).toBe("model");
  expect(options(["--api-key", "cli"], env).config!.apiKey).toBe("cli");
  expect(options([], { OPENAI_API_KEY: "openai" }).config!.apiKey).toBe("openai");
  expect(options([], {}).config!.apiKey).toBeUndefined();
  expect(options(["--temperature", "0.2"], {}).config!.temperature).toBe(0.2);
  expect(() => options(["--temperature", "NaN"])).toThrow();
  expect(() => options(["--temperature", "3"])).toThrow();
});
