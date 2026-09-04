import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { Config } from "../types.ts";

export function sandbox() {
  const cwd = mkdtempSync(join(tmpdir(), "q3x daily 中文 "));
  return { cwd, close: () => rmSync(cwd, { recursive: true, force: true }) };
}
export function config(cwd: string): Config {
  return { cwd, model: "test", baseUrl: "", context: 32768, maxTokens: 2048, maxTurns: 8,
    maxOutput: 4096, autoCompact: true, requestTimeout: 2000, commandTimeout: 2000 };
}
export function event(delta: unknown, finish: string | null = null) {
  return `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
}
export function answer(text = "Done") {
  return new Response(event({ content: text }) + event({}, "stop") + "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } });
}
export function tool(command: string, id = "call") {
  return new Response(event({ tool_calls: [{ index: 0, id, type: "function",
    function: { name: "bash", arguments: JSON.stringify({ command }) } }] }) + event({}, "tool_calls") + "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } });
}
export function serve(handler: (body: any, request: Request) => Response | Promise<Response>) {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
    return handler(request.method === "POST" ? await request.json() : null, request);
  } });
  return { url: `http://127.0.0.1:${server.port}/v1`, close: () => { server.stop(true); } };
}
export function command() {
  return process.env.Q3X_TEST_BINARY ? [resolve(process.env.Q3X_TEST_BINARY)] : [process.execPath, resolve(import.meta.dir, "../main.ts")];
}
export async function until(check: () => boolean, label: string, timeout = 5000) {
  const deadline = Date.now() + timeout;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${label}`);
    await Bun.sleep(10);
  }
}
export function records(file: string): any[] {
  try { return readFileSync(file, "utf8").trim().split("\n").flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } }); }
  catch { return []; }
}
export async function run(args: string[], cwd: string, env = process.env) {
  const child = Bun.spawn([...command(), ...args], { cwd, env, stdin: "ignore", stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  return { stdout, stderr, code };
}
