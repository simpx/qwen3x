import { expect, test } from "bun:test";
import { chmodSync, copyFileSync, existsSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { answer, command, records, sandbox, serve, tool, until } from "./support.ts";

test.skipIf(!process.env.Q3X_TEST_BINARY)("binary alone: no Bun/Node, readonly cwd, Unicode paths, no .env autoload", async () => {
  const box = sandbox(), bin = join(box.cwd, "only-bin"), work = join(box.cwd, "readonly 项目");
  mkdirSync(bin); mkdirSync(work);
  const executable = join(bin, "q3x 程序");
  copyFileSync(resolve(process.env.Q3X_TEST_BINARY!), executable); chmodSync(executable, 0o700);
  symlinkSync(Bun.which("bash")!, join(bin, "bash"));
  writeFileSync(join(work, ".env"), "Q3X_BASE_URL=http://127.0.0.1:1/v1\nQ3X_MODEL=wrong-model\n");
  writeFileSync(join(work, "bunfig.toml"), '[run]\nshell = "system"\n'); chmodSync(work, 0o500);
  let discovered = 0;
  const server = serve((body, req) => {
    if (req.method === "GET") { discovered++; return Response.json({ data: [{ id: "correct-model" }] }); }
    expect(body.model).toBe("correct-model");
    if (body.messages.at(-1).role !== "tool") return tool("printf 'standalone-ready'");
    expect(body.messages.at(-1).content).toContain("exit_code=0\nstandalone-ready");
    return answer("Standalone succeeded");
  });
  try {
    const child = Bun.spawn([executable, "-p", "check", "-o", join(box.cwd, "session.jsonl")], {
      cwd: work, env: { PATH: bin, HOME: box.cwd, Q3X_BASE_URL: server.url }, stdin: "ignore", stdout: "pipe", stderr: "pipe",
    });
    const [out, err, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
    expect(code).toBe(0); expect(out).toContain("Standalone succeeded"); expect(discovered).toBe(1);
    expect(err).not.toContain("wrong-model");
  } finally { chmodSync(work, 0o700); server.close(); box.close(); }
});

test("missing bash becomes a clear tool result and leaves no pending call", async () => {
  const box = sandbox(), file = join(box.cwd, "log.jsonl");
  const server = serve(body => {
    if (body.messages.at(-1).role !== "tool") return tool("printf never");
    expect(body.messages.at(-1).content).toContain("spawn failed");
    return answer("Bash is unavailable");
  });
  try {
    const child = Bun.spawn([...command(), "--base-url", server.url, "-m", "test", "-p", "run", "-o", file],
      { cwd: box.cwd, env: { PATH: box.cwd }, stdin: "ignore", stdout: "pipe", stderr: "pipe" });
    const [out, _err, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
    expect(code).toBe(0); expect(out).toContain("Bash is unavailable");
    expect(records(file).filter(e => e.message?.role === "tool")).toHaveLength(1);
  } finally { server.close(); box.close(); }
});

test("SIGTERM during running bash exits 143, reaps the group and releases the session", async () => {
  const box = sandbox(), file = join(box.cwd, "log.jsonl");
  const server = serve(() => tool("echo $$ > pid; touch running; sleep 10"));
  const child = Bun.spawn([...command(), "--base-url", server.url, "-m", "test", "-p", "run", "-o", file],
    { cwd: box.cwd, stdin: "ignore", stdout: "ignore", stderr: "ignore" });
  try {
    await until(() => existsSync(join(box.cwd, "running")), "running bash");
    const pid = Number(readFileSync(join(box.cwd, "pid"), "utf8"));
    child.kill("SIGTERM"); expect(await child.exited).toBe(143);
    expect(() => process.kill(-pid, 0)).toThrow();
    expect(existsSync(file + ".lock")).toBe(false);
    expect(records(file).at(-1).status).toBe("cancelled");
  } finally { child.kill(); server.close(); box.close(); }
});
