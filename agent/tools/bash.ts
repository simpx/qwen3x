import { spawn } from "node:child_process";
import { type Config, type Display, object, errorText } from "../types.ts";

export const bashTool = {
  type: "function", function: {
    name: "bash",
    description: "Run a bash command in the working directory. Each call uses a fresh shell. Output is bounded; read large files in sections.",
    parameters: { type: "object", properties: { command: { type: "string", description: "Bash command to execute" } },
      required: ["command"], additionalProperties: false },
  },
};

export async function bash(args: unknown, config: Config, signal: AbortSignal, display: Display): Promise<string> {
  if (!object(args) || typeof args.command !== "string" || !args.command.trim() ||
      Object.keys(args).some(key => key !== "command")) return "Tool error: bash requires only a nonempty command string";
  if (signal.aborted) return "Tool cancelled before execution";
  if (process.platform === "win32") return "Tool error: q3x currently requires Linux/macOS or WSL for bash process groups";
  const command: string = args.command;
  // A detached POSIX process group lets cancellation kill descendants too.
  const child = spawn("bash", ["--noprofile", "--norc", "-c", command], {
    cwd: config.cwd, detached: true, stdio: ["ignore", "pipe", "pipe"],
  });
  let reason = "", output = "", remaining = config.maxOutput, omitted = 0;
  let killTimer: ReturnType<typeof setTimeout> | undefined;
  const killGroup = (sig: NodeJS.Signals) => {
    if (!child.pid) return;
    try { process.kill(-child.pid, sig); }
    catch (error: any) { if (error.code !== "ESRCH") child.kill(sig); }
  };
  const stop = (why: string) => {
    if (reason) return;
    reason = why;
    killGroup("SIGTERM");
    killTimer = setTimeout(() => killGroup("SIGKILL"), 300);
  };
  const abort = () => stop("cancelled");
  signal.addEventListener("abort", abort, { once: true });
  if (signal.aborted) abort();
  const timer = setTimeout(() => stop("timed out"), config.commandTimeout);
  const consume = (chunk: Buffer) => {
    const count = Math.min(remaining, chunk.length);
    // setEncoding joins split pipe characters; truncate only at a UTF-8 boundary.
    let end = count;
    while (end > 0 && end < chunk.length && (chunk[end]! & 0xc0) === 0x80) end--;
    const text = chunk.subarray(0, end).toString("utf8");
    output += text; remaining -= count; omitted += chunk.length - end;
    display("tool", text);
  };
  child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
  child.stdout.on("data", (text: string) => consume(Buffer.from(text)));
  child.stderr.on("data", (text: string) => consume(Buffer.from(text)));
  try {
    const result = await new Promise<string>(resolve => {
      child.once("error", error => resolve(`spawn failed: ${errorText(error)}`));
      child.once("close", (code, sig) => resolve(`exit_code=${code ?? "null"}${sig ? ` signal=${sig}` : ""}`));
      // Shells can leave background children with inherited pipes. This tool is foreground-only.
      child.once("exit", () => killGroup("SIGKILL"));
    });
    return `${result}${reason ? ` (${reason})` : ""}\n${output}${omitted ? `\n[${omitted} output bytes omitted]` : ""}`;
  } finally {
    clearTimeout(timer); if (killTimer) clearTimeout(killTimer);
    signal.removeEventListener("abort", abort);
    killGroup("SIGKILL");
  }
}
