#!/usr/bin/env bun
import { parseArgs } from "node:util";
import { createInterface } from "node:readline";
import { Agent } from "./agent.ts";
import { endpoint } from "./client.ts";
import { Session } from "./session.ts";
import { type Config, type Display, errorText } from "./types.ts";
import { version } from "./package.json";
import { TerminalInput } from "./terminal.ts";

const HELP = `q3x ${version} — a small, independent coding agent

Usage: q3x [options]
       q3x -p "task" [options]
       printf 'task' | q3x [options]

  --base-url URL          OpenAI-compatible API base (default http://127.0.0.1:8000/v1)
  -m, --model NAME        Model ID; discover automatically if the server has exactly one
  --api-key KEY           API key (prefer Q3X_API_KEY or OPENAI_API_KEY)
  --cwd DIRECTORY        Working directory (resume uses the recorded directory)
  -p, --prompt TEXT       Run one task and exit
  -o, --output FILE       Create a new JSONL session; never overwrite
  -r, --resume FILE       Resume JSONL; append unless -o creates a fork
  --context N            Context capacity in tokens (default 32768; estimates, not a tokenizer)
  --max-tokens N         Reserved output tokens per request (default 4096)
  --max-turns N          Agent turns per task, excluding summaries (default 32)
  --no-auto-compact      Require manual /compact (default automatic at 85% input budget)
  --thinking on|off      Send enable_thinking; omitted by default for standard servers
  --temperature N        Sampling temperature 0–2 (default: server setting)
  --request-timeout N    Whole HTTP request timeout in seconds (default 300)
  --command-timeout N    Bash timeout in seconds (default 120)
  --max-output N         Retained/displayed bytes per bash call (default 32768)
  -h, --help             Show help without opening files or contacting a server
  -v, --version          Show version

Interactive: /compact, /continue, /status, /help, /exit
End a line with \\ to continue a multiline task. Ctrl-C cancels a running turn;
Ctrl-C at the prompt exits. Bash runs directly with your user permissions.
Environment: Q3X_BASE_URL, Q3X_MODEL, Q3X_API_KEY; OPENAI_BASE_URL/API_KEY fallbacks.
`;

export function options(argv: string[], env = process.env) {
  const { values, positionals } = parseArgs({ args: argv, strict: true, allowPositionals: true, options: {
    help: { type: "boolean", short: "h" }, version: { type: "boolean", short: "v" },
    "base-url": { type: "string" }, model: { type: "string", short: "m" }, "api-key": { type: "string" },
    cwd: { type: "string" }, prompt: { type: "string", short: "p" }, output: { type: "string", short: "o" },
    resume: { type: "string", short: "r" }, context: { type: "string" }, "max-tokens": { type: "string" },
    "max-turns": { type: "string" }, "no-auto-compact": { type: "boolean" }, thinking: { type: "string" }, temperature: { type: "string" },
    "request-timeout": { type: "string" }, "command-timeout": { type: "string" }, "max-output": { type: "string" },
  } });
  if (values.help || values.version) return { text: values.help ? HELP : `q3x ${version}\n` };
  if (positionals.length) throw new Error('Use -p "task" to submit a task, or run q3x interactively');
  const number = (name: "context" | "max-tokens" | "max-turns" | "request-timeout" | "command-timeout" | "max-output",
                  fallback: number, max = 10_000_000) => {
    const n = values[name] === undefined ? fallback : Number(values[name]);
    if (!Number.isSafeInteger(n) || n <= 0 || n > max) throw new Error(`--${name} must be an integer from 1 to ${max}`);
    return n;
  };
  if (values.thinking !== undefined && !["on", "off"].includes(values.thinking)) throw new Error("--thinking must be on or off");
  const temperature = values.temperature === undefined ? undefined : Number(values.temperature);
  if (temperature !== undefined && (!values.temperature!.trim() || !Number.isFinite(temperature) || temperature < 0 || temperature > 2))
    throw new Error("--temperature must be a number between 0 and 2");
  const config: Config = {
    baseUrl: values["base-url"] ?? env.Q3X_BASE_URL ?? env.OPENAI_BASE_URL ?? "http://127.0.0.1:8000/v1",
    model: values.model ?? env.Q3X_MODEL ?? "",
    apiKey: values["api-key"] ?? env.Q3X_API_KEY ?? env.OPENAI_API_KEY ?? env.QWEN_API_KEY,
    cwd: values.cwd ?? process.cwd(), context: number("context", 32768), maxTokens: number("max-tokens", 4096),
    maxTurns: number("max-turns", 32, 10000), requestTimeout: number("request-timeout", 300, 86400) * 1000,
    commandTimeout: number("command-timeout", 120, 86400) * 1000, maxOutput: number("max-output", 32768),
    autoCompact: !values["no-auto-compact"], thinking: values.thinking === undefined ? undefined : values.thinking === "on", temperature,
  };
  endpoint(config.baseUrl, "chat/completions");
  if (config.context < 2048 || config.maxTokens > config.context / 2)
    throw new Error("--context must be at least 2048 and --max-tokens at most half of --context");
  return { config, values };
}

// Escape terminal controls from model/tool output while retaining text layout.
export function terminalText(text: string): string {
  return text.replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, char => char === "\x1b" ? "␛" : "");
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const parsed = options(argv);
  if ("text" in parsed) { process.stdout.write(parsed.text!); return 0; }
  const { config, values } = parsed;
  let prompt = values.prompt;
  if (prompt === undefined && !process.stdin.isTTY) prompt = await Bun.stdin.text();
  if (prompt !== undefined && !prompt.trim()) throw new Error("Task is empty; use -p or provide input on stdin");
  const session = Session.open(values.cwd, values.resume, values.output);
  config.cwd = session.cwd;
  const display: Display = (kind, text) => {
    const stream = kind === "text" ? process.stdout : process.stderr;
    stream.write(terminalText(text));
  };
  const agent = new Agent(config, session, display);
  let controller: AbortController | undefined;
  let rl: ReturnType<typeof createInterface> | undefined;
  let terminal: TerminalInput | undefined;
  const cancel = () => {
    if (controller) controller.abort(new Error("Cancelled"));
    else rl?.close();
  };
  process.on("SIGINT", cancel);
  let terminated = false;
  const terminate = () => { terminated = true; controller?.abort(new Error("Terminated")); rl?.close(); };
  process.on("SIGTERM", terminate);
  const action = async (run: (signal: AbortSignal) => Promise<void>): Promise<boolean> => {
    controller = new AbortController();
    try { await run(controller.signal); return true; }
    catch (error) { display("status", `\nq3x: ${errorText(error)}\n`); return false; }
    finally { controller = undefined; }
  };
  try {
    if (prompt !== undefined) {
      const ok = await action(signal => agent.run(prompt, signal));
      process.stdout.write("\n");
      return terminated ? 143 : ok ? 0 : 1;
    }
    display("status", `q3x ${version} · ${config.cwd}\nBash executes with your permissions. /help for commands.\n`);
    terminal = new TerminalInput(process.stdin);
    terminal.on("error", error => { display("status", `\nq3x: ${errorText(error)}\n`); rl?.close(); });
    rl = createInterface({ input: terminal, output: process.stderr, terminal: true });
    process.stderr.write("\x1b[?2004h");
    rl.on("SIGINT", cancel);
    rl.on("close", () => controller?.abort(new Error("Input closed")));
    let multiline = "";
    rl.setPrompt("q3x> "); rl.prompt();
    for await (const line of rl) {
      if (!multiline && line.trim() === "/exit") break;
      if (!multiline && line.trim() === "/help") display("status", HELP);
      else if (!multiline && line.trim() === "/status") display("status",
        `Model: ${config.model || "auto"}\nContext: ~${agent.contextUsed()}/${config.context} tokens (${config.maxTokens} reserved)\n` +
        `Session: ${session.file ?? "in memory"}\n`);
      else if (!multiline && line.trim() === "/compact") await action(signal => agent.compact(signal));
      else if (!multiline && line.trim() === "/continue") await action(signal => agent.run(undefined, signal));
      else if (!multiline && line.startsWith("/")) display("status", "Unknown command; use /help\n");
      else if (line.endsWith("\\")) { multiline += terminal.expand(line.slice(0, -1)) + "\n"; }
      else if ((multiline + line).trim()) {
        const task = multiline + terminal.expand(line); multiline = "";
        await action(signal => agent.run(task, signal));
        process.stdout.write("\n");
      }
      if (terminated) break;
      rl.setPrompt(multiline ? "...> " : "q3x> "); rl.prompt();
    }
    return terminated ? 143 : 0;
  } finally {
    rl?.close();
    if (terminal) { process.stderr.write("\x1b[?2004l"); terminal.close(); }
    session.close();
    process.off("SIGINT", cancel); process.off("SIGTERM", terminate);
  }
}

if (import.meta.main) {
  try { process.exitCode = await main(); }
  catch (error) { process.stderr.write(`q3x: ${terminalText(errorText(error))}\n`); process.exitCode = 1; }
}
