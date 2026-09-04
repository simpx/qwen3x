// Real model test: repeated compaction, process reload, retained constraints and exactly one recorded effect.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { parseArgs } from "node:util";
import { Session } from "../session.ts";
import { Agent, estimate } from "../agent.ts";
import { options } from "../main.ts";

const { values } = parseArgs({ options: { out: { type: "string" }, url: { type: "string" }, model: { type: "string" } } });
if (!values.out) throw new Error("Required: --out NEW_DIRECTORY");
const out = resolve(values.out); mkdirSync(out);
const cwd = join(out, "work"); mkdirSync(cwd);
writeFileSync(join(cwd, "a.py"), "def answer():\n    return 1\n");
writeFileSync(join(cwd, "b.py"), "# deliberately unrelated; must remain byte-identical\nKEEP = 'unchanged'\n");
writeFileSync(join(cwd, "test_a.py"), "import unittest\nfrom a import answer\nclass T(unittest.TestCase):\n    def test_answer(self): self.assertEqual(answer(), 2)\n");
const protectedFile = readFileSync(join(cwd, "b.py"));
const protectedTest = readFileSync(join(cwd, "test_a.py"));
const parsed = options(["--cwd", cwd, "--base-url", values.url ?? "http://127.0.0.1:8000/v1", "--thinking", "on", "--temperature", "0.2",
  ...(values.model ? ["--model", values.model] : [])]);
if (!parsed.config) throw new Error("Missing configuration");
const cfg = parsed.config, file = join(out, "session.jsonl");
let session = Session.open(cwd, undefined, file);
const stages: any[] = [];
let output = "";
const display = (kind: string, text: string) => { output += `[${kind}]${text}`; };
try {
  await new Agent(cfg, session, display).run("Fix a.py so answer() returns 2. Among existing files only modify a.py. Leave b.py and test_a.py unchanged. " +
    "Run python3 -m unittest -v. After tests pass, append the line verified exactly once to actions.log. " +
    "Do not repeat this append in later continuation turns. Retain the identifier task-4827 for continuation.", AbortSignal.timeout(240000));
  const originalEffects = readFileSync(join(cwd, "actions.log"), "utf8");
  if (originalEffects !== "verified\n") throw new Error(`Expected exactly one effect, got ${JSON.stringify(originalEffects)}`);
  for (let round = 0; round < 3; round++) {
    // Representative old tool output; explicitly logged as such, not claimed as a live command.
    session.add({ role: "assistant", content: "Earlier diagnostic output follows.", tool_calls: [{ id: `fixture-${round}`, type: "function",
      function: { name: "bash", arguments: '{"command":"cat previous-diagnostics.log"}' } }] });
    session.add({ role: "tool", tool_call_id: `fixture-${round}`, content:
      "[Evaluation fixture: historical diagnostic output]\n" + "INFO worker processed request successfully; no changes needed.\n".repeat(120) });
    const before = estimate(session.messages);
    await new Agent(cfg, session, display).compact(AbortSignal.timeout(180000));
    stages.push({ round, before, after: estimate(session.messages) });
    session.close(); session = Session.open(undefined, file);
    await new Agent(cfg, session, display).run("Continue the previous task by verifying the tests again without modifying any file. " +
      "Report the retained task identifier and whether b.py is in scope. Do not repeat completed actions.", AbortSignal.timeout(180000));
    const final = session.messages.at(-1)?.content ?? "";
    if (!final.includes("task-4827")) throw new Error(`Lost task identifier after compaction ${round}`);
    if (!protectedFile.equals(readFileSync(join(cwd, "b.py")))) throw new Error("Modified protected b.py");
    if (!protectedTest.equals(readFileSync(join(cwd, "test_a.py")))) throw new Error("Modified protected test_a.py");
    if (readFileSync(join(cwd, "actions.log"), "utf8") !== originalEffects) throw new Error("Repeated a completed append");
    const test = Bun.spawn(["python3", "-m", "unittest", "-v"], { cwd, stdout: "pipe", stderr: "pipe" });
    const [stdout, stderr, code] = await Promise.all([new Response(test.stdout).text(), new Response(test.stderr).text(), test.exited]);
    writeFileSync(join(out, `verify-${round}.txt`), stdout + stderr);
    if (code !== 0) throw new Error("Tests failed after continuation");
    console.log(`PASS compaction/reload ${round + 1}: ${before} → ${estimate(session.messages)} estimated tokens`);
  }
  writeFileSync(join(out, "result.json"), JSON.stringify({ passed: true, model: cfg.model, thinking: true, temperature: cfg.temperature, stages }, null, 2));
} catch (error) {
  writeFileSync(join(out, "result.json"), JSON.stringify({ passed: false, error: String(error), stages }, null, 2));
  throw error;
} finally { session.close(); writeFileSync(join(out, "transcript.txt"), output); }
