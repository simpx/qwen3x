// Opt-in real-model acceptance: bun run eval/daily.ts --binary ./build/q3x --out ./build/daily
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { parseArgs } from "node:util";

type Task = { id: string; files: Record<string, string>; prompt: string; allowed?: string[];
  change?: Record<string, string>; check: (answer: string) => boolean; tests?: boolean };
const tasks: Task[] = [
  { id: "inspect-flow", files: { "pipeline.py": `import json
def parse_packet(raw):
    value = json.loads(raw)
    if not isinstance(value, dict): raise ValueError('object required')
    return value
def normalize_request(value):
    if not isinstance(value.get('values'), list): raise ValueError('values required')
    return value['op'], value['values']
def dispatch(op, values):
    if op == 'sum': return sum(values)
    raise KeyError(op)
def encode_response(status, value):
    return status, json.dumps(value)
def serve_once(raw):
    try:
        op, values = normalize_request(parse_packet(raw))
        result = dispatch(op, values)
    except ValueError as error: return encode_response(400, {'error': str(error)})
    except KeyError as error: return encode_response(404, {'error': str(error)})
    return encode_response(200, {'result': result})
` }, prompt: "Read pipeline.py. Do not modify files. Explain serve_once's call path and error mapping. Return only JSON with path (function names in execution order for success), invalid_input_status, unknown_operation_status and success_status.",
    check: text => { const v = json(text); return JSON.stringify(v.path) === JSON.stringify(["serve_once", "parse_packet", "normalize_request", "dispatch", "encode_response"]) && v.invalid_input_status === 400 && v.unknown_operation_status === 404 && v.success_status === 200; } },
  { id: "inspect-config", files: { "config.json": '{"timeout":9,"retries":2,"endpoint":"/api/v2"}\n', "settings.py": `import json
from pathlib import Path
DEFAULTS = {'timeout': 30, 'retries': 5, 'endpoint': '/api/v1'}
def load(env):
    result = DEFAULTS | json.loads(Path('config.json').read_text())
    if 'TIMEOUT' in env: result['timeout'] = int(env['TIMEOUT'])
    return result
` }, prompt: "Read settings.py and config.json without modifying files. What exactly does load({'TIMEOUT':'12'}) return? Return only that JSON object.",
    check: text => { const v = json(text); return v.timeout === 12 && v.retries === 2 && v.endpoint === "/api/v2"; } },
  { id: "review-math", files: { "calculator.py": `def safe_divide(a, b):
    if b == 0: raise ValueError('zero denominator')
    return a / b
def average(values):
    if not values: raise ValueError('empty')
    return sum(values) / len(values)
def clamp(value, lower, upper):
    if lower > upper: raise ValueError('reversed bounds')
    return max(lower, min(value, upper))
` }, change: { "calculator.py": `def safe_divide(a, b):
    if b == 0: return 0.0
    return a / b
def average(values):
    if not values: raise ValueError('empty')
    return sum(values) // len(values)
def clamp(value, lower, upper):
    if lower > upper: raise ValueError('reversed bounds')
    return min(lower, max(value, upper))
` }, prompt: "Review the current git diff for correctness regressions. Do not modify files. Return an English JSON array of findings, each with function and explanation. Include a concrete counterexample for each issue.",
    check: text => { const findings = json(text); return Array.isArray(findings) && ["safe_divide", "average", "clamp"].every(name => findings.some(v => v.function === name && typeof v.explanation === "string" && v.explanation.length > 30)); } },
  { id: "review-pagination", files: { "pages.py": `def page_count(items, size):
    if size <= 0: raise ValueError('size')
    return (items + size - 1) // size
def slice_page(items, index, size):
    return items[index * size:(index + 1) * size]
` }, change: { "pages.py": `def page_count(items, size):
    if size <= 0: raise ValueError('size')
    return items // size
def slice_page(items, index, size):
    return items[index * size:(index + 1) * size - 1]
` }, prompt: "Review the current git diff without modifying any file. Find both correctness regressions. Return an English JSON array with function, explanation and a concrete counterexample for each finding.",
    check: text => { const findings = json(text); return Array.isArray(findings) && ["page_count", "slice_page"].every(name => findings.some(v => v.function === name && typeof v.explanation === "string" && v.explanation.length > 30)); } },
  { id: "fix-retry", allowed: ["retry.py"], tests: true, files: {
    "retry.py": `def retry_delays(attempts, base):
    if attempts < 0 or base <= 0: raise ValueError('invalid')
    return [base * 2 ** i for i in range(1, attempts + 1)]
`, "test_retry.py": `import unittest
from retry import retry_delays
class RetryTest(unittest.TestCase):
    def test_first(self): self.assertEqual(retry_delays(4, .5), [.5, 1, 2, 4])
    def test_zero(self): self.assertEqual(retry_delays(0, 1), [])
    def test_invalid(self):
        with self.assertRaises(ValueError): retry_delays(-1, 1)
` }, prompt: "Fix retry_delays so the first delay equals base and subsequent delays double. Only modify retry.py. Run python3 -m unittest -v. Summarize the actual change and test results.", check: text => /test|pass/i.test(text) },
  { id: "fix-csv", allowed: ["totals.py"], tests: true, files: {
    "totals.py": `def total_amount(text):
    lines = text.strip().splitlines()[1:]
    return sum(int(line.split(',')[1]) for line in lines)
`, "test_totals.py": `import unittest
from totals import total_amount
class TotalsTest(unittest.TestCase):
    def test_decimal(self): self.assertAlmostEqual(total_amount('name,amount\\na,1.25\\nb,2.50\\n'), 3.75)
    def test_quoted(self): self.assertAlmostEqual(total_amount('name,amount\\n"x,y",3.5\\n'), 3.5)
    def test_empty(self): self.assertEqual(total_amount('name,amount\\n'), 0)
` }, prompt: "Fix total_amount to correctly parse CSV including quoted commas, decimal amounts and header-only input. Only modify totals.py. Use Python's standard library. Run python3 -m unittest -v and report the results.", check: text => /test|pass/i.test(text) },
];

function json(text: string): any {
  return JSON.parse(text.trim().replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, ""));
}
function files(directory: string, prefix = ""): Record<string, string> {
  const result: Record<string, string> = {};
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if ([".git", "__pycache__"].includes(entry.name)) continue;
    const name = prefix + entry.name, path = join(directory, entry.name);
    if (entry.isDirectory()) Object.assign(result, files(path, name + "/"));
    else result[name] = createHash("sha256").update(readFileSync(path)).digest("hex");
  }
  return result;
}
async function exec(cmd: string[], cwd: string) {
  const child = Bun.spawn(cmd, { cwd, stdout: "pipe", stderr: "pipe", stdin: "ignore" });
  const timer = setTimeout(() => child.kill("SIGTERM"), 240000);
  try {
    const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
    return { stdout, stderr, code };
  } finally { clearTimeout(timer); }
}
const { values } = parseArgs({ options: { binary: { type: "string" }, out: { type: "string" },
  url: { type: "string" }, model: { type: "string" }, runs: { type: "string", default: "3" },
  thinking: { type: "string", default: "off" }, only: { type: "string" }, temperature: { type: "string" } } });
if (!values.binary || !values.out) throw new Error("Required: --binary PATH --out NEW_DIRECTORY");
const binary = resolve(values.binary), out = resolve(values.out), url = values.url ?? "http://127.0.0.1:8000/v1";
const runs = Number(values.runs);
if (!Number.isSafeInteger(runs) || runs < 1 || runs > 10 || !["on", "off"].includes(values.thinking!)) throw new Error("Invalid runs/thinking");
mkdirSync(out); // Refuse overwriting a previous acceptance run.
const results: any[] = [];
const report = { binary_sha256: createHash("sha256").update(readFileSync(binary)).digest("hex"),
  bun: Bun.version, platform: `${process.platform}/${process.arch}`, url, thinking: values.thinking,
  context: 32768, max_tokens: 4096, max_turns: 16, sampling: values.temperature ?? "server defaults", results };
for (const task of tasks.filter(t => !values.only || values.only.split(",").includes(t.id))) {
  for (let repeat = 1; repeat <= runs; repeat++) {
    const id = `${task.id}-${repeat}`, runDir = join(out, id), cwd = join(runDir, "work");
    mkdirSync(cwd, { recursive: true });
    for (const [name, content] of Object.entries(task.files)) writeFileSync(join(cwd, name), content);
    writeFileSync(join(cwd, "AGENTS.md"), "Work only in this directory. Follow the user's requested modification scope.\n");
    if (task.change) {
      for (const args of [["git", "init", "-q"], ["git", "add", "."]]) {
        const r = await exec(args, cwd); if (r.code) throw new Error(r.stderr);
      }
      for (const [name, content] of Object.entries(task.change)) writeFileSync(join(cwd, name), content);
    }
    const before = files(cwd), start = performance.now();
    writeFileSync(join(runDir, "prompt.txt"), task.prompt);
    const file = join(runDir, "session.jsonl");
    const result = await exec([binary, "--base-url", url, ...(values.model ? ["--model", values.model] : []),
      "--thinking", values.thinking!, "--max-turns", "16", "--request-timeout", "120", "--command-timeout", "30",
      ...(values.temperature === undefined ? [] : ["--temperature", values.temperature]),
      "-o", file, "-p", task.prompt], cwd);
    const seconds = (performance.now() - start) / 1000;
    writeFileSync(join(runDir, "stdout.txt"), result.stdout); writeFileSync(join(runDir, "stderr.txt"), result.stderr);
    const events = readFileSync(file, "utf8").trim().split("\n").map(line => JSON.parse(line));
    const final = events.findLast(e => e.message?.role === "assistant")?.message.content ?? "";
    const after = files(cwd), changed = [...new Set([...Object.keys(before), ...Object.keys(after)])].filter(name => before[name] !== after[name]);
    let answerOK = false; try { answerOK = task.check(final); } catch {}
    const scopeOK = changed.every(name => task.allowed?.includes(name));
    let testsOK = true;
    if (task.tests) {
      const verify = await exec(["python3", "-m", "unittest", "-v"], cwd);
      writeFileSync(join(runDir, "verification.txt"), verify.stdout + verify.stderr); testsOK = verify.code === 0;
      testsOK &&= events.some(e => e.message?.tool_calls?.some((call: any) => call.function.arguments.includes("unittest")));
    }
    const usage = events.filter(e => e.type === "usage");
    const summary = { id, passed: result.code === 0 && answerOK && scopeOK && testsOK,
      code: result.code, answerOK, scopeOK, testsOK, changed, seconds,
      model: events.findLast(e => e.type === "turn_end" && e.model)?.model,
      tools: events.filter(e => e.type === "tool_start").length,
      requests: events.filter(e => e.message?.role === "assistant").length,
      prompt_tokens: usage.reduce((n, e) => n + e.usage.prompt_tokens, 0),
      completion_tokens: usage.reduce((n, e) => n + e.usage.completion_tokens, 0), final };
    results.push(summary); writeFileSync(join(out, "results.json"), JSON.stringify(report, null, 2) + "\n");
    console.log(`${summary.passed ? "PASS" : "FAIL"} ${id} ${seconds.toFixed(1)}s ${summary.tools} tools`);
  }
}
if (!results.length || results.some(result => !result.passed)) process.exitCode = 1;
