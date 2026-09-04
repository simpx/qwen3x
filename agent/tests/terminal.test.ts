import { expect, test } from "bun:test";
import { PassThrough } from "node:stream";
import type { ReadStream } from "node:tty";
import { TerminalInput } from "../terminal.ts";

function input() {
  const source = new PassThrough() as unknown as ReadStream;
  source.setRawMode = () => source;
  const terminal = new TerminalInput(source);
  let visible = "";
  terminal.on("data", bytes => { visible += bytes.toString(); });
  return { source, terminal, visible: () => visible };
}

test("paste markers and Unicode survive every wire split without submitting newlines", () => {
  const wire = Buffer.from("before \x1b[200~中文🌲\r\nsecond\x1b[201~ after");
  for (let split = 1; split < wire.length; split++) {
    const t = input();
    t.terminal.write(wire.subarray(0, split)); t.terminal.write(wire.subarray(split));
    expect(t.visible()).toBe("before [paste 1: 2 lines] after");
    expect(t.terminal.expand(t.visible())).toBe("before 中文🌲\nsecond after");
    t.terminal.close();
  }
});

test("paste byte limit applies even when the closing marker arrives in the same chunk", async () => {
  for (const split of [false, true]) {
    const t = input();
    const error = new Promise<Error>(resolve => t.terminal.once("error", resolve));
    const body = "界".repeat(400000); // 1.2 MB, below one million UTF-16 code units.
    if (split) { t.terminal.write("\x1b[200~"); t.terminal.write(body); }
    else t.terminal.write(`\x1b[200~${body}\x1b[201~`);
    expect((await error).message).toContain("1 MiB");
    expect(t.visible()).toBe("");
    t.terminal.close();
  }
}, 1000);
