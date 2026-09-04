import { Transform, type TransformCallback } from "node:stream";
import { StringDecoder } from "node:string_decoder";
import type { ReadStream } from "node:tty";

// Bracketed paste must not become a series of submitted tasks. Readline edits a
// short visible placeholder; expansion happens only after the user presses Enter.
export class TerminalInput extends Transform {
  isTTY = true;
  private decoder = new StringDecoder("utf8");
  private pending = "";
  private paste: string | undefined;
  private sequence = 0;
  private values = new Map<string, string>();

  constructor(private source: ReadStream) { super(); source.pipe(this); }
  setRawMode(value: boolean) { this.source.setRawMode(value); return this; }

  override _transform(chunk: Buffer, _encoding: BufferEncoding, done: TransformCallback) {
    this.pending += this.decoder.write(chunk);
    const start = "\x1b[200~", end = "\x1b[201~";
    while (this.pending) {
      if (this.paste !== undefined) {
        const index = this.pending.indexOf(end);
        if (index < 0) {
          // Leave a possible split end marker buffered, not the entire paste.
          const keep = Math.min(end.length - 1, this.pending.length);
          this.paste += this.pending.slice(0, -keep || undefined);
          this.pending = this.pending.slice(-keep);
          break;
        }
        const pasted = this.paste + this.pending.slice(0, index);
        if (Buffer.byteLength(pasted) > 1024 * 1024) { done(new Error("Paste exceeds 1 MiB")); return; }
        const text = pasted.replace(/\r\n?/g, "\n")
          .replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, "");
        const label = `[paste ${++this.sequence}: ${text.split("\n").length} lines]`;
        this.values.set(label, text);
        this.push(label);
        this.pending = this.pending.slice(index + end.length); this.paste = undefined;
      } else if (this.pending.startsWith(start)) {
        this.pending = this.pending.slice(start.length); this.paste = "";
      } else if (start.startsWith(this.pending)) break;
      else {
        const escape = this.pending.indexOf("\x1b", 1);
        const count = escape < 0 ? this.pending.length : escape;
        this.push(this.pending.slice(0, count)); this.pending = this.pending.slice(count);
      }
    }
    if (this.paste !== undefined && Buffer.byteLength(this.paste + this.pending) > 1024 * 1024) done(new Error("Paste exceeds 1 MiB"));
    else done();
  }

  expand(line: string): string {
    // One pass: pasted text cannot expand a different placeholder recursively.
    const result = line.replace(/\[paste \d+: \d+ lines\]/g, label => this.values.get(label) ?? label);
    this.values.clear();
    return result;
  }

  close() {
    this.source.unpipe(this);
    if (this.source.isTTY) this.source.setRawMode(false);
    this.destroy();
  }
}
