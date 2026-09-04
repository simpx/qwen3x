import { type Completion, type Config, type Display, type Message, type ToolCall, type Usage, object } from "./types.ts";

export class ContextError extends Error {}

// SSE events can span TCP chunks, UTF-8 code points, and multiple data: lines.
export async function* sse(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = "", data: string[] = [], eventSize = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      let end: number;
      while ((end = pending.search(/[\r\n]/)) >= 0) {
        if (pending[end] === "\r" && end === pending.length - 1 && !done) break;
        const line = pending.slice(0, end);
        const delimiter = pending[end] === "\r" && pending[end + 1] === "\n" ? 2 : 1;
        pending = pending.slice(end + delimiter);
        if (!line) {
          if (data.length) yield data.join("\n");
          data = []; eventSize = 0;
        } else if (line.startsWith("data:")) {
          const part = line.slice(5).replace(/^ /, "");
          eventSize += part.length;
          if (eventSize > 4 * 1024 * 1024) throw new Error("SSE event exceeds 4 MiB");
          data.push(part);
        }
      }
      if (pending.length > 4 * 1024 * 1024) throw new Error("SSE line exceeds 4 MiB");
      if (done) {
        if (pending.trim() || data.length) throw new Error("Incomplete SSE event at end of response");
        return;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export function endpoint(base: string, route: string): string {
  const url = new URL(base);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash)
    throw new Error("--base-url must be an HTTP(S) URL without credentials, query, or fragment");
  url.pathname = url.pathname.replace(/\/$/, "") + "/" + route;
  return url.toString();
}

function headers(config: Config): Record<string, string> {
  return { "Content-Type": "application/json", ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}) };
}

export async function discoverModel(config: Config, signal: AbortSignal): Promise<string> {
  const response = await fetch(endpoint(config.baseUrl, "models"), {
    headers: headers(config), signal: AbortSignal.any([signal, AbortSignal.timeout(config.requestTimeout)]),
  });
  if (!response.ok) throw new Error(`Model discovery returned HTTP ${response.status}; specify --model`);
  const body = await response.json();
  const models = object(body) && Array.isArray(body.data) ? body.data : [];
  if (models.length !== 1 || typeof models[0]?.id !== "string" || !models[0].id.trim())
    throw new Error("Server does not advertise exactly one model; specify --model");
  return models[0].id;
}

export async function complete(config: Config, messages: Message[], tools: unknown[], signal: AbortSignal,
                               display: Display, maxTokens = config.maxTokens): Promise<Completion> {
  const response = await fetch(endpoint(config.baseUrl, "chat/completions"), {
    method: "POST", headers: headers(config),
    signal: AbortSignal.any([signal, AbortSignal.timeout(config.requestTimeout)]),
    body: JSON.stringify({ model: config.model, messages, stream: true,
      stream_options: { include_usage: true }, max_tokens: maxTokens,
      ...(tools.length ? { tools } : {}),
      ...(config.temperature === undefined ? {} : { temperature: config.temperature }),
      ...(config.thinking === undefined ? {} : { chat_template_kwargs: { enable_thinking: config.thinking } }),
    }),
  });
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 2000);
    const error = `HTTP ${response.status}: ${detail}`;
    if ([400, 413].includes(response.status) && /context.{0,40}(exceed|length|limit|size)|too many tokens|prompt.{0,30}too long/i.test(detail))
      throw new ContextError(error);
    throw new Error(error);
  }
  if (!response.body) throw new Error("Server returned an empty response body");
  if (!response.headers.get("content-type")?.includes("text/event-stream"))
    throw new Error("Server must support streaming Chat Completions (text/event-stream)");

  const message: Message = { role: "assistant", content: "" };
  const calls = new Map<number, ToolCall>();
  let finish = "", usage: Usage | undefined, size = 0;
  for await (const event of sse(response.body)) {
    if (event.trim() === "[DONE]") break;
    const value: unknown = JSON.parse(event);
    if (!object(value)) throw new Error("Invalid completion event");
    if (value.error) throw new Error(`Server error: ${JSON.stringify(value.error)}`);
    if (object(value.usage) && Number.isSafeInteger(value.usage.prompt_tokens) && value.usage.prompt_tokens >= 0 &&
        Number.isSafeInteger(value.usage.completion_tokens) && value.usage.completion_tokens >= 0)
      usage = { prompt_tokens: value.usage.prompt_tokens, completion_tokens: value.usage.completion_tokens };
    if (!Array.isArray(value.choices)) throw new Error("Completion event has no choices array");
    for (const choice of value.choices) {
      if (!object(choice) || (choice.index !== undefined && choice.index !== 0)) continue;
      if (!object(choice.delta)) throw new Error("Completion event has no delta");
      const delta = choice.delta;
      for (const key of ["content", "reasoning_content", "reasoning"] as const) {
        if (delta[key] == null) continue;
        if (typeof delta[key] !== "string" || finish) throw new Error("Invalid text delta");
        const target = key === "content" ? "content" : "reasoning_content";
        message[target] = (message[target] ?? "") + delta[key];
        display(target === "content" ? "text" : "reasoning", delta[key]);
        size += delta[key].length;
      }
      if (delta.tool_calls !== undefined) {
        if (!Array.isArray(delta.tool_calls) || finish) throw new Error("Invalid tool call delta");
        for (const part of delta.tool_calls) {
          if (!object(part) || !Number.isSafeInteger(part.index) || part.index < 0 || part.index >= 128)
            throw new Error("Invalid tool call index");
          const call = calls.get(part.index) ?? { id: "", type: "function", function: { name: "", arguments: "" } };
          if (part.type !== undefined && part.type !== "function") throw new Error("Unsupported tool call type");
          if (part.id !== undefined) {
            if (typeof part.id !== "string") throw new Error("Invalid tool call ID");
            call.id += part.id; size += part.id.length;
          }
          if (part.function !== undefined) {
            if (!object(part.function)) throw new Error("Invalid tool function delta");
            for (const key of ["name", "arguments"] as const) {
              if (part.function[key] === undefined) continue;
              if (typeof part.function[key] !== "string") throw new Error("Invalid tool function field");
              call.function[key] += part.function[key]; size += part.function[key].length;
            }
          }
          calls.set(part.index, call);
        }
      }
      if (choice.finish_reason != null) {
        if (typeof choice.finish_reason !== "string" || finish) throw new Error("Invalid finish reason");
        finish = choice.finish_reason;
      }
    }
    if (size > 8 * 1024 * 1024) throw new Error("Completion exceeds 8 MiB");
  }
  if (!finish) throw new Error("Stream ended without a finish reason; no tools were executed");
  if (!["stop", "tool_calls"].includes(finish))
    throw new Error(`Completion ended with ${finish}; increase --max-tokens if truncated. No tools were executed`);
  if (calls.size) {
    message.tool_calls = [...calls.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call);
    const ids = new Set<string>();
    for (const call of message.tool_calls) {
      if (!call.id || !call.function.name || ids.has(call.id)) throw new Error("Incomplete or duplicate tool call");
      ids.add(call.id);
    }
    if (!message.content) message.content = null;
  } else if (finish === "tool_calls") throw new Error("Server finished tool_calls without any tool calls");
  return { message, finish, usage };
}
