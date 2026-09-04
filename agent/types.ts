export type ToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

export type Message = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  reasoning_content?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
};

export type Usage = { prompt_tokens: number; completion_tokens: number };
export type Completion = { message: Message; finish: string; usage?: Usage };
export type Display = (kind: "text" | "reasoning" | "tool" | "status", text: string) => void;

export type Config = {
  baseUrl: string;
  model: string;
  apiKey?: string;
  cwd: string;
  context: number;
  maxTokens: number;
  maxTurns: number;
  requestTimeout: number;
  commandTimeout: number;
  maxOutput: number;
  autoCompact: boolean;
  thinking?: boolean;
  temperature?: number;
};

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function object(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function validMessage(value: unknown): value is Message {
  if (!object(value) || !["system", "user", "assistant", "tool"].includes(value.role)) return false;
  if (typeof value.content !== "string" && value.content !== null) return false;
  if (value.reasoning_content !== undefined && typeof value.reasoning_content !== "string") return false;
  if (value.role === "tool" && typeof value.tool_call_id !== "string") return false;
  if (value.tool_calls !== undefined) {
    if (value.role !== "assistant" || !Array.isArray(value.tool_calls) || !value.tool_calls.length) return false;
    const ids = new Set();
    for (const call of value.tool_calls) {
      if (!object(call) || typeof call.id !== "string" || !call.id || ids.has(call.id) ||
          call.type !== "function" || !object(call.function) ||
          typeof call.function.name !== "string" || !call.function.name ||
          typeof call.function.arguments !== "string") return false;
      ids.add(call.id);
    }
  }
  return true;
}
