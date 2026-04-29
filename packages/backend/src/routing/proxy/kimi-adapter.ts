/**
 * Extracts Kimi/Moonshot tool-call delimiter envelopes from OpenAI-format
 * responses and converts them into proper OpenAI `tool_calls[]` arrays.
 *
 * Kimi (Moonshot K2.5/K2.6) emits tool calls inline in the response content
 * using the delimiter syntax:
 *
 *   <|tool_call_begin|>functions.{namespace}.{name}:{call_id}
 *   <|tool_call_argument_begin|>{json args}
 *   <|tool_call_end|>
 *
 * When routed via OpenRouter (which is the production setup for the
 * `manifest/auto` standard tier), OpenClaw's runtime Kimi parser does NOT
 * activate because it gates on `endpoint === 'moonshot'` directly. The
 * delimiters reach the agent as plain text in `choices[*].message.content`
 * and tool calls never execute. This adapter is response-only: requests
 * pass through unchanged; only responses are post-processed.
 *
 * Three structural defects observed in real production output are handled:
 *
 *   1. Missing `<|tool_call_argument_begin|>` separator — the function-name
 *      token and JSON args are glued together. Recovered by greedy-finding
 *      the first `{` in the segment.
 *   2. Truncation — `<|tool_call_end|>` never arrives because the model hit
 *      a turn token budget mid-emission. The partial call is still extracted
 *      with `truncated: true` and `arguments: {}`.
 *   3. Mangled function names — non-alphanumeric characters (dots, dashes,
 *      underscores) and the `:` call_id separator get stripped, producing
 *      tokens like `functionscontentcraftgetslugmasterentries3` instead of
 *      `functions.content-craft.get_slug_master_entries:3`. Reconstructed
 *      via lowercase-alphanumeric suffix match against the request's
 *      registered tool list.
 *
 * Idempotent: text without any `<|tool_call_begin|>` returns unchanged, so
 * the adapter is a safe no-op on responses that don't actually contain
 * Kimi tool-call envelopes.
 */
import { randomUUID } from 'crypto';

const BEGIN = '<|tool_call_begin|>';
const ARG_BEGIN = '<|tool_call_argument_begin|>';
const END = '<|tool_call_end|>';

/** A tool call extracted from a Kimi delimiter envelope. */
export interface KimiToolCall {
  id: string;
  name: string;
  arguments: unknown;
  truncated: boolean;
}

/** Result of parsing a buffer of text that may contain Kimi envelopes. */
export interface ParsedEnvelope {
  calls: KimiToolCall[];
  cleanText: string;
}

/* ── Helpers ── */

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

/**
 * Walk `text` looking for the first `{` that is not inside a JSON string
 * literal. Kimi function-name tokens are alphanumeric with dots/dashes/
 * colons only, so the first `{` reliably marks the start of the JSON args.
 */
function findUnquotedBraceStart(text: string): number {
  let inString = false;
  let escape = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === '\\') {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (!inString && ch === '{') return i;
  }
  return -1;
}

/**
 * If `text` starts with `{`, walk forward and return the slice up to the
 * matching close brace plus the number of input chars consumed. Returns
 * null if the JSON object is unclosed (truncated stream / mid-flight).
 */
function balancedJsonExtract(text: string): { json: string; consumed: number } | null {
  if (!text.startsWith('{')) return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === '\\') {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return { json: text.slice(0, i + 1), consumed: i + 1 };
    }
  }
  return null;
}

/**
 * Reconstruct canonical `{namespace}.{name}` plus call_id from a mangled
 * Kimi function-name token. Uses lowercase-alphanumeric suffix matching
 * against `knownTools` (formatted as `namespace.name`, e.g.
 * `content-craft.get_slug_master_entries`). Returns null when no known
 * tool matches; the caller should pass the raw token through so the
 * runtime fails loudly rather than silently dropping the call.
 */
export function reconstructFunctionName(
  mangled: string,
  knownTools: string[],
): { namespace: string; name: string; callId: string | null } | null {
  // Strip `functions` / `functions.` prefix.
  const stripped = mangled.replace(/^functions\.?/, '');
  // Trailing digits (with optional `:`) carry the call-id.
  const callIdMatch = stripped.match(/:?(\d+)$/);
  const callId = callIdMatch ? callIdMatch[1] : null;
  const body = callIdMatch ? stripped.slice(0, -callIdMatch[0].length) : stripped;
  const normalizedBody = body.replace(/[^a-z0-9]/gi, '').toLowerCase();
  if (!normalizedBody) return null;

  // Sort by length desc so longer (more specific) tool names win first.
  const sorted = [...knownTools].sort((a, b) => b.length - a.length);
  for (const tool of sorted) {
    const dotIdx = tool.indexOf('.');
    if (dotIdx < 0) continue;
    const namespace = tool.slice(0, dotIdx);
    const name = tool.slice(dotIdx + 1);
    const normalizedTool = (namespace + name).replace(/[^a-z0-9]/gi, '').toLowerCase();
    if (normalizedBody === normalizedTool || normalizedBody.endsWith(normalizedTool)) {
      return { namespace, name, callId };
    }
  }
  return null;
}

/**
 * Parse the body of one envelope (everything between `<|tool_call_begin|>`
 * and either `<|tool_call_end|>` or end-of-input). Tolerates defects 1, 2,
 * and 3 described in the file header.
 */
function parseSegment(segmentBody: string, hasEnd: boolean, knownTools: string[]): KimiToolCall {
  const argBeginIdx = segmentBody.indexOf(ARG_BEGIN);
  const firstBraceIdx = findUnquotedBraceStart(segmentBody);

  let nameToken: string;
  let argsRaw: string;

  // If JSON args appear BEFORE the argument-begin marker (defect #1: model
  // emitted name+args glued together, then a stray argument-begin), treat
  // it as the missing-separator case. Same path applies when no
  // argument-begin marker is present at all.
  if (firstBraceIdx >= 0 && (argBeginIdx < 0 || firstBraceIdx < argBeginIdx)) {
    nameToken = segmentBody.slice(0, firstBraceIdx);
    const argsEnd = argBeginIdx >= 0 ? argBeginIdx : segmentBody.length;
    argsRaw = segmentBody.slice(firstBraceIdx, argsEnd);
  } else if (argBeginIdx >= 0) {
    // Canonical path: name<|tool_call_argument_begin|>{json}
    nameToken = segmentBody.slice(0, argBeginIdx);
    argsRaw = segmentBody.slice(argBeginIdx + ARG_BEGIN.length);
  } else {
    // No argument-begin and no JSON brace — name only.
    nameToken = segmentBody;
    argsRaw = '';
  }

  nameToken = nameToken.trim();
  argsRaw = argsRaw.trim();

  // Defect #2: tolerant JSON parsing. Truncated mid-args → empty object + flag.
  let args: unknown = {};
  let truncated = !hasEnd;
  if (argsRaw.startsWith('{')) {
    const balanced = balancedJsonExtract(argsRaw);
    if (balanced) {
      args = safeJsonParse(balanced.json);
    } else {
      args = {};
      truncated = true;
    }
  } else if (argsRaw) {
    args = safeJsonParse(argsRaw);
  }

  // Defect #3: reconstruct mangled function name.
  const reconstructed = reconstructFunctionName(nameToken, knownTools);
  let name: string;
  let callId: string;
  if (reconstructed) {
    name = `${reconstructed.namespace}.${reconstructed.name}`;
    callId = reconstructed.callId ?? `kimi_${randomUUID().slice(0, 8)}`;
  } else {
    // Pass through raw token; runtime will fail with a clear "tool not found".
    name = nameToken || 'unknown';
    callId = `kimi_${randomUUID().slice(0, 8)}`;
  }

  return { id: callId, name, arguments: args, truncated };
}

/**
 * Extract all Kimi tool-call envelopes from `text`. Returns the structured
 * calls plus the input with envelopes (and the surrounding delimiter
 * tokens) removed. Idempotent for inputs without any `<|tool_call_begin|>`.
 */
export function parseKimiToolCallEnvelope(text: string, knownTools: string[] = []): ParsedEnvelope {
  if (!text.includes(BEGIN)) return { calls: [], cleanText: text };

  const calls: KimiToolCall[] = [];
  let cleanText = '';
  let cursor = 0;

  while (cursor < text.length) {
    const beginIdx = text.indexOf(BEGIN, cursor);
    if (beginIdx < 0) {
      cleanText += text.slice(cursor);
      break;
    }
    cleanText += text.slice(cursor, beginIdx);
    const segmentStart = beginIdx + BEGIN.length;
    const endIdx = text.indexOf(END, segmentStart);
    const hasEnd = endIdx >= 0;
    const segmentBody = hasEnd ? text.slice(segmentStart, endIdx) : text.slice(segmentStart);
    calls.push(parseSegment(segmentBody, hasEnd, knownTools));
    cursor = hasEnd ? endIdx + END.length : text.length;
  }

  return { calls, cleanText };
}

/* ── Stream-buffer logic ── */

/**
 * Return the length of the longest suffix of `text` that could be a prefix
 * of any Kimi delimiter. Used by the stream transformer to hold back text
 * that might be a partial `<|tool_call_*|>` token straddling a chunk
 * boundary.
 */
function partialDelimiterSuffixLength(text: string): number {
  const delimiters = [BEGIN, ARG_BEGIN, END];
  let maxLen = 0;
  for (const delim of delimiters) {
    const limit = Math.min(text.length, delim.length - 1);
    for (let len = limit; len > 0; len--) {
      if (delim.startsWith(text.slice(text.length - len))) {
        if (len > maxLen) maxLen = len;
        break;
      }
    }
  }
  return maxLen;
}

interface DrainResult {
  cleanText: string;
  calls: KimiToolCall[];
  remaining: string;
}

/**
 * Drain a streaming buffer up to the last safe boundary. Anything before a
 * complete envelope or before a known-clean text region is emitted; a
 * partial-delimiter suffix or an unfinished envelope is held back as
 * `remaining` for the next chunk to complete.
 */
function drainSafeBuffer(buffer: string, knownTools: string[]): DrainResult {
  const calls: KimiToolCall[] = [];
  let cleanText = '';
  let cursor = 0;

  while (cursor < buffer.length) {
    const beginIdx = buffer.indexOf(BEGIN, cursor);
    if (beginIdx < 0) {
      const tail = buffer.slice(cursor);
      const partialLen = partialDelimiterSuffixLength(tail);
      cleanText += tail.slice(0, tail.length - partialLen);
      return { cleanText, calls, remaining: tail.slice(tail.length - partialLen) };
    }
    cleanText += buffer.slice(cursor, beginIdx);
    const segmentStart = beginIdx + BEGIN.length;
    const endIdx = buffer.indexOf(END, segmentStart);
    if (endIdx < 0) {
      // Envelope not closed yet — hold from the BEGIN marker onward.
      return { cleanText, calls, remaining: buffer.slice(beginIdx) };
    }
    const segmentBody = buffer.slice(segmentStart, endIdx);
    calls.push(parseSegment(segmentBody, true, knownTools));
    cursor = endIdx + END.length;
  }

  return { cleanText, calls, remaining: '' };
}

/* ── Response conversion (non-streaming) ── */

/**
 * Extract tool-call names from the request body's `tools[]` for use as the
 * known-tools list when un-mangling. Returns names in `namespace.name`
 * shape, derived from OpenAI tool definitions (`tools[i].function.name`).
 */
export function extractKnownToolNames(body: Record<string, unknown> | undefined): string[] {
  if (!body) return [];
  const tools = body.tools as Array<Record<string, unknown>> | undefined;
  if (!tools || tools.length === 0) return [];
  const out: string[] = [];
  for (const t of tools) {
    const fn = t.function as { name?: string } | undefined;
    if (fn?.name) out.push(fn.name);
  }
  return out;
}

/**
 * Post-process a non-streaming OpenAI ChatCompletion response. Scans every
 * choice's message content for Kimi delimiter envelopes, extracts them as
 * `tool_calls[]`, and writes back the cleaned text. No-op when no
 * envelopes are present.
 */
export function fromKimiResponse(
  resp: Record<string, unknown>,
  _model: string,
  knownTools: string[] = [],
): Record<string, unknown> {
  const choices = resp.choices as Array<Record<string, unknown>> | undefined;
  if (!choices || choices.length === 0) return resp;

  let mutated = false;
  const newChoices = choices.map((choice) => {
    const message = choice.message as Record<string, unknown> | undefined;
    if (!message) return choice;
    const content = message.content;
    if (typeof content !== 'string' || !content.includes(BEGIN)) return choice;

    const { calls, cleanText } = parseKimiToolCallEnvelope(content, knownTools);
    if (calls.length === 0) return choice;
    mutated = true;

    const existing = (message.tool_calls as Array<Record<string, unknown>>) || [];
    const newToolCalls = calls.map((call) => ({
      id: call.id,
      type: 'function',
      function: {
        name: call.name,
        arguments: JSON.stringify(call.arguments),
      },
    }));

    const trimmedText = cleanText.trim();
    return {
      ...choice,
      message: {
        ...message,
        content: trimmedText.length > 0 ? trimmedText : null,
        tool_calls: [...existing, ...newToolCalls],
      },
      finish_reason: 'tool_calls',
    };
  });

  return mutated ? { ...resp, choices: newChoices } : resp;
}

/* ── Stream conversion ── */

function makeContentDelta(model: string, content: string): string {
  return `data: ${JSON.stringify({
    id: `chatcmpl-${randomUUID()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  })}\n\n`;
}

function makeToolCallDelta(model: string, index: number, call: KimiToolCall): string {
  return `data: ${JSON.stringify({
    id: `chatcmpl-${randomUUID()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index,
              id: call.id,
              type: 'function',
              function: { name: call.name, arguments: JSON.stringify(call.arguments) },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  })}\n\n`;
}

/**
 * Create a stateful stream transformer. For each incoming SSE chunk, returns
 * either a transformed SSE chunk or null. The transformer:
 *
 *   - Buffers `choices[0].delta.content` text across chunks.
 *   - Emits content deltas verbatim once we know the trailing buffer can't
 *     be the prefix of a Kimi delimiter.
 *   - When a complete envelope is buffered, emits a `tool_calls[]` delta in
 *     OpenAI streaming shape and resumes content emission for any text
 *     after the envelope.
 *   - Passes through non-content chunks (usage, role, finish_reason) as-is.
 */
export function createKimiStreamTransformer(
  model: string,
  knownTools: string[] = [],
): (chunk: string) => string | null {
  let buffer = '';
  let toolCallIndex = 0;
  let pendingFinishReason: string | null = null;

  const flushBuffer = (): string => {
    if (buffer.length === 0) return '';
    // End-of-stream: any remaining envelope is treated as truncated, any
    // remaining text is emitted verbatim.
    const { cleanText, calls } = parseKimiToolCallEnvelope(buffer, knownTools);
    buffer = '';
    let out = '';
    if (cleanText) out += makeContentDelta(model, cleanText);
    for (const call of calls) {
      out += makeToolCallDelta(model, toolCallIndex++, call);
    }
    return out;
  };

  return (chunk: string): string | null => {
    const lines = chunk.split('\n');
    let out = '';
    let consumed = false;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (!payload) continue;

      if (payload === '[DONE]') {
        out += flushBuffer();
        if (pendingFinishReason) {
          // Re-emit the saved finish_reason now that all tool_calls are out.
          out += `data: ${JSON.stringify({
            id: `chatcmpl-${randomUUID()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [{ index: 0, delta: {}, finish_reason: pendingFinishReason }],
          })}\n\n`;
          pendingFinishReason = null;
        }
        out += 'data: [DONE]\n\n';
        consumed = true;
        continue;
      }

      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(payload);
      } catch {
        // Pass through unparseable lines unchanged.
        out += `data: ${payload}\n\n`;
        consumed = true;
        continue;
      }

      const choices = parsed.choices as Array<Record<string, unknown>> | undefined;
      if (!choices || choices.length === 0) {
        out += `data: ${JSON.stringify(parsed)}\n\n`;
        consumed = true;
        continue;
      }

      const choice = choices[0];
      const delta = choice.delta as Record<string, unknown> | undefined;
      const finishReason = choice.finish_reason as string | null | undefined;

      const contentDelta = delta?.content;
      if (typeof contentDelta === 'string' && contentDelta.length > 0) {
        buffer += contentDelta;
        const drained = drainSafeBuffer(buffer, knownTools);
        if (drained.cleanText) out += makeContentDelta(model, drained.cleanText);
        for (const call of drained.calls) {
          out += makeToolCallDelta(model, toolCallIndex++, call);
        }
        buffer = drained.remaining;
        consumed = true;
        continue;
      }

      // Non-content delta (role-only, tool_calls already structured, etc.) —
      // pass through, but defer finish_reason until buffer is flushed.
      if (finishReason && buffer.length > 0) {
        pendingFinishReason = finishReason;
        out += flushBuffer();
        // Re-emit the chunk without finish_reason; we'll send it after [DONE].
        const stripped = {
          ...parsed,
          choices: choices.map((c) => ({ ...c, finish_reason: null })),
        };
        out += `data: ${JSON.stringify(stripped)}\n\n`;
      } else {
        out += `data: ${JSON.stringify(parsed)}\n\n`;
      }
      consumed = true;
    }

    if (!consumed) return null;
    return out.length > 0 ? out : null;
  };
}

/** Stateless convenience for single-chunk tests. */
export function transformKimiStreamChunk(chunk: string, model: string): string | null {
  return createKimiStreamTransformer(model)(chunk);
}
