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
 * Parse one tool-call segment (`[name][:id][{json}]`, with or without an
 * `<|tool_call_argument_begin|>` separator between name and args). Tolerates
 * the structural defects described in the file header. Returns null when
 * the segment is empty or has neither a name token nor JSON args (e.g. an
 * empty `<|tool_call_begin|><|tool_call_end|>` sentinel pair).
 */
function parseSegment(
  segmentBody: string,
  isComplete: boolean,
  knownTools: string[],
): KimiToolCall | null {
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

  // Skip empty segments — Chutes emits a leading <|begin|><|end|> sentinel
  // pair before the actual call list which would otherwise produce a
  // bogus name="unknown" call.
  if (!nameToken && !argsRaw) return null;

  // Truncation is determined by whether the args were successfully parsed,
  // not by the presence of an outer terminator. A call with name+args is
  // complete even if the section terminator never arrived; a call with
  // only a name (no JSON brace) is truncated regardless.
  let args: unknown = {};
  let truncated = false;
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
  } else {
    // Name only, no args at all.
    truncated = true;
  }
  // If the parser was told the segment is incomplete (no terminator) AND we
  // didn't get usable args, surface as truncated.
  if (!isComplete && truncated === false && !argsRaw) truncated = true;

  // Defect #3: reconstruct mangled function name.
  const reconstructed = reconstructFunctionName(nameToken, knownTools);
  let name: string;
  let callId: string;
  if (reconstructed) {
    name = `${reconstructed.namespace}.${reconstructed.name}`;
    callId = reconstructed.callId ?? `kimi_${randomUUID().slice(0, 8)}`;
  } else {
    name = nameToken || 'unknown';
    callId = `kimi_${randomUUID().slice(0, 8)}`;
  }

  return { id: callId, name, arguments: args, truncated };
}

/**
 * Locate the first occurrence of any Kimi delimiter in `text`. Returns the
 * index plus which delimiter matched, or null when none are present.
 */
function findFirstDelimiter(text: string, fromIndex = 0): { index: number; token: string } | null {
  let bestIdx = -1;
  let bestToken = '';
  for (const token of [BEGIN, ARG_BEGIN, END]) {
    const idx = text.indexOf(token, fromIndex);
    if (idx >= 0 && (bestIdx < 0 || idx < bestIdx)) {
      bestIdx = idx;
      bestToken = token;
    }
  }
  return bestIdx < 0 ? null : { index: bestIdx, token: bestToken };
}

/**
 * Extract all Kimi tool-call envelopes from `text`.
 *
 * Two on-the-wire shapes are accepted:
 *   - **Canonical**: `<|tool_call_begin|> name <|tool_call_argument_begin|>
 *     {json} <|tool_call_end|>` — one call per BEGIN/END envelope, with
 *     `<|tool_call_argument_begin|>` separating name from args.
 *   - **Chutes / OpenRouter**: a leading `<|tool_call_begin|><|tool_call_end|>`
 *     sentinel pair, followed by bare `name{json}<|tool_call_end|>` per
 *     call, terminated by a final `<|tool_call_argument_begin|>` (or
 *     end-of-buffer on truncation).
 *
 * The parser uses `<|tool_call_end|>` as the primary call terminator.
 * `<|tool_call_argument_begin|>` is a within-call separator in canonical
 * shape and a section terminator in Chutes shape. `<|tool_call_begin|>`
 * is treated as a section opener and never terminates a call.
 *
 * Returns the structured calls plus `text` with all envelope content (and
 * their delimiters) removed. Idempotent for inputs without delimiters.
 */
export function parseKimiToolCallEnvelope(text: string, knownTools: string[] = []): ParsedEnvelope {
  const firstDelim = findFirstDelimiter(text);
  if (!firstDelim) return { calls: [], cleanText: text };

  const cleanText = text.slice(0, firstDelim.index);
  const calls: KimiToolCall[] = [];

  // Skip leading BEGIN tokens (canonical envelope opener / Chutes sentinel).
  let cursor = firstDelim.index;
  while (text.startsWith(BEGIN, cursor)) cursor += BEGIN.length;

  while (cursor < text.length) {
    // Each call ends at the next `<|tool_call_end|>`, the trailing
    // `<|tool_call_argument_begin|>` section terminator, or end-of-buffer.
    const endIdx = text.indexOf(END, cursor);
    let segmentEnd: number;
    let isComplete: boolean;

    if (endIdx >= 0) {
      segmentEnd = endIdx;
      isComplete = true;
    } else {
      // No END remains. Strip a trailing ARG_BEGIN section terminator if
      // present and treat the section as complete; otherwise treat as
      // truncated.
      const argEndsSection = text.endsWith(ARG_BEGIN);
      segmentEnd = argEndsSection ? text.length - ARG_BEGIN.length : text.length;
      isComplete = argEndsSection;
    }

    const segmentBody = text.slice(cursor, segmentEnd);
    const call = parseSegment(segmentBody, isComplete, knownTools);
    if (call) calls.push(call);

    if (endIdx < 0) break;
    cursor = endIdx + END.length;
    // Skip any consecutive BEGIN tokens (defensive: nested sentinels).
    while (text.startsWith(BEGIN, cursor)) cursor += BEGIN.length;
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
 * Locate the LAST occurrence of any Kimi delimiter in `text`. Returns the
 * index of the delimiter plus its end (start + token length), or null if
 * no delimiters are present.
 */
function findLastDelimiter(text: string): { start: number; end: number } | null {
  let bestStart = -1;
  let bestEnd = -1;
  for (const token of [BEGIN, ARG_BEGIN, END]) {
    const idx = text.lastIndexOf(token);
    if (idx > bestStart) {
      bestStart = idx;
      bestEnd = idx + token.length;
    }
  }
  return bestStart < 0 ? null : { start: bestStart, end: bestEnd };
}

/**
 * Drain a streaming buffer up to the last safe boundary.
 *
 *   • Plain text before the first delimiter is emitted as `cleanText`.
 *   • Any tail that could be a partial delimiter (`<|tool_ca` straddling a
 *     chunk boundary) is held back.
 *   • Once a delimiter is seen, the region between the first and the last
 *     delimiter contains zero or more *complete* tool-call segments — those
 *     parse and emit immediately.
 *   • The text after the last delimiter is held back as the in-flight
 *     segment, since its terminating delimiter has not yet arrived.
 */
function drainSafeBuffer(buffer: string, knownTools: string[]): DrainResult {
  const firstDelim = findFirstDelimiter(buffer);
  if (!firstDelim) {
    const partialLen = partialDelimiterSuffixLength(buffer);
    return {
      cleanText: buffer.slice(0, buffer.length - partialLen),
      calls: [],
      remaining: buffer.slice(buffer.length - partialLen),
    };
  }

  const cleanText = buffer.slice(0, firstDelim.index);
  const lastDelim = findLastDelimiter(buffer)!;

  // Everything between the first and last delimiter inclusive forms zero or
  // more complete segments. After the last delimiter, the (possibly empty)
  // tail is the in-flight segment whose terminator we haven't seen yet.
  const completeRegion = buffer.slice(firstDelim.index, lastDelim.end);
  const tail = buffer.slice(lastDelim.end);
  const { calls } = parseKimiToolCallEnvelope(completeRegion, knownTools);

  // Hold the trailing delimiter + any in-flight segment text so the next
  // chunk can complete it (or a follow-up flush can finalize a truncated
  // call). If `tail` is empty we still hold the delimiter so the next
  // chunk knows it sits inside a tool-call region.
  const remaining = buffer.slice(lastDelim.start);
  return { cleanText, calls, remaining };
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
 * Post-process a non-streaming OpenAI ChatCompletion response. Scans both
 * `message.content` and `message.reasoning` (Kimi via OpenRouter/Chutes
 * routes the model's native output through the reasoning channel) for
 * Kimi delimiter envelopes, extracts them as `tool_calls[]`, and writes
 * back the cleaned text. No-op when no envelopes are present.
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

    const fields: Array<'content' | 'reasoning'> = ['content', 'reasoning'];
    const allCalls: KimiToolCall[] = [];
    const cleaned: Partial<Record<'content' | 'reasoning', string | null>> = {};
    let foundEnvelope = false;

    for (const field of fields) {
      const value = message[field];
      if (typeof value !== 'string' || !value.includes(BEGIN)) continue;
      const { calls, cleanText } = parseKimiToolCallEnvelope(value, knownTools);
      if (calls.length === 0) continue;
      foundEnvelope = true;
      allCalls.push(...calls);
      const trimmed = cleanText.trim();
      cleaned[field] = trimmed.length > 0 ? trimmed : null;
    }

    if (!foundEnvelope) return choice;
    mutated = true;

    const existing = (message.tool_calls as Array<Record<string, unknown>>) || [];
    const newToolCalls = allCalls.map((call) => ({
      id: call.id,
      type: 'function',
      function: {
        name: call.name,
        arguments: JSON.stringify(call.arguments),
      },
    }));

    return {
      ...choice,
      message: {
        ...message,
        ...cleaned,
        tool_calls: [...existing, ...newToolCalls],
      },
      finish_reason: 'tool_calls',
    };
  });

  return mutated ? { ...resp, choices: newChoices } : resp;
}

/* ── Stream conversion ── */

function makeContentDelta(model: string, content: string, field: 'content' | 'reasoning'): string {
  const delta: Record<string, string> = {};
  delta[field] = content;
  return `data: ${JSON.stringify({
    id: `chatcmpl-${randomUUID()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta, finish_reason: null }],
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
 * Create a stateful stream transformer. Each call receives a single
 * already-parsed SSE event payload (the JSON body of one `data: ...` line —
 * `pipeStream` strips the `data: ` prefix and the `[DONE]` sentinel before
 * invoking the transform). The transformer returns a raw SSE-formatted chunk
 * (with `data: ` prefix and trailing `\n\n`) ready for the wire, or null to
 * indicate the event was buffered and produced no output.
 *
 * Behavior:
 *   - Buffers `choices[0].delta.content` text across events.
 *   - Emits content deltas once the trailing buffer cannot be a partial
 *     Kimi delimiter (held-back suffix is small — at most `<|tool_call_argument_begin|>` length).
 *   - When a complete `<|tool_call_begin|>...<|tool_call_end|>` envelope is
 *     buffered, emits a `tool_calls[]` delta in OpenAI streaming shape and
 *     resumes content emission for any text after the envelope.
 *   - Passes through non-content events (role-only, finish_reason, usage)
 *     as-is. When `finish_reason` arrives with content still buffered, the
 *     buffer is force-flushed first so any envelope completes before the
 *     end-of-message signal.
 */
export function createKimiStreamTransformer(
  model: string,
  knownTools: string[] = [],
): (chunk: string) => string | null {
  // Separate buffers per delta field. Kimi via OpenRouter/Chutes emits its
  // entire native output (including tool-call delimiters) through
  // `delta.reasoning`; direct Moonshot endpoints emit through `delta.content`.
  // Buffer each independently so envelopes that span multiple chunks within
  // the same channel are recovered without crossing channels.
  const buffers: Record<'content' | 'reasoning', string> = { content: '', reasoning: '' };
  let toolCallIndex = 0;

  const flushBuffer = (field: 'content' | 'reasoning'): string => {
    const buf = buffers[field];
    if (buf.length === 0) return '';
    const { cleanText, calls } = parseKimiToolCallEnvelope(buf, knownTools);
    buffers[field] = '';
    let out = '';
    if (cleanText) out += makeContentDelta(model, cleanText, field);
    for (const call of calls) {
      out += makeToolCallDelta(model, toolCallIndex++, call);
    }
    return out;
  };

  const flushAll = (): string => flushBuffer('content') + flushBuffer('reasoning');

  const processField = (field: 'content' | 'reasoning', text: string): string => {
    buffers[field] += text;
    const drained = drainSafeBuffer(buffers[field], knownTools);
    let out = '';
    if (drained.cleanText) out += makeContentDelta(model, drained.cleanText, field);
    for (const call of drained.calls) {
      out += makeToolCallDelta(model, toolCallIndex++, call);
    }
    buffers[field] = drained.remaining;
    return out;
  };

  return (chunk: string): string | null => {
    const trimmed = chunk.trim();
    if (!trimmed) return null;

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      // Pass through unparseable events untouched.
      return `data: ${trimmed}\n\n`;
    }

    const choices = parsed.choices as Array<Record<string, unknown>> | undefined;
    if (!choices || choices.length === 0) {
      return `data: ${JSON.stringify(parsed)}\n\n`;
    }

    const choice = choices[0];
    const delta = choice.delta as Record<string, unknown> | undefined;
    const finishReason = choice.finish_reason as string | null | undefined;
    const contentDelta = delta?.content;
    const reasoningDelta = delta?.reasoning;

    const hasContent = typeof contentDelta === 'string' && contentDelta.length > 0;
    const hasReasoning = typeof reasoningDelta === 'string' && reasoningDelta.length > 0;

    let out = '';

    if (hasContent) out += processField('content', contentDelta);
    if (hasReasoning) out += processField('reasoning', reasoningDelta);

    if (hasContent || hasReasoning) {
      // If this event also carried a finish_reason, force-flush both buffers
      // before re-emitting the terminal event so any held-back partial
      // envelope surfaces before the end-of-message signal.
      if (finishReason) {
        out += flushAll();
        const stripped = {
          ...parsed,
          choices: choices.map((c) => ({ ...c, delta: {}, finish_reason: finishReason })),
        };
        out += `data: ${JSON.stringify(stripped)}\n\n`;
      }
      return out.length > 0 ? out : null;
    }

    // Non-content event. Flush both buffers if a finish_reason is arriving so
    // any pending envelope completes before the end-of-message signal.
    if (finishReason && (buffers.content.length > 0 || buffers.reasoning.length > 0)) {
      out += flushAll();
    }
    out += `data: ${JSON.stringify(parsed)}\n\n`;
    return out;
  };
}

/** Stateless convenience for single-chunk tests. */
export function transformKimiStreamChunk(chunk: string, model: string): string | null {
  return createKimiStreamTransformer(model)(chunk);
}
