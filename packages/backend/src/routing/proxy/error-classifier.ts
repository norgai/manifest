/**
 * Classify upstream LLM provider errors so the proxy can react with
 * targeted fallbacks (e.g. escalate context-overflow to a larger-window
 * model) instead of treating every 4xx the same.
 */

export type UpstreamErrorKind = 'context_overflow' | 'other';

/**
 * Patterns surfaced by Anthropic, OpenAI, Google, and the OpenRouter
 * passthrough wrapper when the prompt exceeds the model's context window.
 * Match anywhere in the body to be resilient across error envelope shapes
 * (Anthropic-direct, OpenRouter-wrapped, plain OpenAI-style).
 */
const OVERFLOW_PATTERNS: ReadonlyArray<RegExp> = [
  /prompt is too long/i,
  /context_length_exceeded/i,
  /maximum context length/i,
  /context window/i,
  /input is too long/i,
];

/**
 * Returns 'context_overflow' iff the upstream returned a client error (4xx)
 * AND the body contains a known context-overflow pattern. 5xx and 2xx are
 * always 'other' — server failures and successful responses do not carry
 * reliable overflow signals.
 */
export function classifyUpstreamError(status: number, body: string): UpstreamErrorKind {
  if (status < 400 || status >= 500) return 'other';
  if (!body) return 'other';
  return OVERFLOW_PATTERNS.some((re) => re.test(body)) ? 'context_overflow' : 'other';
}
