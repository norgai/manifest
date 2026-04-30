/**
 * Cache control injection utilities for providers that support
 * prompt caching via the OpenAI-compatible format (e.g. OpenRouter).
 */

const CACHE_CONTROL = { type: 'ephemeral' } as const;
const CACHE_CONTROL_1H = { type: 'ephemeral', ttl: '1h' } as const;

type CacheControlMarker = typeof CACHE_CONTROL | typeof CACHE_CONTROL_1H;

/**
 * Attach a cache_control marker to the trailing content block of a message,
 * coercing string content to a single-block array first. Null/undefined
 * content (e.g. assistant messages with only `tool_calls`) is a no-op.
 */
function ensureBlockArrayWithCache(msg: Record<string, unknown>, marker: CacheControlMarker): void {
  if (typeof msg.content === 'string') {
    msg.content = [{ type: 'text', text: msg.content, cache_control: marker }];
    return;
  }
  if (Array.isArray(msg.content)) {
    const blocks = msg.content as Array<Record<string, unknown>>;
    if (blocks.length > 0) {
      blocks[blocks.length - 1].cache_control = marker;
    }
  }
}

/**
 * Walk backward through messages and attach a cache_control marker to the
 * trailing block of the last non-system, non-developer message. This is the
 * conversation-history breakpoint — the rolling prefix (system + tools +
 * prior turns) becomes a cache hit on the next call.
 *
 * Uses the default 5m TTL because history changes every turn, so longer TTLs
 * add no value (the entry would be superseded before its extra life pays off).
 */
function markLastConversationMessage(messages: Array<Record<string, unknown>>): void {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'system' || msg.role === 'developer') continue;
    ensureBlockArrayWithCache(msg, CACHE_CONTROL);
    return;
  }
}

/**
 * Injects cache_control breakpoints into an OpenAI-format body for
 * OpenRouter requests targeting Anthropic models. OpenRouter passes
 * these through to the Anthropic backend.
 *
 * Three breakpoints (Anthropic supports up to 4):
 * - Last system/developer message — 1h TTL (system prompts are stable)
 * - Last tool definition — 1h TTL (tool catalog is stable)
 * - Last conversation message — default 5m TTL (history rolls every turn)
 */
export function injectOpenRouterCacheControl(body: Record<string, unknown>): void {
  const messages = body.messages as Array<Record<string, unknown>> | undefined;
  if (!messages || messages.length === 0) return;

  // Find the last system/developer message and inject 1h cache_control on its content
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== 'system' && msg.role !== 'developer') continue;
    ensureBlockArrayWithCache(msg, CACHE_CONTROL_1H);
    break;
  }

  // Inject 1h cache_control on the last tool definition
  const tools = body.tools as Array<Record<string, unknown>> | undefined;
  if (tools && tools.length > 0) {
    tools[tools.length - 1].cache_control = CACHE_CONTROL_1H;
  }

  // Inject 5m cache_control on the trailing conversation message
  markLastConversationMessage(messages);
}
