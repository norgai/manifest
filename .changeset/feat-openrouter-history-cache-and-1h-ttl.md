---
manifest: minor
---

Extend OpenRouter cache_control injection with a conversation-history breakpoint and a 1-hour TTL on the static prefix. Three breakpoints now used (Anthropic's max is 4):

- **System message — 1h TTL** (was 5m). System prompts are stable across an entire session; the 1.6× write premium pays for itself by avoiding re-writes during 5–60 minute idle gaps between user turns.
- **Last tool definition — 1h TTL** (was 5m). Tool catalogs are equally stable.
- **Last conversation message — 5m TTL** (new). Mirrors the direct-Anthropic path's history breakpoint added previously. Each turn writes only the small delta (D tokens) at 1.25× the input rate, but reads the entire prior history at 0.10× — a 10× saving on the rolling prefix that compounds with conversation length.

Math: every-turn caching beats every-Nth-turn for any N > 1 because writes are charged on the delta only. Single-sliding-breakpoint captures ~90% of available savings without burning a 4th breakpoint slot. Pre-compaction, asymptotic input savings approach 90% on long conversations.

The history breakpoint handles all OpenAI-format trailing message shapes — string content, array-of-blocks, tool messages with stringified JSON (string→array conversion), assistant messages with tool_calls — and no-ops gracefully when the trailing assistant message has only `tool_calls` and no content.
