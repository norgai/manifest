---
manifest: patch
---

On a `context_overflow` upstream error (e.g. Anthropic 400 `prompt is too long`), the proxy now re-resolves at the `reasoning` tier and prepends that model to the fallback chain so the request transparently completes on a larger-context model. Recorded with `routing_tier='reasoning'` and `routing_reason='context_overflow_escalation'` for queryable observability. Falls through to normal tier fallbacks when already on `reasoning` or when reasoning has no configured model.
