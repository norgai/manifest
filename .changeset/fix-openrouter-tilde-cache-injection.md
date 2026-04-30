---
manifest: patch
---

Fix `injectOpenRouterCacheControl` skipping auto-routed Anthropic models. The auto-router prepends `~` to routed model ids (e.g. `~anthropic/claude-haiku-latest`), but the cache-injection guard checked `model.startsWith('anthropic/')` against the raw string and silently skipped injection for every auto-routed call. Strip the leading `~` before matching.
