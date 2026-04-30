---
manifest: patch
---

Extract cache_read / cache_creation tokens from OpenRouter (and other OpenAI-compat) responses. The proxy was reading `usage.cache_read_tokens` directly, which is the manifest-internal field name set by the Anthropic adapter — but never present on raw OpenAI/OpenRouter responses, where cached input tokens live at `usage.prompt_tokens_details.cached_tokens`. New `pickCacheTokens` helper reads all known shapes (manifest-internal, OpenAI standard, Anthropic-native) so cache stats now record correctly for OpenRouter routing in addition to direct Anthropic.
