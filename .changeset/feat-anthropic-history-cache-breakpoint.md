---
manifest: patch
---

Add a third `cache_control` breakpoint on the last conversation message in `toAnthropicRequest`. Anthropic supports up to 4 breakpoints; manifest already used 2 (system + tools). The new history breakpoint lets the rolling prefix (system + tools + prior turns) be cached on subsequent calls, dramatically reducing input tokens billed at full rate for chatty multi-turn agents. Behaviour-preserving when `injectCacheControl: false`.
