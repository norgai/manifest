---
manifest: minor
---

Apply cache_read / cache_creation pricing in cost calculation. The cost calculator was billing all input tokens at the full input rate, ignoring Anthropic prompt caching discounts (cache_read at ~10% of input, cache_creation at ~125%). Result: dashboard cost was systematically over-reported when caching was active — ~3× higher than what OpenRouter / Anthropic actually billed. This fix:

- Extends `PricingEntry` with `cache_read_price_per_token` and `cache_creation_price_per_token`
- Reads `pricing.input_cache_read` and `pricing.input_cache_write` from OpenRouter's `/api/v1/models` response
- Propagates models.dev's `cacheReadPricePerToken` / `cacheWritePricePerToken` (already extracted but never plumbed through)
- Extends `computeTokenCost` with `cacheReadTokens` and `cacheCreationTokens` inputs that apply the discounted rates, falling back to the full input rate when cache rates are absent (preserves prior behaviour)
- Updates both `proxy-message-recorder` call sites to pass cache token counts from the recorded usage

Cosmetic / wallet-impact note: real billing was always correct at the upstream provider — this fix corrects manifest's reported cost so it matches the actual OpenRouter / direct-Anthropic invoice.
