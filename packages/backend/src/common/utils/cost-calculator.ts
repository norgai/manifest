import { PricingEntry } from '../../model-prices/model-pricing-cache.service';

export interface CostInput {
  inputTokens: number;
  outputTokens: number;
  /**
   * Tokens served from a cached prefix. Billed at `cache_read_price_per_token`
   * when available; otherwise fall back to the full input rate. Subset of `inputTokens`.
   */
  cacheReadTokens?: number;
  /**
   * Tokens written to a fresh cache prefix. Billed at `cache_creation_price_per_token`
   * when available; otherwise fall back to the full input rate. Subset of `inputTokens`.
   */
  cacheCreationTokens?: number;
  model: string | null | undefined;
  pricing: PricingEntry | undefined;
  /**
   * When true, cost is always 0 (subscription-based usage).
   * Callers determine this from authType or subscription-provider sets.
   */
  isSubscription?: boolean;
}

/**
 * Computes the USD cost for a set of tokens given a pricing entry.
 *
 * Anthropic prompt caching: `inputTokens` is the total prompt token count
 * (including any portion served from cache). `cacheReadTokens` and
 * `cacheCreationTokens` are subsets of that total — the remainder is billed
 * at the full input rate. When the pricing entry lacks cache rates, cached
 * tokens fall back to the full input rate (preserving prior behaviour).
 *
 * Returns:
 * - `0` when the usage is subscription-based
 * - `null` when the model is unknown, tokens are zero, or pricing is unavailable
 * - the computed cost otherwise
 */
export function computeTokenCost(input: CostInput): number | null {
  if (!input.model) return null;
  if (input.isSubscription) return 0;
  if (input.inputTokens === 0 && input.outputTokens === 0) return null;

  const pricing = input.pricing;
  if (!pricing || pricing.input_price_per_token == null || pricing.output_price_per_token == null) {
    return null;
  }

  const inputRate = Number(pricing.input_price_per_token);
  const outputRate = Number(pricing.output_price_per_token);
  const cacheReadRate =
    pricing.cache_read_price_per_token != null
      ? Number(pricing.cache_read_price_per_token)
      : inputRate;
  const cacheCreationRate =
    pricing.cache_creation_price_per_token != null
      ? Number(pricing.cache_creation_price_per_token)
      : inputRate;

  const cacheRead = Math.max(0, input.cacheReadTokens ?? 0);
  const cacheCreation = Math.max(0, input.cacheCreationTokens ?? 0);
  const fullRateInput = Math.max(0, input.inputTokens - cacheRead - cacheCreation);

  const cost =
    fullRateInput * inputRate +
    cacheRead * cacheReadRate +
    cacheCreation * cacheCreationRate +
    input.outputTokens * outputRate;

  return cost < 0 ? null : cost;
}
