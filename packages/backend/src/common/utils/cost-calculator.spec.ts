import { computeTokenCost } from './cost-calculator';
import { PricingEntry } from '../../model-prices/model-pricing-cache.service';

describe('computeTokenCost', () => {
  const pricing: PricingEntry = {
    model_name: 'gpt-4o',
    provider: 'OpenAI',
    input_price_per_token: 0.0000025,
    output_price_per_token: 0.00001,
    display_name: 'GPT-4o',
  };

  it('returns null when model is null', () => {
    expect(
      computeTokenCost({ inputTokens: 100, outputTokens: 50, model: null, pricing }),
    ).toBeNull();
  });

  it('returns null when model is undefined', () => {
    expect(
      computeTokenCost({ inputTokens: 100, outputTokens: 50, model: undefined, pricing }),
    ).toBeNull();
  });

  it('returns null when both token counts are zero', () => {
    expect(
      computeTokenCost({ inputTokens: 0, outputTokens: 0, model: 'gpt-4o', pricing }),
    ).toBeNull();
  });

  it('returns 0 when isSubscription is true', () => {
    expect(
      computeTokenCost({
        inputTokens: 1000,
        outputTokens: 500,
        model: 'gpt-4o',
        pricing,
        isSubscription: true,
      }),
    ).toBe(0);
  });

  it('returns null when pricing is undefined', () => {
    expect(
      computeTokenCost({
        inputTokens: 100,
        outputTokens: 50,
        model: 'unknown-model',
        pricing: undefined,
      }),
    ).toBeNull();
  });

  it('returns null when input_price_per_token is null', () => {
    const partial: PricingEntry = { ...pricing, input_price_per_token: null };
    expect(
      computeTokenCost({ inputTokens: 100, outputTokens: 50, model: 'gpt-4o', pricing: partial }),
    ).toBeNull();
  });

  it('returns null when output_price_per_token is null', () => {
    const partial: PricingEntry = { ...pricing, output_price_per_token: null };
    expect(
      computeTokenCost({ inputTokens: 100, outputTokens: 50, model: 'gpt-4o', pricing: partial }),
    ).toBeNull();
  });

  it('computes cost correctly with both token types', () => {
    const result = computeTokenCost({
      inputTokens: 1000,
      outputTokens: 500,
      model: 'gpt-4o',
      pricing,
    });
    // 1000 * 0.0000025 + 500 * 0.00001 = 0.0025 + 0.005 = 0.0075
    expect(result).toBeCloseTo(0.0075, 10);
  });

  it('computes cost with only input tokens', () => {
    const result = computeTokenCost({
      inputTokens: 2000,
      outputTokens: 0,
      model: 'gpt-4o',
      pricing,
    });
    // inputTokens > 0 && outputTokens === 0 → not both zero, so compute
    // 2000 * 0.0000025 + 0 * 0.00001 = 0.005
    expect(result).toBeCloseTo(0.005, 10);
  });

  it('computes cost with only output tokens', () => {
    const result = computeTokenCost({
      inputTokens: 0,
      outputTokens: 300,
      model: 'gpt-4o',
      pricing,
    });
    // 0 * 0.0000025 + 300 * 0.00001 = 0.003
    expect(result).toBeCloseTo(0.003, 10);
  });

  it('handles string-typed price fields via Number() coercion', () => {
    const stringPricing: PricingEntry = {
      model_name: 'test-model',
      provider: 'Test',
      input_price_per_token: '0.000005' as unknown as number,
      output_price_per_token: '0.00002' as unknown as number,
      display_name: null,
    };
    const result = computeTokenCost({
      inputTokens: 100,
      outputTokens: 200,
      model: 'test-model',
      pricing: stringPricing,
    });
    // 100 * 0.000005 + 200 * 0.00002 = 0.0005 + 0.004 = 0.0045
    expect(result).toBeCloseTo(0.0045, 10);
  });

  it('returns 0 for subscription even when pricing has null prices', () => {
    const noPricing: PricingEntry = {
      ...pricing,
      input_price_per_token: null,
      output_price_per_token: null,
    };
    expect(
      computeTokenCost({
        inputTokens: 500,
        outputTokens: 200,
        model: 'gpt-4o',
        pricing: noPricing,
        isSubscription: true,
      }),
    ).toBe(0);
  });

  it('returns null when computed cost would be negative (both prices negative)', () => {
    const negativePricing: PricingEntry = {
      model_name: 'bad-model',
      provider: 'Test',
      input_price_per_token: -1,
      output_price_per_token: -1,
      display_name: null,
    };
    expect(
      computeTokenCost({
        inputTokens: 100,
        outputTokens: 50,
        model: 'bad-model',
        pricing: negativePricing,
      }),
    ).toBeNull();
  });

  it('returns null when net cost is negative from mixed positive/negative prices', () => {
    const mixedPricing: PricingEntry = {
      model_name: 'mixed-model',
      provider: 'Test',
      input_price_per_token: -0.01,
      output_price_per_token: 0.000001,
      display_name: null,
    };
    // -0.01 * 1000 + 0.000001 * 10 = -10 + 0.00001 = -9.99999 < 0
    expect(
      computeTokenCost({
        inputTokens: 1000,
        outputTokens: 10,
        model: 'mixed-model',
        pricing: mixedPricing,
      }),
    ).toBeNull();
  });

  it('returns the computed value when cost is exactly zero (free model)', () => {
    const freePricing: PricingEntry = {
      model_name: 'free-model',
      provider: 'Free',
      input_price_per_token: 0,
      output_price_per_token: 0,
      display_name: null,
    };
    // 0 * 500 + 0 * 200 = 0, which is >= 0 so it should return 0
    expect(
      computeTokenCost({
        inputTokens: 500,
        outputTokens: 200,
        model: 'free-model',
        pricing: freePricing,
      }),
    ).toBe(0);
  });

  it('returns null for empty-string model (falsy)', () => {
    expect(
      computeTokenCost({
        inputTokens: 100,
        outputTokens: 50,
        model: '' as unknown as string,
        pricing,
      }),
    ).toBeNull();
  });

  it('returns 0 for subscription even when tokens are zero', () => {
    expect(
      computeTokenCost({
        inputTokens: 0,
        outputTokens: 0,
        model: 'gpt-4o',
        pricing: undefined,
        isSubscription: true,
      }),
    ).toBe(0);
  });

  it('subscription check takes priority over negative pricing guard', () => {
    const negativePricing: PricingEntry = {
      model_name: 'bad-model',
      provider: 'Test',
      input_price_per_token: -1,
      output_price_per_token: -1,
      display_name: null,
    };
    expect(
      computeTokenCost({
        inputTokens: 100,
        outputTokens: 50,
        model: 'bad-model',
        pricing: negativePricing,
        isSubscription: true,
      }),
    ).toBe(0);
  });

  describe('cache token discounts', () => {
    const haikuPricing: PricingEntry = {
      model_name: 'anthropic/claude-haiku-4.5',
      provider: 'OpenRouter',
      input_price_per_token: 0.000001, // $1 / M
      output_price_per_token: 0.000005, // $5 / M
      cache_read_price_per_token: 0.0000001, // $0.10 / M (90% discount)
      cache_creation_price_per_token: 0.00000125, // $1.25 / M (25% premium)
      display_name: 'Claude Haiku 4.5',
    };

    it('discounts cache_read tokens at the cache_read rate', () => {
      // 100k input total, 80k served from cache, 20k full-rate, 1k output
      // = 20000 * $1/M + 80000 * $0.10/M + 1000 * $5/M
      // = $0.020 + $0.008 + $0.005 = $0.033
      const cost = computeTokenCost({
        inputTokens: 100_000,
        outputTokens: 1_000,
        cacheReadTokens: 80_000,
        cacheCreationTokens: 0,
        model: 'anthropic/claude-haiku-4.5',
        pricing: haikuPricing,
      });
      expect(cost).toBeCloseTo(0.033, 6);
    });

    it('charges cache_creation tokens at the cache_creation rate', () => {
      // 100k input total, 90k freshly cached, 10k full-rate, 1k output
      // = 10000 * $1/M + 90000 * $1.25/M + 1000 * $5/M
      // = $0.010 + $0.1125 + $0.005 = $0.1275
      const cost = computeTokenCost({
        inputTokens: 100_000,
        outputTokens: 1_000,
        cacheReadTokens: 0,
        cacheCreationTokens: 90_000,
        model: 'anthropic/claude-haiku-4.5',
        pricing: haikuPricing,
      });
      expect(cost).toBeCloseTo(0.1275, 6);
    });

    it('reproduces the OpenRouter haiku-4.5 ~85%-cached scenario from prod', () => {
      // 11:51 AM real call: 116,071 in, 424 out, OpenRouter charged $0.0267
      // Estimating cache_read at ~98k (matches OpenRouter's bill within rounding)
      const cost = computeTokenCost({
        inputTokens: 116_071,
        outputTokens: 424,
        cacheReadTokens: 98_000,
        cacheCreationTokens: 0,
        model: 'anthropic/claude-haiku-4.5',
        pricing: haikuPricing,
      });
      // (116071 - 98000) * 1e-6 + 98000 * 1e-7 + 424 * 5e-6 = 0.018071 + 0.0098 + 0.00212
      expect(cost).toBeCloseTo(0.029991, 5);
    });

    it('falls back to full input rate when cache rates are absent', () => {
      // Without cache_read_price_per_token, cached tokens are billed at full rate
      // — preserves the pre-fix behaviour, just doesn't apply the discount
      const noCacheRates: PricingEntry = { ...haikuPricing };
      delete noCacheRates.cache_read_price_per_token;
      delete noCacheRates.cache_creation_price_per_token;
      const cost = computeTokenCost({
        inputTokens: 100_000,
        outputTokens: 0,
        cacheReadTokens: 80_000,
        model: 'anthropic/claude-haiku-4.5',
        pricing: noCacheRates,
      });
      // Should bill full rate on all 100k input (cached portion not discounted)
      expect(cost).toBeCloseTo(0.1, 6);
    });

    it('omitted cache token counts behave as zero (no discount)', () => {
      // Existing callers that don't pass cache fields get the same result as before
      const cost = computeTokenCost({
        inputTokens: 100_000,
        outputTokens: 0,
        model: 'anthropic/claude-haiku-4.5',
        pricing: haikuPricing,
      });
      expect(cost).toBeCloseTo(0.1, 6);
    });

    it('clamps full-rate input to zero when cache totals exceed inputTokens', () => {
      // Defensive: if usage reports cache_read > input (shouldn't happen, but...)
      // we shouldn't generate negative full-rate billing
      const cost = computeTokenCost({
        inputTokens: 50_000,
        outputTokens: 0,
        cacheReadTokens: 100_000, // larger than inputTokens
        model: 'anthropic/claude-haiku-4.5',
        pricing: haikuPricing,
      });
      // 0 full-rate + 100000 * $0.10/M = $0.010
      expect(cost).toBeCloseTo(0.01, 6);
    });
  });
});
