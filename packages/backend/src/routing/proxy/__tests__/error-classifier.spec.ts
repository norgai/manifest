import { classifyUpstreamError } from '../error-classifier';

describe('classifyUpstreamError', () => {
  describe('context_overflow detection', () => {
    it('detects the OpenRouter-wrapped Anthropic overflow we observed in prod', () => {
      const body = JSON.stringify({
        error: {
          message: 'Provider returned error',
          code: 400,
          metadata: {
            raw: '{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 207001 tokens > 200000 maximum"},"request_id":"req_011CaZAuUy91ZwkkJWzi2nak"}',
            provider_name: 'Anthropic',
          },
        },
      });
      expect(classifyUpstreamError(400, body)).toBe('context_overflow');
    });

    it('detects the Anthropic-direct overflow shape', () => {
      const body = JSON.stringify({
        type: 'error',
        error: {
          type: 'invalid_request_error',
          message: 'prompt is too long: 250000 tokens > 200000 maximum',
        },
      });
      expect(classifyUpstreamError(400, body)).toBe('context_overflow');
    });

    it('detects the OpenAI context_length_exceeded error code', () => {
      const body = JSON.stringify({
        error: {
          code: 'context_length_exceeded',
          message:
            "This model's maximum context length is 128000 tokens, however you requested 150000.",
        },
      });
      expect(classifyUpstreamError(400, body)).toBe('context_overflow');
    });

    it('detects "maximum context length" phrase variants', () => {
      const body = '{"error":"This model\'s maximum context length is 200000 tokens"}';
      expect(classifyUpstreamError(400, body)).toBe('context_overflow');
    });

    it('detects "input is too long" (Cohere-style)', () => {
      const body = '{"message":"input is too long for this model"}';
      expect(classifyUpstreamError(400, body)).toBe('context_overflow');
    });

    it('matches case-insensitively', () => {
      const body = 'PROMPT IS TOO LONG: foo';
      expect(classifyUpstreamError(400, body)).toBe('context_overflow');
    });
  });

  describe('not-overflow cases', () => {
    it('returns "other" for plain 400 with no overflow signal', () => {
      expect(classifyUpstreamError(400, '{"error":"invalid model"}')).toBe('other');
    });

    it('returns "other" for 401 unauthorized', () => {
      expect(classifyUpstreamError(401, '{"error":"invalid api key"}')).toBe('other');
    });

    it('returns "other" for 429 rate limit', () => {
      expect(classifyUpstreamError(429, '{"error":"rate limit exceeded"}')).toBe('other');
    });

    it('returns "other" for 5xx server errors even with overflow text', () => {
      // 5xx is a server problem, not a client-side overflow we can fix by escalating
      expect(classifyUpstreamError(503, 'prompt is too long')).toBe('other');
    });

    it('returns "other" for 2xx success even with overflow text in body', () => {
      expect(classifyUpstreamError(200, 'prompt is too long')).toBe('other');
    });

    it('returns "other" for empty body', () => {
      expect(classifyUpstreamError(400, '')).toBe('other');
    });
  });
});
