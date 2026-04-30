import {
  parseKimiToolCallEnvelope,
  reconstructFunctionName,
  fromKimiResponse,
  createKimiStreamTransformer,
  transformKimiStreamChunk,
  extractKnownToolNames,
} from '../kimi-adapter';

const KNOWN_TOOLS = [
  'content-craft.get_slug_master_entries',
  'content-craft.scan_tree_broken_links',
  'content-craft.targeted_cleanup',
];

describe('Kimi Adapter', () => {
  describe('parseKimiToolCallEnvelope', () => {
    it('is a no-op idempotent passthrough on plain text', () => {
      const text = 'Hello, this is a plain assistant response with no tool calls.';
      const result = parseKimiToolCallEnvelope(text, KNOWN_TOOLS);
      expect(result.calls).toEqual([]);
      expect(result.cleanText).toBe(text);
    });

    it('parses a well-formed envelope with both delimiters', () => {
      const text =
        'Sure, let me check.<|tool_call_begin|>functions.content-craft.get_slug_master_entries:1<|tool_call_argument_begin|>{"parent_full_slug":"foo/bar"}<|tool_call_end|>';
      const result = parseKimiToolCallEnvelope(text, KNOWN_TOOLS);

      expect(result.cleanText).toBe('Sure, let me check.');
      expect(result.calls).toHaveLength(1);
      expect(result.calls[0]).toEqual({
        id: '1',
        name: 'content-craft.get_slug_master_entries',
        arguments: { parent_full_slug: 'foo/bar' },
        truncated: false,
      });
    });

    it('handles defect #1: missing argument-begin separator (greedy brace-find)', () => {
      const text =
        '<|tool_call_begin|>functions.content-craft.scan_tree_broken_links:2{"dry_run":true}<|tool_call_end|>';
      const result = parseKimiToolCallEnvelope(text, KNOWN_TOOLS);

      expect(result.calls).toHaveLength(1);
      expect(result.calls[0]).toEqual({
        id: '2',
        name: 'content-craft.scan_tree_broken_links',
        arguments: { dry_run: true },
        truncated: false,
      });
      expect(result.cleanText).toBe('');
    });

    it('handles defect #2: truncation (no <|tool_call_end|>)', () => {
      const text =
        '<|tool_call_begin|>functions.content-craft.scan_tree_broken_links:4<|tool_call_argument_begin|>';
      const result = parseKimiToolCallEnvelope(text, KNOWN_TOOLS);

      expect(result.calls).toHaveLength(1);
      expect(result.calls[0].name).toBe('content-craft.scan_tree_broken_links');
      expect(result.calls[0].truncated).toBe(true);
      expect(result.calls[0].arguments).toEqual({});
    });

    it('handles defect #3: mangled function name (suffix-match against knownTools)', () => {
      const text =
        '<|tool_call_begin|>functionscontentcraftgetslugmasterentries3{"parent_full_slug":"cleaning-maintenance/bbq-outdoor-cleaning"}<|tool_call_end|>';
      const result = parseKimiToolCallEnvelope(text, KNOWN_TOOLS);

      expect(result.calls).toHaveLength(1);
      expect(result.calls[0]).toEqual({
        id: '3',
        name: 'content-craft.get_slug_master_entries',
        arguments: { parent_full_slug: 'cleaning-maintenance/bbq-outdoor-cleaning' },
        truncated: false,
      });
    });

    it('extracts both calls from the exact selleys failure payload', () => {
      // The exact payload observed in production: defect #3 on the function
      // name for both calls, defect #1 on both (missing argument-begin
      // separator, name+args glued). The second call ends with a section
      // terminator (`<|tool_call_argument_begin|>`) instead of an END
      // terminator — Chutes' way of closing a tool-call section. Both
      // calls have complete name+args so neither is truncated.
      const text =
        '<|tool_call_begin|>functionscontentcraftgetslugmasterentries3{"parent_full_slug": "cleaning-maintenance/bbq-outdoor-cleaning"}<|tool_call_end|>' +
        '<|tool_call_begin|>functionscontentcraftscanTreeBrokenLinks4{"dry_run": true}<|tool_call_argument_begin|>';
      const result = parseKimiToolCallEnvelope(text, KNOWN_TOOLS);

      expect(result.calls).toHaveLength(2);
      expect(result.calls[0].name).toBe('content-craft.get_slug_master_entries');
      expect(result.calls[0].arguments).toEqual({
        parent_full_slug: 'cleaning-maintenance/bbq-outdoor-cleaning',
      });
      expect(result.calls[0].truncated).toBe(false);

      expect(result.calls[1].name).toBe('content-craft.scan_tree_broken_links');
      expect(result.calls[1].arguments).toEqual({ dry_run: true });
      expect(result.calls[1].truncated).toBe(false);
    });

    it('handles the Chutes/OpenRouter sentinel-then-bare-calls shape', () => {
      // OpenRouter via Chutes emits a leading <|begin|><|end|> sentinel
      // followed by bare `name{args}<|end|>` tool-call segments and a
      // final <|argument_begin|> section terminator. The sentinel pair
      // produces no call; each bare segment yields one extracted call.
      const text =
        "I'll investigate.<|tool_call_begin|><|tool_call_end|>" +
        'functions.content-craft.get_slug_master_entries:0{"parent_full_slug":"cleaning-maintenance"}<|tool_call_end|>' +
        'functions.content-craft.get_slug_master_entries:1{"parent_full_slug":"cleaning-maintenance/bbq-outdoor-cleaning"}<|tool_call_argument_begin|>';
      const result = parseKimiToolCallEnvelope(text, KNOWN_TOOLS);

      expect(result.cleanText).toBe("I'll investigate.");
      expect(result.calls).toHaveLength(2);
      expect(result.calls[0].name).toBe('content-craft.get_slug_master_entries');
      expect(result.calls[0].arguments).toEqual({
        parent_full_slug: 'cleaning-maintenance',
      });
      expect(result.calls[0].truncated).toBe(false);
      expect(result.calls[1].name).toBe('content-craft.get_slug_master_entries');
      expect(result.calls[1].arguments).toEqual({
        parent_full_slug: 'cleaning-maintenance/bbq-outdoor-cleaning',
      });
      expect(result.calls[1].truncated).toBe(false);
    });
  });

  describe('reconstructFunctionName', () => {
    it('returns null when no known tool matches', () => {
      expect(reconstructFunctionName('functionsfoobarbaz', KNOWN_TOOLS)).toBeNull();
    });

    it('strips the functions prefix and matches by alphanumeric suffix', () => {
      const result = reconstructFunctionName('functionscontentcrafttargetedcleanup7', KNOWN_TOOLS);
      expect(result).toEqual({
        namespace: 'content-craft',
        name: 'targeted_cleanup',
        callId: '7',
      });
    });

    it('handles canonical form (functions.namespace.name:id)', () => {
      const result = reconstructFunctionName(
        'functions.content-craft.scan_tree_broken_links:42',
        KNOWN_TOOLS,
      );
      expect(result).toEqual({
        namespace: 'content-craft',
        name: 'scan_tree_broken_links',
        callId: '42',
      });
    });
  });

  describe('fromKimiResponse', () => {
    it('passes through unchanged when no envelopes present', () => {
      const resp = {
        choices: [
          { index: 0, message: { role: 'assistant', content: 'Hello' }, finish_reason: 'stop' },
        ],
      };
      const result = fromKimiResponse(resp, 'moonshotai/kimi-k2.5', KNOWN_TOOLS);
      expect(result).toBe(resp);
    });

    it('extracts envelopes into tool_calls and cleans content', () => {
      const resp = {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content:
                'Investigating now.<|tool_call_begin|>functions.content-craft.get_slug_master_entries:1<|tool_call_argument_begin|>{"parent_full_slug":"x"}<|tool_call_end|>',
            },
            finish_reason: 'stop',
          },
        ],
      };
      const result = fromKimiResponse(resp, 'moonshotai/kimi-k2.5', KNOWN_TOOLS);
      const choice = (result.choices as Array<Record<string, unknown>>)[0];
      const message = choice.message as Record<string, unknown>;
      expect(message.content).toBe('Investigating now.');
      expect(message.tool_calls).toEqual([
        {
          id: '1',
          type: 'function',
          function: {
            name: 'content-craft.get_slug_master_entries',
            arguments: '{"parent_full_slug":"x"}',
          },
        },
      ]);
      expect(choice.finish_reason).toBe('tool_calls');
    });

    it('sets content to null when only tool calls remain after cleaning', () => {
      const resp = {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content:
                '<|tool_call_begin|>functions.content-craft.scan_tree_broken_links:1<|tool_call_argument_begin|>{"dry_run":true}<|tool_call_end|>',
            },
            finish_reason: 'stop',
          },
        ],
      };
      const result = fromKimiResponse(resp, 'moonshotai/kimi-k2.5', KNOWN_TOOLS);
      const message = (result.choices as Array<Record<string, unknown>>)[0].message as Record<
        string,
        unknown
      >;
      expect(message.content).toBeNull();
    });

    it('extracts envelopes from message.reasoning (OpenRouter/Chutes path)', () => {
      // OpenRouter via Chutes emits Kimi's native output through `reasoning`
      // rather than `content`. Production response shape from selleys VM.
      const resp = {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: '',
              reasoning:
                'I\'d be happy to help.<|tool_call_begin|>functions.content-craft.scan_tree_broken_links:1<|tool_call_argument_begin|>{"dry_run":true}<|tool_call_end|>',
            },
            finish_reason: 'stop',
          },
        ],
      };
      const result = fromKimiResponse(resp, 'moonshotai/kimi-k2.5', KNOWN_TOOLS);
      const choice = (result.choices as Array<Record<string, unknown>>)[0];
      const message = choice.message as Record<string, unknown>;
      expect(message.tool_calls).toEqual([
        {
          id: '1',
          type: 'function',
          function: {
            name: 'content-craft.scan_tree_broken_links',
            arguments: '{"dry_run":true}',
          },
        },
      ]);
      expect(message.reasoning).toBe("I'd be happy to help.");
      expect(choice.finish_reason).toBe('tool_calls');
    });
  });

  describe('createKimiStreamTransformer (chunk straddling)', () => {
    it('does not emit a partial delimiter when a chunk ends mid-token', () => {
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);

      // Chunks arrive as already-stripped JSON payloads — pipeStream removes
      // the `data: ` prefix and the trailing `\n\n` before invoking transform.
      const chunk1 = JSON.stringify({
        choices: [
          { index: 0, delta: { content: 'Looking now. <|tool_call_be' }, finish_reason: null },
        ],
      });
      const out1 = transform(chunk1);
      // Should emit "Looking now. " but hold "<|tool_call_be" in buffer.
      expect(out1).toContain('Looking now. ');
      expect(out1).not.toContain('<|tool_call_be');

      const chunk2 = JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {
              content:
                'gin|>functions.content-craft.scan_tree_broken_links:1<|tool_call_argument_begin|>{"dry_run":true}<|tool_call_end|>',
            },
            finish_reason: null,
          },
        ],
      });
      const out2 = transform(chunk2);
      expect(out2).toContain('"tool_calls"');
      expect(out2).toContain('content-craft.scan_tree_broken_links');
    });

    it('passes through plain content events with no envelopes', () => {
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);
      const chunk = JSON.stringify({
        choices: [{ index: 0, delta: { content: 'Hello!' }, finish_reason: null }],
      });
      const out = transform(chunk);
      expect(out).toContain('"content":"Hello!"');
      expect(out).toMatch(/^data: /);
      expect(out!.endsWith('\n\n')).toBe(true);
    });

    it('drops openrouter SSE keepalive comments (non-JSON data lines)', () => {
      // OpenRouter emits `data: : OPENROUTER PROCESSING\n\n` keepalive
      // markers that arrive at the transformer (after `data: ` is stripped)
      // as `: OPENROUTER PROCESSING`. Forwarding them downstream causes the
      // openclaw runtime SSE parser to throw on JSON.parse — drop instead.
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);
      expect(transform(': OPENROUTER PROCESSING')).toBeNull();
      expect(transform(': keepalive')).toBeNull();
      expect(transform('not json at all')).toBeNull();
    });

    it('passes through role-only and finish_reason events unchanged', () => {
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);
      const role = JSON.stringify({
        choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      });
      const stop = JSON.stringify({
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      });
      expect(transform(role)).toContain('"role":"assistant"');
      expect(transform(stop)).toContain('"finish_reason":"stop"');
    });

    it('flushes buffered envelope when finish_reason arrives', () => {
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);
      // Send a complete envelope as one content delta (envelope ends, no
      // trailing text yet — buffer holds nothing, all calls emit immediately).
      const c1 = JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {
              content:
                '<|tool_call_begin|>functions.content-craft.targeted_cleanup:5<|tool_call_argument_begin|>{"slugs":["a"]}<|tool_call_end|>',
            },
            finish_reason: null,
          },
        ],
      });
      const out1 = transform(c1);
      expect(out1).toContain('"tool_calls"');

      const c2 = JSON.stringify({
        choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
      });
      const out2 = transform(c2);
      expect(out2).toContain('"finish_reason":"tool_calls"');
    });

    it('extracts envelopes from delta.reasoning (Chutes streaming path)', () => {
      // Chutes provider streams Kimi's output token-by-token through
      // `delta.reasoning` rather than `delta.content`. The adapter must
      // buffer reasoning text the same way as content text.
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);
      const tokens = [
        'OK ',
        '<|tool_call_begin|>',
        'functions.content-craft.scan_tree_broken_links:7',
        '<|tool_call_argument_begin|>',
        '{"dry_run":true}',
        '<|tool_call_end|>',
      ];
      let combined = '';
      for (const tok of tokens) {
        const chunk = JSON.stringify({
          choices: [{ index: 0, delta: { reasoning: tok }, finish_reason: null }],
        });
        const out = transform(chunk);
        if (out) combined += out;
      }
      expect(combined).toContain('"tool_calls"');
      expect(combined).toContain('content-craft.scan_tree_broken_links');
      expect(combined).toContain('"reasoning":"OK "');
    });

    it('flushes both content and reasoning buffers on finish_reason', () => {
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);
      // Reasoning channel carries a complete envelope, content stays empty.
      const c1 = JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {
              reasoning:
                '<|tool_call_begin|>functions.content-craft.targeted_cleanup:9<|tool_call_argument_begin|>{"slugs":["x"]}<|tool_call_end|>',
            },
            finish_reason: null,
          },
        ],
      });
      const out1 = transform(c1);
      expect(out1).toContain('"tool_calls"');

      const c2 = JSON.stringify({
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      });
      const out2 = transform(c2);
      expect(out2).toContain('"finish_reason":"stop"');
    });
  });

  describe('extractKnownToolNames', () => {
    it('pulls function names from request body tools array', () => {
      const body = {
        tools: [
          { type: 'function', function: { name: 'content-craft.scan_tree_broken_links' } },
          { type: 'function', function: { name: 'content-craft.get_slug_master_entries' } },
        ],
      };
      expect(extractKnownToolNames(body)).toEqual([
        'content-craft.scan_tree_broken_links',
        'content-craft.get_slug_master_entries',
      ]);
    });

    it('returns empty array when body has no tools', () => {
      expect(extractKnownToolNames(undefined)).toEqual([]);
      expect(extractKnownToolNames({})).toEqual([]);
      expect(extractKnownToolNames({ tools: [] })).toEqual([]);
    });
  });

  describe('transformKimiStreamChunk (stateless convenience)', () => {
    it('handles a single complete envelope chunk', () => {
      const chunk = JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {
              content:
                'OK<|tool_call_begin|>functions.content-craft.targeted_cleanup:9<|tool_call_argument_begin|>{"slugs":["x"]}<|tool_call_end|>',
            },
            finish_reason: null,
          },
        ],
      });
      const out = transformKimiStreamChunk(chunk, 'moonshotai/kimi-k2.5');
      expect(out).toContain('"content":"OK"');
      expect(out).toContain('content-craft.targeted_cleanup');
    });
  });
});
