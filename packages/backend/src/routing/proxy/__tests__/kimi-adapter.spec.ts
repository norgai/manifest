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
      // The exact payload observed in production: defect #3 on the function name
      // for both calls, defect #1 on the first call (missing argument-begin),
      // and defect #2 on the second call (truncated mid-args, no end).
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
      expect(result.calls[1].truncated).toBe(true);
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
  });

  describe('createKimiStreamTransformer (chunk straddling)', () => {
    it('does not emit a partial delimiter when a chunk ends mid-token', () => {
      const transform = createKimiStreamTransformer('moonshotai/kimi-k2.5', KNOWN_TOOLS);

      // Chunk 1: text + first half of <|tool_call_begin|>
      const chunk1 = `data: ${JSON.stringify({
        choices: [
          { index: 0, delta: { content: 'Looking now. <|tool_call_be' }, finish_reason: null },
        ],
      })}\n\n`;
      const out1 = transform(chunk1);
      // Should emit "Looking now. " but hold "<|tool_call_be" in buffer.
      expect(out1).toContain('Looking now. ');
      expect(out1).not.toContain('<|tool_call_be');

      // Chunk 2: rest of begin marker + full envelope.
      const chunk2 = `data: ${JSON.stringify({
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
      })}\n\n`;
      const out2 = transform(chunk2);
      expect(out2).toContain('"tool_calls"');
      expect(out2).toContain('content-craft.scan_tree_broken_links');
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
      const chunk = `data: ${JSON.stringify({
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
      })}\n\n`;
      const out = transformKimiStreamChunk(chunk, 'moonshotai/kimi-k2.5');
      expect(out).toContain('"content":"OK"');
      expect(out).toContain('content-craft.targeted_cleanup');
    });
  });
});
