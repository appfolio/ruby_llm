# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::Converse::Streaming do
  let(:streaming) do
    Object.new.tap do |object|
      object.extend(described_class)
      object.instance_variable_set(:@model, instance_double(RubyLLM::Model::Info, id: 'bedrock-test-model'))
    end
  end

  it 'extracts thinking text from Bedrock Converse Stream reasoningContent deltas' do
    event = {
      'contentBlockDelta' => {
        'delta' => {
          'reasoningContent' => {
            'text' => 'thinking text'
          }
        }
      }
    }

    chunk = streaming.send(:build_chunk, event)

    expect(chunk.thinking.text).to eq('thinking text')
  end

  it 'extracts thinking signatures from Bedrock Converse Stream reasoningContent deltas' do
    event = {
      'contentBlockDelta' => {
        'delta' => {
          'reasoningContent' => {
            'signature' => 'thinking-signature'
          }
        }
      }
    }

    chunk = streaming.send(:build_chunk, event)

    expect(chunk.thinking.signature).to eq('thinking-signature')
  end

  it 'preserves raw stopReason from messageStop events' do
    event = {
      'messageStop' => {
        'stopReason' => 'max_tokens'
      }
    }

    chunk = streaming.send(:build_chunk, event)

    expect(chunk.finish_reason).to eq('max_tokens')
  end

  it 'extracts thinking tokens from nested usage output token details' do
    event = {
      'metadata' => {
        'usage' => {
          'inputTokens' => 10,
          'outputTokens' => 5,
          'outputTokensDetails' => { 'reasoningTokens' => 7 }
        }
      }
    }

    chunk = streaming.send(:build_chunk, event)

    expect(chunk.thinking_tokens).to eq(7)
  end

  it 'accumulates Bedrock Converse Stream thinking deltas into the final message' do
    accumulator = RubyLLM::StreamAccumulator.new
    text_event = {
      'contentBlockDelta' => {
        'delta' => {
          'reasoningContent' => {
            'text' => 'thinking text'
          }
        }
      }
    }
    signature_event = {
      'contentBlockDelta' => {
        'delta' => {
          'reasoningContent' => {
            'signature' => 'thinking-signature'
          }
        }
      }
    }

    accumulator.add(streaming.send(:build_chunk, text_event))
    accumulator.add(streaming.send(:build_chunk, signature_event))
    message = accumulator.to_message(nil)

    expect(message.thinking.text).to eq('thinking text')
    expect(message.thinking.signature).to eq('thinking-signature')
  end

  describe 'multi-block thinking' do
    # Feeds events through build_chunk with a single shared thinking_state hash,
    # mirroring how stream_response threads it across the whole event stream.
    def accumulate(events)
      accumulator = RubyLLM::StreamAccumulator.new
      thinking_state = {}
      events.each { |e| accumulator.add(streaming.send(:build_chunk, e, thinking_state)) }
      accumulator.to_message(nil)
    end

    it 'captures a single normal thinking block, matching today\'s fallback text/signature output' do
      events = [
        { 'contentBlockStart' => { 'contentBlockIndex' => 0,
                                   'start' => { 'reasoningContent' => {} } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 0,
                                   'delta' => { 'reasoningContent' => { 'text' => 'thinking text' } } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 0,
                                   'delta' => { 'reasoningContent' => { 'signature' => 'thinking-signature' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 0 } }
      ]

      message = accumulate(events)

      expect(message.thinking.text).to eq('thinking text')
      expect(message.thinking.signature).to eq('thinking-signature')
      expect(message.thinking.blocks).to eq(
        [
          { 'reasoningContent' => { 'reasoningText' => { 'text' => 'thinking text',
                                                         'signature' => 'thinking-signature' } } }
        ]
      )
    end

    it 'preserves a redacted thinking block followed by a normal thinking block, in order' do
      events = [
        { 'contentBlockStart' => { 'contentBlockIndex' => 0,
                                   'start' => { 'reasoningContent' => { 'redactedContent' => 'opaque-blob-1' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 0 } },
        { 'contentBlockStart' => { 'contentBlockIndex' => 1,
                                   'start' => { 'reasoningContent' => {} } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 1,
                                   'delta' => { 'reasoningContent' => { 'text' => 'step two' } } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 1,
                                   'delta' => { 'reasoningContent' => { 'signature' => 'sig-2' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 1 } },
        { 'contentBlockStart' => { 'contentBlockIndex' => 2, 'start' => { 'text' => {} } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 2, 'delta' => { 'text' => 'Done' } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 2 } }
      ]

      message = accumulate(events)

      expect(message.thinking.blocks).to eq(
        [
          { 'reasoningContent' => { 'redactedContent' => 'opaque-blob-1' } },
          { 'reasoningContent' => { 'reasoningText' => { 'text' => 'step two', 'signature' => 'sig-2' } } }
        ]
      )
      expect(message.content).to eq('Done')
    end

    it 'preserves multiple normal thinking blocks separated by a tool_use block' do
      events = [
        { 'contentBlockStart' => { 'contentBlockIndex' => 0,
                                   'start' => { 'reasoningContent' => {} } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 0,
                                   'delta' => { 'reasoningContent' => { 'text' => 'first thought' } } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 0,
                                   'delta' => { 'reasoningContent' => { 'signature' => 'sig-1' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 0 } },
        { 'contentBlockStart' => { 'contentBlockIndex' => 1,
                                   'start' => { 'toolUse' => { 'toolUseId' => 'call_1', 'name' => 'search' } } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 1,
                                   'delta' => { 'toolUse' => { 'input' => '{}' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 1 } },
        { 'contentBlockStart' => { 'contentBlockIndex' => 2,
                                   'start' => { 'reasoningContent' => {} } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 2,
                                   'delta' => { 'reasoningContent' => { 'text' => 'second thought' } } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 2,
                                   'delta' => { 'reasoningContent' => { 'signature' => 'sig-2' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 2 } }
      ]

      message = accumulate(events)

      expect(message.thinking.blocks).to eq(
        [
          { 'reasoningContent' => { 'reasoningText' => { 'text' => 'first thought', 'signature' => 'sig-1' } } },
          { 'reasoningContent' => { 'reasoningText' => { 'text' => 'second thought', 'signature' => 'sig-2' } } }
        ]
      )
      expect(message.tool_calls['call_1'].name).to eq('search')
    end

    it 'drops a thinking block left open when the turn is truncated before contentBlockStop' do
      events = [
        { 'contentBlockStart' => { 'contentBlockIndex' => 0,
                                   'start' => { 'reasoningContent' => {} } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 0,
                                   'delta' => { 'reasoningContent' => { 'text' => 'cut off mid-thought' } } } },
        { 'messageStop' => { 'stopReason' => 'max_tokens' } }
      ]

      message = accumulate(events)

      expect(message.thinking.blocks).to be_nil
      expect(message.finish_reason).to eq('max_tokens')
    end

    it 'finalizes a redacted thinking block via messageStop when Bedrock never sends its own contentBlockStop' do
      events = [
        { 'contentBlockStart' => { 'contentBlockIndex' => 0,
                                   'start' => { 'reasoningContent' => { 'redactedContent' => 'opaque-blob-1' } } } },
        # No contentBlockStop for index 0 — the next block starts immediately, as Bedrock has
        # been observed to do for a redacted-thinking block with no visible content.
        { 'contentBlockStart' => { 'contentBlockIndex' => 1,
                                   'start' => { 'toolUse' => { 'toolUseId' => 'call_1', 'name' => 'search' } } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 1,
                                   'delta' => { 'toolUse' => { 'input' => '{}' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 1 } },
        { 'messageStop' => { 'stopReason' => 'tool_use' } }
      ]

      message = accumulate(events)

      expect(message.thinking.blocks).to eq(
        [{ 'reasoningContent' => { 'redactedContent' => 'opaque-blob-1' } }]
      )
      expect(message.tool_calls['call_1'].name).to eq('search')
    end

    it 'finalizes a signature-only thinking block as an empty reasoningText' do
      events = [
        { 'contentBlockDelta' => { 'contentBlockIndex' => 0,
                                   'delta' => { 'reasoningContent' => { 'signature' => 'sig-only' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 0 } }
      ]

      message = accumulate(events)

      expect(message.thinking.blocks).to eq(
        [{ 'reasoningContent' => { 'reasoningText' => { 'text' => '', 'signature' => 'sig-only' } } }]
      )
    end

    it 'round-trips a streamed multi-block thinking turn through format_thinking_blocks unmodified' do
      events = [
        { 'contentBlockStart' => { 'contentBlockIndex' => 0,
                                   'start' => { 'reasoningContent' => { 'redactedContent' => 'opaque-blob-1' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 0 } },
        { 'contentBlockStart' => { 'contentBlockIndex' => 1,
                                   'start' => { 'reasoningContent' => {} } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 1,
                                   'delta' => { 'reasoningContent' => { 'text' => 'step two' } } } },
        { 'contentBlockDelta' => { 'contentBlockIndex' => 1,
                                   'delta' => { 'reasoningContent' => { 'signature' => 'sig-2' } } } },
        { 'contentBlockStop' => { 'contentBlockIndex' => 1 } }
      ]

      message = accumulate(events)
      message.instance_variable_set(:@content, 'Done')
      formatted = RubyLLM::Protocols::Converse::Chat.format_messages([message])
      thinking_blocks = formatted.first[:content].select { |b| b.key?(:reasoningContent) || b.key?('reasoningContent') }

      expect(thinking_blocks).to eq(message.thinking.blocks)
    end
  end

  # ConverseStream's real wire format carries the event type in the eventstream
  # :event-type HEADER; the JSON payload is the bare member struct — e.g.
  # {"contentBlockIndex":0,"delta":{...}}, {"stopReason":"tool_use"} — never nested
  # under the event name like the hand-built hashes above. These specs frame events
  # exactly as Bedrock does (captured from a live ConverseStream response) and drive
  # them through the same decode path stream_response uses.
  describe 'real wire framing (:event-type header + flat payload)' do
    require 'aws-eventstream'

    def wire_chunk(*typed_payloads)
      encoder = Aws::EventStream::Encoder.new
      typed_payloads.map do |(type, payload)|
        message = Aws::EventStream::Message.new(
          headers: {
            ':event-type' => Aws::EventStream::HeaderValue.new(value: type, type: 'string'),
            ':content-type' => Aws::EventStream::HeaderValue.new(value: 'application/json', type: 'string'),
            ':message-type' => Aws::EventStream::HeaderValue.new(value: 'event', type: 'string')
          },
          payload: StringIO.new(JSON.generate(payload))
        )
        encoder.encode(message)
      end.join
    end

    def stream_to_message(*typed_payloads)
      accumulator = RubyLLM::StreamAccumulator.new
      decoder = Aws::EventStream::Decoder.new
      chunks = []
      streaming.send(:parse_stream_chunk, decoder, wire_chunk(*typed_payloads), accumulator, {}) do |chunk|
        chunks << chunk
      end
      accumulator.to_message(nil)
    end

    it 'captures thinking blocks from a flat-framed tool-call turn' do
      message = stream_to_message(
        ['messageStart', { 'role' => 'assistant' }],
        ['contentBlockDelta', { 'contentBlockIndex' => 0,
                                'delta' => { 'reasoningContent' => { 'text' => 'need the weather' } } }],
        ['contentBlockDelta', { 'contentBlockIndex' => 0,
                                'delta' => { 'reasoningContent' => { 'signature' => 'sig-1' } } }],
        ['contentBlockStop', { 'contentBlockIndex' => 0 }],
        ['contentBlockStart', { 'contentBlockIndex' => 1,
                                'start' => { 'toolUse' => { 'toolUseId' => 'call_1', 'name' => 'weather' } } }],
        ['contentBlockDelta', { 'contentBlockIndex' => 1, 'delta' => { 'toolUse' => { 'input' => '{}' } } }],
        ['contentBlockStop', { 'contentBlockIndex' => 1 }],
        ['messageStop', { 'stopReason' => 'tool_use' }],
        ['metadata', { 'usage' => { 'inputTokens' => 10, 'outputTokens' => 5 }, 'metrics' => {} }]
      )

      expect(message.thinking.blocks).to eq(
        [{ 'reasoningContent' => { 'reasoningText' => { 'text' => 'need the weather',
                                                        'signature' => 'sig-1' } } }]
      )
      expect(message.thinking.text).to eq('need the weather')
      expect(message.thinking.signature).to eq('sig-1')
      expect(message.tool_calls['call_1'].name).to eq('weather')
      expect(message.finish_reason).to eq('tool_use')
      expect(message.output_tokens).to eq(5)
    end

    it 'finalizes a flat-framed signature-only thinking block left open at messageStop' do
      message = stream_to_message(
        ['messageStart', { 'role' => 'assistant' }],
        ['contentBlockDelta', { 'contentBlockIndex' => 0,
                                'delta' => { 'reasoningContent' => { 'signature' => 'sig-only' } } }],
        ['messageStop', { 'stopReason' => 'tool_use' }]
      )

      expect(message.thinking.blocks).to eq(
        [{ 'reasoningContent' => { 'reasoningText' => { 'text' => '', 'signature' => 'sig-only' } } }]
      )
    end

    it 'accumulates flat-framed redacted content deltas into a redacted block' do
      message = stream_to_message(
        ['contentBlockDelta', { 'contentBlockIndex' => 0,
                                'delta' => { 'reasoningContent' => { 'redactedContent' => 'blob-part-1' } } }],
        ['contentBlockDelta', { 'contentBlockIndex' => 0,
                                'delta' => { 'reasoningContent' => { 'redactedContent' => 'blob-part-2' } } }],
        ['contentBlockStop', { 'contentBlockIndex' => 0 }],
        ['messageStop', { 'stopReason' => 'end_turn' }]
      )

      expect(message.thinking.blocks).to eq(
        [{ 'reasoningContent' => { 'redactedContent' => 'blob-part-1blob-part-2' } }]
      )
    end

    it 'nests an exception payload under its :exception-type header' do
      message = Aws::EventStream::Message.new(
        headers: {
          ':exception-type' => Aws::EventStream::HeaderValue.new(value: 'throttlingException', type: 'string'),
          ':message-type' => Aws::EventStream::HeaderValue.new(value: 'exception', type: 'string')
        },
        payload: StringIO.new(JSON.generate({ 'message' => 'Too many requests' }))
      )

      event = streaming.send(:nest_event_under_type, { 'message' => 'Too many requests' }, message)

      expect(event).to eq({ 'throttlingException' => { 'message' => 'Too many requests' } })
      expect(streaming.send(:stream_error_event?, event)).to be(true)
    end

    it 'passes an already-nested payload through unchanged' do
      message = Aws::EventStream::Message.new(
        headers: { ':event-type' => Aws::EventStream::HeaderValue.new(value: 'messageStop', type: 'string') },
        payload: StringIO.new('{}')
      )
      nested = { 'messageStop' => { 'stopReason' => 'end_turn' } }

      expect(streaming.send(:nest_event_under_type, nested, message)).to equal(nested)
    end
  end
end
