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

  it 'normalizes cache read and write tokens out of input tokens from metadata usage' do
    event = {
      'metadata' => {
        'usage' => {
          'inputTokens' => 100,
          'outputTokens' => 5,
          'cacheReadInputTokens' => 40,
          'cacheWriteInputTokens' => 10
        }
      }
    }

    chunk = streaming.send(:build_chunk, event)

    expect(chunk.input_tokens).to eq(50)
    expect(chunk.output_tokens).to eq(5)
    expect(chunk.cached_tokens).to eq(40)
    expect(chunk.cache_creation_tokens).to eq(10)
  end

  it 'reports input_tokens of 0 through the accumulator for a fully-cached streamed request' do
    accumulator = RubyLLM::StreamAccumulator.new
    usage_event = {
      'metadata' => {
        'usage' => {
          'inputTokens' => 100,
          'outputTokens' => 5,
          'cacheReadInputTokens' => 100,
          'cacheWriteInputTokens' => 0
        }
      }
    }

    accumulator.add(streaming.send(:build_chunk, usage_event))
    message = accumulator.to_message(nil)

    expect(message.input_tokens).to eq(0)
    expect(message.cached_tokens).to eq(100)
  end

  it 'does not fabricate an ephemeral cache-creation TTL breakdown, which Converse does not return' do
    event = {
      'metadata' => {
        'usage' => {
          'inputTokens' => 100,
          'outputTokens' => 5,
          'cacheReadInputTokens' => 40,
          'cacheWriteInputTokens' => 10
        }
      }
    }

    chunk = streaming.send(:build_chunk, event)

    expect(chunk.tokens&.cache_creation_ephemeral_5m).to be_nil
    expect(chunk.tokens&.cache_creation_ephemeral_1h).to be_nil
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
end
