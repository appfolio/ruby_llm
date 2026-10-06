# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::Responses::Streaming do
  let(:protocol) { RubyLLM::Protocols::Responses.allocate }
  let(:second_reasoning) { { 'type' => 'reasoning', 'id' => 'rs_2', 'encrypted_content' => 'ENC2' } }
  let(:first_reasoning) { { 'type' => 'reasoning', 'id' => 'rs_1', 'encrypted_content' => 'ENC1' } }

  def build_chunk(data)
    protocol.send(:build_chunk, data)
  end

  it 'streams output text deltas as content' do
    chunk = build_chunk({ 'type' => 'response.output_text.delta', 'delta' => 'Hel' })

    expect(chunk.content).to eq('Hel')
  end

  it 'streams reasoning summary deltas as thinking' do
    chunk = build_chunk({ 'type' => 'response.reasoning_summary_text.delta', 'delta' => 'hmm' })

    expect(chunk.thinking.text).to eq('hmm')
  end

  it 'accumulates a function call across item and argument events' do
    accumulator = RubyLLM::StreamAccumulator.new

    accumulator.add build_chunk({
                                  'type' => 'response.output_item.added',
                                  'output_index' => 1,
                                  'item' => { 'type' => 'function_call', 'call_id' => 'call_1', 'name' => 'weather' }
                                })
    accumulator.add build_chunk({
                                  'type' => 'response.function_call_arguments.delta',
                                  'output_index' => 1,
                                  'delta' => '{"city":'
                                })
    accumulator.add build_chunk({
                                  'type' => 'response.function_call_arguments.delta',
                                  'output_index' => 1,
                                  'delta' => '"Berlin"}'
                                })

    message = accumulator.to_message(instance_double(Faraday::Response, body: {}))

    expect(message.tool_calls.keys).to eq(['call_1'])
    expect(message.tool_calls['call_1'].name).to eq('weather')
    expect(message.tool_calls['call_1'].arguments).to eq({ 'city' => 'Berlin' })
  end

  it 'captures encrypted reasoning from completed items' do
    chunk = build_chunk({
                          'type' => 'response.output_item.done',
                          'item' => { 'type' => 'reasoning', 'encrypted_content' => 'ENCRYPTED' }
                        })

    expect(chunk.thinking.signature).to eq('ENCRYPTED')
  end

  def stream_reasoning_items(model_id)
    protocol.instance_variable_set(:@model, instance_double(RubyLLM::Model::Info, id: model_id))
    accumulator = RubyLLM::StreamAccumulator.new
    [first_reasoning, second_reasoning].each_with_index do |item, index|
      accumulator.add build_chunk({ 'type' => 'response.output_item.done', 'output_index' => index * 2,
                                    'item' => item })
    end
    accumulator.to_message(instance_double(Faraday::Response, body: {}))
  end

  it 'accumulates every completed reasoning item as a thinking block, in order, for GPT-6' do
    message = stream_reasoning_items('openai.gpt-6-sol')

    expect(message.thinking.signature).to eq('ENC1')
    expect(message.thinking.blocks).to eq([first_reasoning, second_reasoning])
  end

  it 'keeps only the first signature and no blocks for GPT-5.6 Sol' do
    message = stream_reasoning_items('openai.gpt-5.6-sol')

    expect(message.thinking.signature).to eq('ENC1')
    expect(message.thinking.blocks).to be_nil
  end

  it 'reads usage and model from the completed event' do
    chunk = build_chunk({
                          'type' => 'response.completed',
                          'response' => {
                            'model' => 'gpt-5-nano',
                            'status' => 'completed',
                            'usage' => {
                              'input_tokens' => 10,
                              'output_tokens' => 7,
                              'input_tokens_details' => { 'cached_tokens' => 4 },
                              'output_tokens_details' => { 'reasoning_tokens' => 3 }
                            }
                          }
                        })

    expect(chunk.model_id).to eq('gpt-5-nano')
    expect(chunk.input_tokens).to eq(6)
    expect(chunk.output_tokens).to eq(7)
    expect(chunk.cached_tokens).to eq(4)
    expect(chunk.thinking_tokens).to eq(3)
    expect(chunk.finish_reason).to be_nil
  end

  it 'does not synthesize finish_reason for completed function-call responses' do
    chunk = build_chunk({
                          'type' => 'response.completed',
                          'response' => {
                            'model' => 'gpt-5-nano',
                            'status' => 'completed',
                            'output' => [
                              { 'type' => 'function_call', 'call_id' => 'call_1', 'name' => 'weather',
                                'arguments' => '{}' }
                            ]
                          }
                        })

    expect(chunk.finish_reason).to be_nil
  end

  it 'preserves incomplete_details reason on completed events' do
    chunk = build_chunk({
                          'type' => 'response.completed',
                          'response' => {
                            'model' => 'gpt-5-nano',
                            'status' => 'incomplete',
                            'incomplete_details' => { 'reason' => 'max_output_tokens' }
                          }
                        })

    expect(chunk.finish_reason).to eq('max_output_tokens')
  end
end
