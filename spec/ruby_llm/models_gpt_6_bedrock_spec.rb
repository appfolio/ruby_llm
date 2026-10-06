# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # The limits were measured live on bedrock-mantle (us-east-1). The API never states a limit.
  # Luna accepts 921,793 input tokens and rejects 922,000, which is 1,050,000 minus a 128,000
  # output reservation. Both Sols also reject 922,000.
  {
    'openai.gpt-6-luna' => {
      efforts: %w[none low medium high xhigh max],
      cost: { input: 0.11, output: 0.55, cache_read: 0.011, cache_write: 0.1375 }
    },
    'openai.gpt-6-sol' => {
      efforts: %w[none low medium high xhigh max],
      cost: { input: 2.2, output: 11.0, cache_read: 0.22, cache_write: 2.75 }
    },
    'openai.gpt-6.1-sol' => {
      efforts: %w[low medium high xhigh max],
      cost: { input: 2.2, output: 11.0, cache_read: 0.11, cache_write: 2.75 }
    }
  }.each do |id, expected|
    it "resolves #{id} from the bedrock provider with its effort values, cost and limits" do
      model = RubyLLM.models.find(id, :bedrock)

      expect(model.provider).to eq('bedrock')
      expect(model.reasoning_option_values('effort')).to eq(expected[:efforts])
      expect(model.metadata[:cost]).to eq(expected[:cost])
      expect(model.input_price_per_million).to eq(expected[:cost][:input])
      expect(model.output_price_per_million).to eq(expected[:cost][:output])
      expect(model.context_window).to eq(1_050_000)
      expect(model.max_output_tokens).to eq(128_000)
      expect(model.metadata[:limit]).to eq(context: 1_050_000, output: 128_000)
    end
  end

  it 'resolves the bare-id aliases to their bedrock-qualified ids' do
    expect(RubyLLM.models.find('gpt-6-luna').id).to eq('openai.gpt-6-luna')
    expect(RubyLLM.models.find('gpt-6-sol').id).to eq('openai.gpt-6-sol')
    expect(RubyLLM.models.find('gpt-6.1-sol').id).to eq('openai.gpt-6.1-sol')
  end
end
