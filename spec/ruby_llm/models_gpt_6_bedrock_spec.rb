# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # The limits were measured live on bedrock-mantle (us-east-1). The API never states a limit.
  # Luna accepts 921,793 input tokens and rejects 922,000, which is 1,050,000 minus a 128,000
  # output reservation. Both Sols also reject 922,000. 6.1 Terra, measured 2026-11-09, accepts
  # 921,007 and rejects 923,500. 6.2 Sol and 6.2 Astra, measured 2026-11-18, accept 904,217
  # and reject about 935,000.
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
    },
    'openai.gpt-6.1-terra' => {
      efforts: %w[none low medium high xhigh max],
      cost: { input: 1.1, output: 5.5, cache_read: 0.055, cache_write: 1.375 }
    },
    'openai.gpt-6.2-sol' => {
      efforts: %w[low medium high xhigh max],
      cost: { input: 2.2, output: 11.0, cache_read: 0.11, cache_write: 2.75 }
    },
    'openai.gpt-6.2-astra' => {
      efforts: %w[low medium high xhigh max],
      cost: { input: 8.8, output: 44.0, cache_read: 0.44, cache_write: 11.0 }
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
    expect(RubyLLM.models.find('gpt-6.1-terra').id).to eq('openai.gpt-6.1-terra')
    expect(RubyLLM.models.find('gpt-6.2-sol').id).to eq('openai.gpt-6.2-sol')
    expect(RubyLLM.models.find('gpt-6.2-astra').id).to eq('openai.gpt-6.2-astra')
  end

  # Both 6.2 models reject input_file PDFs on mantle ("does not support PDF input").
  %w[openai.gpt-6.2-sol openai.gpt-6.2-astra].each do |id|
    it "lists text and image input for #{id}, and not pdf" do
      expect(RubyLLM.models.find(id, :bedrock).modalities.input).to eq(%w[text image])
    end
  end
end
