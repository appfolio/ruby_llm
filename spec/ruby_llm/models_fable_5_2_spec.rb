# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # Prices are per million tokens (input/output/cache_read/cache_write), from the
  # Bedrock agreement offer rate card for Claude Fable 5.2 (offer-agsprgw6z3dg6,
  # read 2026-11-11). The global rate, which matches first-party pricing, is
  # 8 / 40 / 0.20 / 10. The US rate, used by us. inference profiles, is
  # 8.8 / 44 / 0.22 / 11. The cache read is 0.025x the base input price, as for
  # Fable 5.1.
  {
    'claude-fable-5-2' => { input: 8, output: 40, cache_read: 0.2, cache_write: 10 },
    'global.anthropic.claude-fable-5-2' => { input: 8, output: 40, cache_read: 0.2, cache_write: 10 },
    'us.anthropic.claude-fable-5-2' => { input: 8.8, output: 44, cache_read: 0.22, cache_write: 11 }
  }.each do |id, cost|
    it "registers #{id} with the documented effort values, pricing and limits" do
      # Look up by raw id rather than RubyLLM.models.find(id, :bedrock): bedrock
      # lookups renormalize the region prefix to the configured bedrock_region.
      model = RubyLLM.models.all.find { |m| m.id == id }

      expect(model).not_to be_nil
      expect(model.reasoning_option_values('effort')).to eq(%w[low medium high xhigh max])
      expect(model.metadata[:cost]).to eq(cost)
      expect(model.context_window).to eq(1_000_000)
      expect(model.max_output_tokens).to eq(128_000)
    end
  end

  it 'resolves the claude-fable-5-2 alias to the anthropic provider' do
    found = RubyLLM.models.find('claude-fable-5-2')

    expect(found.id).to eq('claude-fable-5-2')
    expect(found.provider).to eq('anthropic')
  end

  it 'resolves us.anthropic.claude-fable-5-2 on bedrock when bedrock_region is configured' do
    RubyLLM.config.bedrock_region = 'us-west-2'

    found = RubyLLM.models.find('us.anthropic.claude-fable-5-2', :bedrock)

    expect(found.id).to eq('us.anthropic.claude-fable-5-2')
    expect(found.provider).to eq('bedrock')
    expect(found.input_price_per_million).to eq(8.8)
  end
end
