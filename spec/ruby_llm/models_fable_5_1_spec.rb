# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # Prices are per million tokens (input/output/cache_read/cache_write).
  # Anthropic's pricing page (platform.claude.com/docs/en/about-claude/pricing,
  # read 2026-10-14) lists Claude Fable 5.1 at 10 / 50 / 0.25 / 12.5 on global
  # endpoints. The 0.25 cache read is verified by its footnote 1: "Cache hits
  # and refreshes on Claude Fable 5.1 and Claude Mythos 5.1 are priced at
  # 0.025x the base input price." The eu. and us. entries are regional
  # endpoints and are recorded at 1.1x: the same page says "Regional and
  # multi-region endpoints include a 10% premium over global endpoints."
  {
    'claude-fable-5-1' => { input: 10, output: 50, cache_read: 0.25, cache_write: 12.5 },
    'global.anthropic.claude-fable-5-1' => { input: 10, output: 50, cache_read: 0.25, cache_write: 12.5 },
    'us.anthropic.claude-fable-5-1' => { input: 11, output: 55, cache_read: 0.275, cache_write: 13.75 },
    'eu.anthropic.claude-fable-5-1' => { input: 11, output: 55, cache_read: 0.275, cache_write: 13.75 }
  }.each do |id, cost|
    it "registers #{id} with the documented effort values and pricing" do
      # Look up by raw id rather than RubyLLM.models.find(id, :bedrock) — bedrock
      # lookups renormalize the region prefix to the configured bedrock_region,
      # which would mask the EU-specific entry's own pricing.
      model = RubyLLM.models.all.find { |m| m.id == id }

      expect(model).not_to be_nil
      expect(model.reasoning_option_values('effort')).to eq(%w[low medium high xhigh max])
      expect(model.metadata[:cost]).to include(cost)
    end
  end

  it 'resolves the claude-fable-5-1 alias to the anthropic provider' do
    expect(RubyLLM.models.find('claude-fable-5-1').provider).to eq('anthropic')
  end

  it 'resolves us.anthropic.claude-fable-5-1 when bedrock_region is configured' do
    entry = RubyLLM::Model::Info.new(
      id: 'us.anthropic.claude-fable-5-1',
      name: 'Claude Fable 5.1 (US)',
      provider: 'bedrock',
      metadata: { 'inference_types' => ['INFERENCE_PROFILE'] }
    )
    models = described_class.new([entry])
    allow(RubyLLM).to receive(:config).and_return(
      instance_double(RubyLLM::Configuration, bedrock_region: 'us-west-2')
    )
    found = models.find('us.anthropic.claude-fable-5-1', :bedrock)
    expect(found.id).to eq('us.anthropic.claude-fable-5-1')
    expect(found.provider).to eq('bedrock')
  end
end
