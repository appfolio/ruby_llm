# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # Pricing follows the registry's existing convention for Claude Opus 5: each
  # id is recorded at the same rate structure as that id's Opus 5 entry. Base
  # rates (input/output/cache_read/cache_write per million tokens) come from
  # AWS's price list ("Standard, Global" tier for Claude Opus 5.5 on Amazon
  # Bedrock), and only the eu. entry is recorded at 1.1x base, matching the
  # 1.1x the eu.anthropic.claude-opus-5 entry already uses. This does not
  # correct the separate, pre-existing question of whether us. (and other
  # non-eu regional prefixes) should also be at 1.1x on Bedrock's own price
  # list; that is out of scope here and applies equally to the existing Opus
  # 5 entries.
  {
    'claude-opus-5-5' => { input: 4, output: 20, cache_read: 0.20, cache_write: 5 },
    'global.anthropic.claude-opus-5-5' => { input: 4, output: 20, cache_read: 0.20, cache_write: 5 },
    'us.anthropic.claude-opus-5-5' => { input: 4, output: 20, cache_read: 0.20, cache_write: 5 },
    'eu.anthropic.claude-opus-5-5' => { input: 4.4, output: 22, cache_read: 0.22, cache_write: 5.5 }
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

  it 'resolves the claude-opus-5-5 alias to the anthropic provider' do
    expect(RubyLLM.models.find('claude-opus-5-5').provider).to eq('anthropic')
  end

  it 'resolves us.anthropic.claude-opus-5-5 when bedrock_region is configured' do
    entry = RubyLLM::Model::Info.new(
      id: 'us.anthropic.claude-opus-5-5',
      name: 'Claude Opus 5.5 (US)',
      provider: 'bedrock',
      metadata: { 'inference_types' => ['INFERENCE_PROFILE'] }
    )
    models = described_class.new([entry])
    allow(RubyLLM).to receive(:config).and_return(
      instance_double(RubyLLM::Configuration, bedrock_region: 'us-west-2')
    )
    found = models.find('us.anthropic.claude-opus-5-5', :bedrock)
    expect(found.id).to eq('us.anthropic.claude-opus-5-5')
    expect(found.provider).to eq('bedrock')
  end
end
