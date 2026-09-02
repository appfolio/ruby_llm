# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # cache_read_input_per_million (0.25 on the base/global/US entries, 0.275 on
  # EU) is NOT independently verified against Anthropic's live pricing page.
  # It comes from Anthropic's Claude Fable 5.1 launch announcement
  # (anthropic.com/claude-fable-and-mythos-5-1) and same-day AWS/Microsoft
  # partner blog posts reporting a ~75% cache-read price cut versus Claude
  # Fable 5's cache_read_input_per_million of 1 (i.e. roughly 1/4 of Fable 5's
  # rate, scaled per-region the same way Fable 5's own regional entries are).
  # Base input/output pricing ($10/$50) and cache_write_input_per_million
  # (12.5, i.e. 1.25x input, unchanged from Fable 5) are confirmed unchanged
  # by the same sources. Revisit the cache_read figures once models.dev or
  # Anthropic's pricing page publish real data for claude-fable-5-1.
  {
    'claude-fable-5-1' => { input: 10, output: 50, cache_read: 0.25, cache_write: 12.5 },
    'global.anthropic.claude-fable-5-1' => { input: 10, output: 50, cache_read: 0.25, cache_write: 12.5 },
    'us.anthropic.claude-fable-5-1' => { input: 10, output: 50, cache_read: 0.25, cache_write: 12.5 },
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
