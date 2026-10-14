# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  overrides = described_class.read_overrides

  def bedrock_entry(id)
    described_class.read_from_json.find { |m| m.provider == 'bedrock' && m.id == id }
  end

  def standard_pricing(model)
    model.pricing.text_tokens.standard.to_h.slice(
      :input_per_million, :output_per_million,
      :cache_read_input_per_million, :cache_write_input_per_million
    )
  end

  def expected_standard(cost)
    {
      input_per_million: cost[:input],
      output_per_million: cost[:output],
      cache_read_input_per_million: cost[:cache_read],
      cache_write_input_per_million: cost[:cache_write]
    }
  end

  it 'has overrides to check' do
    expect(overrides).not_to be_empty
  end

  # models.json must already hold each override, so the JSON file and a regenerated registry
  # agree. A hand edit to one file without the other fails here.
  overrides.each do |key, override|
    provider, id = key.split(':', 2)

    it "keeps #{key} in models.json at the override price" do
      model = described_class.read_from_json.find { |m| m.provider == provider && m.id == id }

      expect(model).not_to be_nil
      expect(model.metadata[:cost]).to eq(override[:cost])
      expect(standard_pricing(model)).to eq(expected_standard(override[:cost]))
    end
  end

  # Anthropic's pricing page (platform.claude.com/docs/en/about-claude/pricing, read
  # 2026-10-14), "Long context pricing": "Claude 4.6 and later models (except Claude Haiku 5.5)
  # ... include the full 1M token context window at standard pricing." The AWS Price List
  # (AmazonBedrockFoundationModels, us-west-2, publication 2026-10-12) has long_ctx SKUs only
  # for Claude Haiku 5.5 and Claude Sonnet 4. A tier on any of these ids would overcharge every
  # request over 200k tokens.
  %w[
    us.anthropic.claude-opus-4-8 us.anthropic.claude-opus-5 us.anthropic.claude-opus-5-5
    us.anthropic.claude-sonnet-5 us.anthropic.claude-sonnet-5-5
    us.anthropic.claude-fable-5 us.anthropic.claude-fable-5-1
  ].each do |id|
    it "prices #{id} with no long-context tier" do
      model = bedrock_entry(id)

      expect(model.context_window).to eq(1_000_000)
      expect(model.metadata[:cost].keys).to contain_exactly(:input, :output, :cache_read, :cache_write)
      expect(model.pricing.to_h[:text_tokens].keys).to eq([:standard])
    end
  end

  describe '.apply_overrides' do
    let(:cost) { { input: 4.4, output: 22, cache_read: 0.22, cache_write: 5.5 } }
    let(:tiered_model) do
      RubyLLM::Model::Info.new(
        id: 'us.anthropic.claude-opus-5-5', name: 'Claude Opus 5.5 (US)', provider: 'bedrock',
        pricing: {
          text_tokens: {
            standard: { input_per_million: 4, output_per_million: 20 },
            context_over_200k: { input_per_million: 8, output_per_million: 30 }
          }
        },
        metadata: {
          cost: { input: 4, output: 20, context_over_200k: { input: 8, output: 30 },
                  tiers: [{ input: 8, output: 30, tier: { type: 'context', size: 200_000 } }] }
        }
      )
    end

    # Model::Pricing already drops any text tier but standard and batch, so the tier that
    # matters is the one in metadata.cost; this pins both places.
    it 'drops a long-context tier from both pricing and metadata.cost' do
      model = described_class.apply_overrides([tiered_model],
                                              { 'bedrock:us.anthropic.claude-opus-5-5' => { cost: cost } })
                             .first

      expect(model.pricing.to_h[:text_tokens]).to eq(standard: expected_standard(cost))
      expect(model.metadata[:cost]).to eq(cost)
    end
  end

  describe 'refresh' do
    let(:overridden_id) { 'us.anthropic.claude-opus-5-5' }
    let(:models_dev_model) do
      RubyLLM::Model::Info.new(
        id: overridden_id, name: 'Claude Opus 5.5 (US)', provider: 'bedrock',
        pricing: { text_tokens: { standard: { input_per_million: 4, output_per_million: 20 } } },
        metadata: {
          source: 'models.dev',
          cost: { input: 4, output: 20, tiers: [{ input: 8, output: 30, tier: { type: 'context', size: 200_000 } }] }
        }
      )
    end
    let(:manual_model) do
      RubyLLM::Model::Info.new(id: 'us.anthropic.claude-hand-added', name: 'Hand added', provider: 'bedrock',
                               metadata: { source: 'manual', cost: { input: 1, output: 2 } })
    end

    before do
      allow(described_class).to receive(:read_existing_models).and_return([manual_model])
      allow(described_class).to receive_messages(
        fetch_provider_models: { models: [], fetched_providers: ['bedrock'], configured_names: [], failed: [] },
        fetch_models_dev_models: { models: [models_dev_model], fetched: true }
      )
    end

    it 'applies the override over the models.dev price, tiers included' do
      model = described_class.fetch_merged_models.find { |m| m.id == overridden_id }

      expect(model.metadata[:cost]).to eq(overrides["bedrock:#{overridden_id}"][:cost])
      expect(standard_pricing(model)).to eq(expected_standard(overrides["bedrock:#{overridden_id}"][:cost]))
    end

    it 'keeps a manual model that no source returns' do
      ids = described_class.fetch_merged_models.map(&:id)

      expect(ids).to include('us.anthropic.claude-hand-added')
    end
  end
end
