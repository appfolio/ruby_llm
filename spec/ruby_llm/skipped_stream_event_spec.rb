# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::SkippedStreamEvent do
  let(:config) { RubyLLM::Configuration.new }
  let(:event) do
    described_class.new(provider: 'bedrock', model: 'claude-test', site: :event_payload,
                        event_type: 'contentBlockDelta', byte_size: 42, error_class: 'JSON::ParserError')
  end

  before { allow(RubyLLM.logger).to receive(:warn) }

  describe '.report' do
    it 'logs every field at warn level' do
      described_class.report(event, config)

      expect(RubyLLM.logger).to have_received(:warn).with(
        'RubyLLM skipped a stream event it could not parse: provider=bedrock model=claude-test ' \
        'site=event_payload event_type=contentBlockDelta byte_size=42 error_class=JSON::ParserError'
      )
    end

    it 'logs unknown fields as a dash' do
      described_class.report(described_class.new(site: :sse_data, byte_size: 4), config)

      expect(RubyLLM.logger).to have_received(:warn).with(
        a_string_including('provider=- model=- site=sse_data event_type=- byte_size=4 error_class=-')
      )
    end

    it 'calls the configured hook with the event' do
      received = []
      config.on_stream_parse_error = ->(skipped) { received << skipped }

      described_class.report(event, config)

      expect(received).to eq([event])
    end

    it 'only logs when no hook is configured' do
      expect(config.on_stream_parse_error).to be_nil
      expect { described_class.report(event, config) }.not_to raise_error
      expect(RubyLLM.logger).to have_received(:warn).once
    end

    it 'rescues and logs an exception the hook raises' do
      config.on_stream_parse_error = ->(_skipped) { raise ArgumentError, 'metrics down' }

      expect { described_class.report(event, config) }.not_to raise_error
      expect(RubyLLM.logger).to have_received(:warn)
        .with('RubyLLM on_stream_parse_error hook raised ArgumentError: metrics down')
    end
  end

  describe described_class::Reporting do
    let(:accumulator) { RubyLLM::StreamAccumulator.new }
    let(:received) { [] }
    let(:reporter) do
      Object.new.tap do |object|
        object.extend(described_class)
        object.instance_variable_set(:@config, config)
        object.instance_variable_set(:@stream_accumulator, accumulator)
        object.instance_variable_set(:@provider, instance_double(RubyLLM::Provider, slug: 'bedrock'))
        object.instance_variable_set(:@model, instance_double(RubyLLM::Model::Info, id: 'claude-test'))
      end
    end

    before { config.on_stream_parse_error = ->(skipped) { received << skipped } }

    it 'builds a frozen event from the protocol, counts it and reports it without the payload' do
      result = reporter.send(:skip_stream_event, :event_payload, 'secret payload',
                             event_type: 'chunk', error: JSON::ParserError.new('secret payload'))

      expect(result).to be_nil
      expect(received.first.to_h).to eq(provider: 'bedrock', model: 'claude-test', site: :event_payload,
                                        event_type: 'chunk', byte_size: 14, error_class: 'JSON::ParserError')
      expect(received.first).to be_frozen
      expect(accumulator.skipped_events).to eq(1)
      expect(RubyLLM.logger).not_to have_received(:warn).with(a_string_including('secret'))
    end

    it 'measures IO payloads by size and counts bytes, not characters' do
      reporter.send(:skip_stream_event, :event_type_header, StringIO.new('{"é":1}'))

      expect(received.first.byte_size).to eq(8)
      expect(received.first.error_class).to be_nil
    end

    it 'falls back to the global config outside a protocol' do
      bare = Object.new.tap { |object| object.extend(described_class) }
      global = []
      RubyLLM.config.on_stream_parse_error = ->(skipped) { global << skipped }

      bare.send(:skip_stream_event, :sse_data, 'x')

      expect(global.map(&:site)).to eq([:sse_data])
    ensure
      RubyLLM.config.on_stream_parse_error = nil
    end
  end
end
