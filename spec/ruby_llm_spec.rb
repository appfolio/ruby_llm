# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM do
  describe '.logger' do
    let(:logger) { instance_double(Logger) }
    let(:log_file) { double }
    let(:log_level) { double }

    # Swap in a fresh config/logger for these examples, then restore the originals.
    # Leaving @config nil here leaks across the whole worker process: the next lazy
    # RubyLLM.config builds a bare Configuration whose model_registry_source is nil,
    # which breaks acts_as_model's registry integration (its ActiveRecordSource is
    # installed once, when ActsAs is included at load time) in any spec file that
    # happens to run later in the same process.
    around do |example|
      original_config = described_class.instance_variable_get(:@config)
      original_logger = described_class.instance_variable_get(:@logger)
      described_class.instance_variable_set(:@config, nil)
      described_class.instance_variable_set(:@logger, nil)
      example.run
    ensure
      described_class.instance_variable_set(:@config, original_config)
      described_class.instance_variable_set(:@logger, original_logger)
    end

    context 'with configuration options' do
      before do
        described_class.configure do |config|
          config.log_file = log_file
          config.log_level = log_level
        end
      end

      it 'returns a default Logger' do
        allow(Logger).to receive(:new).with(log_file, progname: 'RubyLLM', level: log_level).and_return(logger)

        expect(described_class.logger).to eq(logger)
      end
    end

    context 'with a custom logger' do
      before do
        described_class.configure do |config|
          config.logger = logger
          config.log_file = log_file
          config.log_level = log_level
        end
      end

      it 'returns a the custom Logger' do
        expect(described_class.logger).to eq(logger)
      end
    end
  end
end
