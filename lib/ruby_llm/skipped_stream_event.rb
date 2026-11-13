# frozen_string_literal: true

module RubyLLM
  SkippedStreamEvent = Struct.new(:provider, :model, :site, :event_type, :byte_size, :error_class,
                                  keyword_init: true)

  # A streamed event a parser could not decode and dropped. Passed to config.on_stream_parse_error;
  # its fields are documented in docs/_core_features/streaming.md and must stay stable.
  class SkippedStreamEvent
    SITES = %i[sse_data event_payload event_type_header exception_type_header content_block_delta].freeze

    def self.report(event, config)
      RubyLLM.logger.warn("RubyLLM skipped a stream event it could not parse: #{event.log_fields}")
      config.on_stream_parse_error&.call(event)
    rescue StandardError => e
      RubyLLM.logger.warn("RubyLLM on_stream_parse_error hook raised #{e.class}: #{e.message}")
    end

    def log_fields
      to_h.map { |key, value| "#{key}=#{value.nil? ? '-' : value}" }.join(' ')
    end

    # Mixed into the streaming parsers. A skip never logs the payload itself, only its size.
    module Reporting
      private

      def skip_stream_event(site, payload, event_type: nil, error: nil)
        event = SkippedStreamEvent.new(
          provider: @provider&.slug,
          model: @model&.id,
          site: site,
          event_type: event_type,
          byte_size: skipped_payload_bytesize(payload),
          error_class: error&.class&.name
        ).freeze
        @stream_accumulator&.skip_event
        SkippedStreamEvent.report(event, @config || RubyLLM.config)
        nil
      end

      def skipped_payload_bytesize(payload)
        payload.respond_to?(:bytesize) ? payload.bytesize : payload&.size
      end
    end
  end
end
