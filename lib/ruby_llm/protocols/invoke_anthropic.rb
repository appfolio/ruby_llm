# frozen_string_literal: true

module RubyLLM
  module Protocols
    # The Anthropic Messages format sent to AWS Bedrock InvokeModel and
    # InvokeModelWithResponseStream rather than translated through Converse.
    # Rendering, parsing and stream events are the Anthropic protocol's; this
    # class owns the envelope:
    #
    # - POST /model/{id}/invoke and /invoke-with-response-stream on
    #   bedrock-runtime, SigV4-signed by the provider over the final body.
    # - The body carries +anthropic_version+ and +anthropic_beta+, never
    #   +model+ or +stream+, and betas never travel as a header.
    # - The stream is an AWS eventstream whose +chunk+ events each wrap one
    #   Anthropic stream event as base64 +bytes+.
    #
    # Two provider options exist only here and are removed from the body:
    # +deferred_tools:+ names chat tools to render with +defer_loading+, and
    # +server_tools:+ holds tool definitions sent as given ahead of them.
    # +additionalModelRequestFields+, the Converse spelling, is merged into
    # the body so one set of options serves both Bedrock routes.
    class InvokeAnthropic < Anthropic
      include Bedrock::EventStream

      ANTHROPIC_VERSION = 'bedrock-2023-05-31'

      def completion_url
        "/model/#{Converse::Chat.escape_model_id(@model.id)}/invoke"
      end

      def server_tool_aliases
        nil
      end

      def count_tokens(*, **)
        raise Error, 'Bedrock InvokeModel does not support token counting'
      end

      def compact(*, **)
        raise Error, 'Bedrock InvokeModel does not support manual compaction'
      end

      private

      def apply_compaction(payload, compaction)
        compacted = super
        compacted.merge(anthropic_beta: (Array(compacted[:anthropic_beta]) + [COMPACTION_BETA]).uniq)
      end

      def apply_compaction_headers(headers, _compaction)
        headers
      end

      def supports_provider_file_references?
        false
      end

      def stream_url
        "/model/#{Converse::Chat.escape_model_id(@model.id)}/invoke-with-response-stream"
      end

      def apply_provider_tools(payload, entries)
        finalize_invoke_payload(super)
      end

      # The provider-options deep merge replaces arrays, so a caller's
      # anthropic_beta would drop the compaction beta; it is derived again
      # from the compaction edit itself.
      def finalize_invoke_payload(payload)
        payload = lift_additional_fields(payload)
        payload.delete(:model)
        payload.delete(:stream)
        betas = (Array(take_option(payload, :anthropic_beta)) + required_betas(payload)).map(&:to_s).uniq
        apply_tool_params(payload, deferred: take_option(payload, :deferred_tools),
                                   server_tools: take_option(payload, :server_tools))

        envelope = { anthropic_version: ANTHROPIC_VERSION }
        envelope[:anthropic_beta] = betas unless betas.empty?
        envelope.merge(payload)
      end

      def lift_additional_fields(payload)
        payload = payload.dup
        fields = take_option(payload, :additionalModelRequestFields)
        return payload unless fields.is_a?(Hash)

        Support::Utils.deep_merge(payload, Support::Utils.deep_symbolize_keys(fields))
      end

      def take_option(payload, key)
        string_value = payload.delete(key.to_s)
        payload.key?(key) ? payload.delete(key) : string_value
      end

      def required_betas(payload)
        edits = Array(payload.dig(:context_management, :edits))
        compaction = edits.any? { |edit| edit.is_a?(Hash) && (edit[:type] || edit['type']) == COMPACTION_EDIT_TYPE }
        compaction ? [COMPACTION_BETA] : []
      end

      def apply_tool_params(payload, deferred:, server_tools:)
        deferred = Array(deferred).map(&:to_s)
        server_tools = Array(server_tools)
        return if deferred.empty? && server_tools.empty?

        tools = mark_deferred_tools(Array(payload[:tools]), deferred)
        payload[:tools] = server_tools + tools
      end

      def mark_deferred_tools(tools, deferred)
        unknown = deferred - tools.map { |tool| tool[:name].to_s }
        raise ArgumentError, "deferred_tools names no chat tool: #{unknown.join(', ')}" unless unknown.empty?

        tools.map { |tool| deferred.include?(tool[:name].to_s) ? tool.merge(defer_loading: true) : tool }
      end

      # Only leading system messages form the top-level system; a later one
      # stays in place as a system turn, leaving the cached prefix ahead of
      # it unchanged.
      def separate_messages(messages)
        leading = messages.take_while { |msg| msg.role == :system }
        [leading, messages.drop(leading.size)]
      end

      def convert_role(role)
        case role
        when :assistant then 'assistant'
        when :system then 'system'
        else 'user'
        end
      end

      # Sorted keys make a reply replayed from a JSON column render byte for
      # byte as it was first sent, so the prompt cache prefix still matches.
      def format_messages(messages, **)
        super.map do |message|
          content = message[:content]
          content.is_a?(Array) ? message.merge(content: sort_keys(content)) : message
        end
      end

      def sort_keys(value)
        case value
        when Hash then value.sort_by { |key, _| key.to_s }.to_h.transform_values { |inner| sort_keys(inner) }
        when Array then value.map { |inner| sort_keys(inner) }
        else value
        end
      end

      def sync_response(payload, additional_headers = {})
        url = completion_url
        body = JSON.generate(payload)
        response = @connection.post(url, payload, usage: @usage_tracker) do |req|
          req.headers.merge!(@provider.sign_headers('POST', url, body))
          req.headers.merge!(without_beta_header(additional_headers))
        end
        parse_completion_response(response)
      end

      def stream_response(payload, additional_headers = {}, &)
        reset_stream_blocks
        @stream_provider_data = {}
        message = eventstream_response(payload, without_beta_header(additional_headers), &)
        message.provider_data = @stream_provider_data
        message
      end

      def without_beta_header(headers)
        headers.reject { |name, _| name.to_s.casecmp?(BETA_HEADER) }
      end

      def parse_completion_body(data, raw:)
        message = super
        message.provider_data = provider_data_for(data)
        message
      end

      def build_chunk(data)
        raise_invoke_stream_error(data) if stream_error_event?(data)
        track_stream_provider_data(data)
        super
      end

      def raise_invoke_stream_error(event)
        return raise_stream_error(event) unless event['type'] == 'error' && event['error'].is_a?(Hash)

        status = event.dig('error', 'type') == 'overloaded_error' ? 529 : 500
        message = event.dig('error', 'message') || 'Bedrock streaming error'
        response = ErrorResponse.new({ 'message' => message }, status)
        Transport::ErrorMiddleware.parse_error(provider: self, response: response)
      end

      # message_start carries input_transformations; message_delta carries
      # context_management, usage.iterations and the final cache_creation.
      def track_stream_provider_data(data)
        case data['type']
        when 'message_start' then @stream_provider_data = provider_data_for(data['message'] || {})
        when 'message_delta' then @stream_provider_data = (@stream_provider_data || {}).merge(provider_data_for(data))
        end
      end

      def provider_data_for(data)
        {
          'context_management' => data['context_management'],
          'iterations' => data.dig('usage', 'iterations'),
          'cache_creation' => data.dig('usage', 'cache_creation'),
          'input_transformations' => data['input_transformations']
        }.compact
      end

      # message_delta's usage holds the count after compaction and server
      # tool iterations; message_start's is taken before either.
      def extract_input_tokens(data)
        delta_usage(data)['input_tokens'] || super
      end

      def merge_turn_segments(segments)
        merged = super
        merged.provider_data ||= segments.last.provider_data
        merged
      end
    end
  end
end
