# frozen_string_literal: true

require 'json'

module RubyLLM
  module Protocols
    class BedrockInvokeModel
      # Chat methods for the Bedrock InvokeModel API (raw Anthropic Messages format).
      module Chat
        BEDROCK_INLINE_DOCUMENT_LIMIT = 4_500_000

        module_function

        def completion_url
          "/model/#{escape_model_id(@model.id)}/invoke"
        end

        # An application inference profile ARN contains "/" which must be percent-encoded
        # so it forms a single URL path segment. See Converse::Chat#escape_model_id for the
        # full explanation; the same logic applies to InvokeModel URLs.
        def escape_model_id(model_id)
          model_id.to_s.gsub('/', '%2F')
        end

        def warn_unsupported_schema(model)
          RubyLLM.logger.warn(
            'RubyLLM does not support structured output (schema:) on the BedrockInvokeModel path. ' \
            "Ignoring schema for #{model.id}."
          )
        end

        def warn_unsupported_citations(model)
          RubyLLM.logger.warn(
            'RubyLLM does not support citations on the BedrockInvokeModel path. ' \
            "Ignoring with_citations for #{model.id}."
          )
        end

        def supports_provider_file_references?
          true
        end

        def default_large_file_upload_threshold
          BEDROCK_INLINE_DOCUMENT_LIMIT
        end

        def provider_file_attachable?(attachment)
          attachment.pdf? || attachment.document? || attachment.text?
        end

        # rubocop:disable Metrics/ParameterLists,Lint/UnusedMethodArgument
        def render_payload(messages, tools:, temperature:, model:, stream: false,
                           schema: nil, thinking: nil, citations: false, tool_prefs: nil)
          warn_unsupported_schema(model) if schema
          warn_unsupported_citations(model) if citations
          tool_prefs ||= {}
          system_messages, chat_messages = messages.partition { |msg| msg.role == :system }

          payload = {
            anthropic_version: 'bedrock-2023-05-31',
            max_tokens: model.max_tokens || 4096,
            messages: format_messages(chat_messages)
          }

          system_blocks = format_system(system_messages)
          payload[:system] = system_blocks unless system_blocks.empty?

          payload[:temperature] = temperature unless temperature.nil?

          add_tool_fields(payload, tools, tool_prefs)
          add_thinking_fields(payload, thinking)
          add_beta_fields(payload)

          payload
        end
        # rubocop:enable Metrics/ParameterLists,Lint/UnusedMethodArgument

        def add_tool_fields(payload, tools, tool_prefs)
          return unless tools.any?

          payload[:tools] = tools.values.map { |tool| format_tool(tool) }
          tool_choice = format_tool_choice(tool_prefs[:choice])
          payload[:tool_choice] = tool_choice if tool_choice
        end

        def add_thinking_fields(payload, thinking)
          fields = format_thinking_fields(thinking)
          payload.merge!(fields) if fields
        end

        def add_beta_fields(payload)
          beta = @config.anthropic_beta
          payload[:anthropic_beta] = Array(beta) if beta

          context_mgmt = @config.anthropic_context_management
          payload[:context_management] = context_mgmt if context_mgmt
        end

        def parse_completion_response(response)
          parse_completion_body(response.body, raw: response)
        end

        def parse_completion_body(data, raw:)
          return if data.nil? || data.empty?

          content_blocks = data['content'] || []
          usage = data['usage'] || {}

          Message.new(
            role: :assistant,
            content: parse_text_content(content_blocks),
            thinking: parse_thinking(content_blocks),
            tool_calls: parse_tool_calls(content_blocks),
            input_tokens: input_tokens(usage),
            output_tokens: usage['output_tokens'],
            cached_tokens: usage['cache_read_input_tokens'],
            cache_creation_tokens: usage['cache_creation_input_tokens'],
            finish_reason: data['stop_reason'],
            model_id: data['model'],
            raw: raw
          )
        end

        def input_tokens(usage)
          input = usage['input_tokens']
          return unless input

          [input.to_i - usage['cache_read_input_tokens'].to_i - usage['cache_creation_input_tokens'].to_i, 0].max
        end

        def format_messages(messages)
          rendered = []
          tool_result_blocks = []

          messages.each do |msg|
            if msg.tool_result?
              tool_result_blocks << format_tool_result_block(msg)
              next
            end

            unless tool_result_blocks.empty?
              rendered << { role: 'user', content: tool_result_blocks }
              tool_result_blocks = []
            end

            formatted = format_non_tool_message(msg)
            rendered << formatted if formatted
          end

          rendered << { role: 'user', content: tool_result_blocks } unless tool_result_blocks.empty?
          rendered
        end

        def format_non_tool_message(msg)
          content = format_message_content(msg)
          return nil if content.empty?

          { role: format_role(msg.role), content: content }
        end

        def format_message_content(msg)
          if msg.content.is_a?(RubyLLM::Content::Raw)
            raw = msg.content.value
            return raw.is_a?(Array) ? raw : [raw]
          end

          blocks = []

          if msg.role == :assistant
            thinking_blocks = format_thinking_blocks(msg.thinking)
            blocks.concat(thinking_blocks) if thinking_blocks
          end

          blocks.concat(format_text_and_media(msg.content))

          if msg.tool_call?
            msg.tool_calls.each_value do |tool_call|
              blocks << { type: 'tool_use', id: tool_call.id, name: tool_call.name, input: tool_call.arguments }
            end
          end

          blocks
        end

        def format_text_and_media(content) # rubocop:disable Metrics/PerceivedComplexity
          return [] if content.nil? || (content.respond_to?(:empty?) && content.empty?)
          return [{ type: 'text', text: content.to_json }] if content.is_a?(Hash) || content.is_a?(Array)
          return [{ type: 'text', text: content }] unless content.is_a?(RubyLLM::Content)

          blocks = []
          blocks << build_text_block(content.text) if content.text
          content.attachments.each { |att| blocks << format_attachment(att) }
          blocks
        end

        def build_text_block(text)
          { type: 'text', text: text }
        end

        def format_attachment(attachment)
          case attachment.type
          when :image
            format_image_attachment(attachment)
          when :pdf, :document
            format_document_attachment(attachment)
          when :text
            { type: 'text', text: attachment.for_llm }
          else
            raise UnsupportedAttachmentError, attachment.mime_type
          end
        end

        def format_image_attachment(attachment)
          # Bedrock InvokeModel (Anthropic Messages format) does not support the `url`
          # image source type — AWS rejects it. Only base64 is accepted.
          if attachment.url?
            raise UnsupportedAttachmentError,
                  'Bedrock InvokeModel does not support URL image sources; ' \
                  "provide a local file or IO instead. (attachment: #{attachment.source})"
          end

          {
            type: 'image',
            source: { type: 'base64', media_type: attachment.mime_type, data: attachment.encoded }
          }
        end

        def format_document_attachment(attachment)
          {
            type: 'document',
            source: { type: 'base64', media_type: attachment.mime_type, data: attachment.encoded }
          }
        end

        def format_tool_result_block(msg)
          {
            type: 'tool_result',
            tool_use_id: msg.tool_call_id,
            content: format_tool_result_content(msg.content)
          }
        end

        def format_tool_result_content(content)
          return content.value if content.is_a?(RubyLLM::Content::Raw)
          return [{ type: 'text', text: content.to_json }] if content.is_a?(Hash) || content.is_a?(Array)
          return content_to_blocks_or_fallback(content) if content.is_a?(RubyLLM::Content)

          text = content.to_s
          text = '(no output)' if text.empty?
          [{ type: 'text', text: text }]
        end

        def content_to_blocks_or_fallback(content)
          blocks = []
          blocks << { type: 'text', text: content.text } unless content.text.to_s.empty?
          content.attachments.each { |att| blocks << format_attachment(att) }
          blocks.empty? ? [{ type: 'text', text: '(no output)' }] : blocks
        end

        def format_role(role)
          case role
          when :assistant then 'assistant'
          else 'user'
          end
        end

        def format_system(messages)
          messages.flat_map { |msg| format_text_and_media(msg.content) }
        end

        def format_tool(tool)
          input_schema = tool.params_schema ||
                         RubyLLM::Tool::SchemaDefinition.from_parameters(tool.parameters)&.json_schema

          declaration = {
            name: tool.name,
            description: tool.description,
            input_schema: input_schema || default_input_schema
          }

          return declaration if tool.provider_params.empty?

          RubyLLM::Utils.deep_merge(declaration, tool.provider_params)
        end

        def format_tool_choice(choice)
          case choice
          when :auto then { type: 'auto' }
          when :required then { type: 'any' }
          when nil, :none then nil
          else { type: 'tool', name: choice.to_s }
          end
        end

        def format_thinking_fields(thinking)
          return nil unless thinking&.enabled?

          budget = thinking.budget
          if budget.is_a?(Integer)
            # An explicit integer budget is a deliberate manual-budget opt-in and must use
            # type: 'enabled'. Only meaningful on models that still accept budget_tokens.
            { thinking: { type: 'enabled', budget_tokens: budget } }
          else
            effort = thinking.effort.to_s
            return nil if effort.empty? || effort == 'none'

            # Manual thinking: { type: 'enabled' } returns a 400 on current Anthropic models
            # (Sonnet 5, Opus 4.7/4.8, Fable 5, ...). Effort-based extended thinking must use
            # adaptive thinking with the effort carried in output_config.
            { thinking: { type: 'adaptive' }, output_config: { effort: effort } }
          end
        end

        # Anthropic requires every thinking/redacted_thinking block from the most recent
        # assistant turn to be replayed verbatim. When thinking.blocks is present, it holds
        # the exact original blocks and must be replayed in order; text/signature are kept
        # only for display purposes.
        def format_thinking_blocks(thinking)
          return nil unless thinking

          return thinking.blocks if thinking.blocks

          block = format_single_thinking_block(thinking)
          block ? [block] : nil
        end

        # A text-only block with no signature (e.g. a turn truncated mid-thinking) is dropped:
        # Anthropic rejects replayed thinking blocks without a valid signature, so a
        # signature-less reconstruction is worse than omitting the block entirely. This mirrors
        # the streaming path's drop-on-truncation behavior (see streaming.rb).
        def format_single_thinking_block(thinking)
          if thinking.text && thinking.signature
            { type: 'thinking', thinking: thinking.text, signature: thinking.signature }
          elsif thinking.signature
            { type: 'redacted_thinking', data: thinking.signature }
          end
        end

        def parse_text_content(content_blocks)
          text = content_blocks.filter_map do |block|
            block['text'] if block['type'] == 'text' && block['text'].is_a?(String)
          end.join
          text.empty? ? nil : text
        end

        THINKING_BLOCK_TYPES = %w[thinking redacted_thinking].freeze
        private_constant :THINKING_BLOCK_TYPES

        # Returns a Thinking instance whose blocks field is the source of truth for replay;
        # text/signature are merged across all blocks and kept only for display purposes.
        def parse_thinking(content_blocks)
          raw_blocks = content_blocks.select { |b| THINKING_BLOCK_TYPES.include?(b['type']) }
          return nil if raw_blocks.empty?

          text, signature = merge_thinking_blocks(raw_blocks)
          Thinking.build(text: text, signature: signature, blocks: raw_blocks)
        end

        def merge_thinking_blocks(raw_blocks)
          text = +''
          signature = nil
          raw_blocks.each do |b|
            text << b['thinking'] if b['thinking'].is_a?(String)
            signature ||= b['signature'] || b['data']
          end
          [text.empty? ? nil : text, signature]
        end

        def parse_tool_calls(content_blocks)
          tool_calls = {}

          content_blocks.each do |block|
            next unless block['type'] == 'tool_use'

            tool_calls[block['id']] = ToolCall.new(
              id: block['id'],
              name: block['name'],
              arguments: block['input'] || {}
            )
          end

          tool_calls.empty? ? nil : tool_calls
        end

        def default_input_schema
          {
            'type' => 'object',
            'properties' => {},
            'required' => []
          }
        end
      end
    end
  end
end
