# frozen_string_literal: true

require 'json'

module RubyLLM
  module Protocols
    class BedrockInvokeModel
      # Chat methods for the Bedrock InvokeModel API (raw Anthropic Messages format).
      module Chat
        module_function

        def completion_url
          "/model/#{escape_model_id(@model.id)}/invoke"
        end

        # rubocop:disable Metrics/ParameterLists,Metrics/PerceivedComplexity,Lint/UnusedMethodArgument
        def render_payload(messages, tools:, temperature:, model:, stream: false,
                           schema: nil, thinking: nil, citations: false, tool_prefs: nil)
          tool_prefs ||= {}
          system_messages, chat_messages = messages.partition { |msg| msg.role == :system }

          payload = {
            anthropic_version: 'bedrock-2023-05-31',
            max_tokens: model.max_tokens || 4096,
            messages: format_messages(chat_messages, thinking:)
          }

          system_blocks = format_invoke_system(system_messages)
          payload[:system] = system_blocks unless system_blocks.empty?

          payload[:temperature] = temperature unless temperature.nil?

          if tools.any?
            payload[:tools] = tools.values.map { |tool| format_invoke_tool(tool) }
            tool_choice = format_invoke_tool_choice(tool_prefs[:choice])
            payload[:tool_choice] = tool_choice if tool_choice
          end

          thinking_config = format_invoke_thinking(thinking)
          payload[:thinking] = thinking_config if thinking_config

          beta = @config.anthropic_beta
          payload[:anthropic_beta] = Array(beta) unless beta.nil? || (beta.respond_to?(:empty?) && beta.empty?)

          context_mgmt = @config.anthropic_context_management
          payload[:context_management] = context_mgmt if context_mgmt

          payload
        end
        # rubocop:enable Metrics/ParameterLists,Metrics/PerceivedComplexity,Lint/UnusedMethodArgument

        def format_messages(messages, thinking: nil)
          thinking_enabled = thinking&.enabled?
          rendered = []
          tool_result_blocks = []

          messages.each do |msg|
            if msg.tool_result?
              tool_result_blocks << format_invoke_tool_result_message(msg)
              next
            end

            unless tool_result_blocks.empty?
              rendered << { role: 'user', content: tool_result_blocks }
              tool_result_blocks = []
            end

            message = format_invoke_non_tool_message(msg, thinking_enabled:)
            rendered << message if message
          end

          rendered << { role: 'user', content: tool_result_blocks } unless tool_result_blocks.empty?
          rendered
        end

        def parse_completion_response(response)
          parse_completion_body(response.body, raw: response)
        end

        def parse_completion_body(data, raw:)
          return if data.nil? || data.empty?

          content_blocks = data['content'] || []
          usage = data['usage'] || {}

          text = invoke_extract_text(content_blocks)
          thinking_text, thinking_signature = invoke_extract_thinking(content_blocks)
          tool_calls = invoke_extract_tool_calls(content_blocks)

          Message.new(
            role: :assistant,
            content: text,
            thinking: Thinking.build(text: thinking_text, signature: thinking_signature),
            tool_calls: tool_calls,
            input_tokens: invoke_input_tokens(usage),
            output_tokens: usage['output_tokens'],
            cached_tokens: usage['cache_read_input_tokens'],
            cache_creation_tokens: usage['cache_creation_input_tokens'],
            finish_reason: data['stop_reason'],
            model_id: data['model'],
            raw: raw
          )
        end

        def invoke_input_tokens(usage)
          input = usage['input_tokens']
          return unless input

          [input.to_i - usage['cache_read_input_tokens'].to_i - usage['cache_creation_input_tokens'].to_i, 0].max
        end

        def invoke_extract_text(blocks)
          text = blocks.filter_map { |b| b['text'] if b['type'] == 'text' }.join
          text.empty? ? nil : text
        end

        def invoke_extract_thinking(blocks)
          text = +''
          signature = nil

          blocks.each do |block|
            next unless block['type'] == 'thinking'

            text << block['thinking'].to_s if block['thinking'].is_a?(String)
            signature ||= block['signature'] if block['signature'].is_a?(String)
          end

          [text.empty? ? nil : text, signature]
        end

        def invoke_extract_tool_calls(blocks)
          calls = {}

          blocks.each do |block|
            next unless block['type'] == 'tool_use'

            id = block['id']
            calls[id] = ToolCall.new(
              id: id,
              name: block['name'],
              arguments: block['input'] || {}
            )
          end

          calls.empty? ? nil : calls
        end

        def format_invoke_system(system_messages)
          system_messages.flat_map do |msg|
            content = msg.content
            if content.is_a?(RubyLLM::Content::Raw)
              content.value.is_a?(Array) ? content.value : [content.value]
            else
              Protocols::Converse::Media.format_content(content, used_document_names: {})
                                        .map { |block| converse_block_to_anthropic(block) }
            end
          end
        end

        def format_invoke_non_tool_message(msg, thinking_enabled:)
          content = format_invoke_message_content(msg, thinking_enabled:)
          return nil if content.empty?

          { role: format_role(msg.role), content: content }
        end

        # rubocop:disable Metrics/PerceivedComplexity
        def format_invoke_message_content(msg, thinking_enabled:)
          if msg.content.is_a?(RubyLLM::Content::Raw)
            raw = msg.content.value
            return raw.is_a?(Array) ? raw : [raw]
          end

          blocks = []

          if msg.role == :assistant && thinking_enabled && msg.thinking
            thinking_block = format_invoke_thinking_block(msg.thinking)
            blocks << thinking_block if thinking_block
          end

          text_blocks = invoke_extract_text_blocks(msg.content)
          blocks.concat(text_blocks)

          if msg.tool_call?
            msg.tool_calls.each_value do |tool_call|
              blocks << {
                type: 'tool_use',
                id: tool_call.id,
                name: tool_call.name,
                input: tool_call.arguments
              }
            end
          end

          blocks
        end
        # rubocop:enable Metrics/PerceivedComplexity

        # rubocop:disable Metrics/PerceivedComplexity
        def invoke_extract_text_blocks(content)
          return [] if content.nil? || (content.respond_to?(:empty?) && content.empty?)
          return [{ type: 'text', text: content.to_json }] if content.is_a?(Hash) || content.is_a?(Array)
          return [{ type: 'text', text: content.to_s }] unless content.is_a?(RubyLLM::Content)

          blocks = []
          blocks << { type: 'text', text: content.text } if content.text
          content.attachments.each do |attachment|
            converse_block = Protocols::Converse::Media.format_attachment(attachment, used_document_names: {})
            blocks << converse_block_to_anthropic(converse_block)
          end
          blocks
        end
        # rubocop:enable Metrics/PerceivedComplexity

        def converse_block_to_anthropic(block)
          # Converse media blocks are mostly Anthropic-compatible.
          # {text: "..."} → {type: "text", text: "..."}
          return { type: 'text', text: block[:text] } if block.key?(:text) && !block.key?(:type)

          block
        end

        # rubocop:disable Metrics/PerceivedComplexity
        def format_invoke_tool_result_message(msg)
          content = msg.content

          result_content = if content.is_a?(RubyLLM::Content::Raw)
                             raw = content.value
                             raw.is_a?(Array) ? raw : [raw]
                           elsif content.is_a?(String) || content.nil?
                             text = content.to_s
                             text = '(no output)' if text.empty?
                             [{ type: 'text', text: text }]
                           elsif content.is_a?(Hash) || content.is_a?(Array)
                             [{ type: 'text', text: content.to_json }]
                           else
                             [{ type: 'text', text: content.to_s }]
                           end

          {
            type: 'tool_result',
            tool_use_id: msg.tool_call_id,
            content: result_content
          }
        end
        # rubocop:enable Metrics/PerceivedComplexity

        def format_invoke_thinking_block(thinking)
          return nil unless thinking

          if thinking.text
            { type: 'thinking', thinking: thinking.text, signature: thinking.signature }.compact
          elsif thinking.signature
            { type: 'redacted_thinking', data: thinking.signature }
          end
        end

        def format_invoke_tool(tool)
          input_schema = tool.params_schema ||
                         RubyLLM::Tool::SchemaDefinition.from_parameters(tool.parameters)&.json_schema

          declaration = {
            name: tool.name,
            description: tool.description,
            input_schema: input_schema || default_invoke_input_schema
          }

          return declaration if tool.provider_params.empty?

          RubyLLM::Utils.deep_merge(declaration, tool.provider_params)
        end

        def format_invoke_tool_choice(choice)
          case choice
          when nil, :auto, :none
            nil
          when :required
            { type: 'any' }
          else
            { type: 'tool', name: choice.to_s }
          end
        end

        def format_invoke_thinking(thinking)
          return nil unless thinking&.enabled?

          budget = thinking.budget
          return { type: 'enabled', budget_tokens: budget } if budget.is_a?(Integer)

          nil
        end

        def default_invoke_input_schema
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
