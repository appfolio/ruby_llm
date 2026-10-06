# frozen_string_literal: true

module RubyLLM
  module Protocols
    class Responses
      # Chat methods of the OpenAI Responses API
      module Chat
        def completion_url
          'responses'
        end

        def compaction_url
          "#{completion_url}/compact"
        end

        OPENAI_INLINE_FILE_LIMIT = 50 * 1024 * 1024
        OPENAI_FILE_UPLOAD_LIMIT = 512 * 1024 * 1024

        # Output item types RubyLLM models directly. A GPT-6 reply holding any other type (e.g.
        # compaction) keeps its full item list as OutputItems so it replays item-for-item.
        CLIENT_OUTPUT_ITEM_TYPES = %w[message reasoning function_call].freeze

        # Full output-item handling (every reasoning item, OutputItems, compaction, cache-write
        # accounting) applies only to GPT-6 ids; every other model keeps its existing behaviour.
        GPT6_MODEL_PATTERN = /\A(?:openai\.)?gpt-6/

        module_function

        # rubocop:disable-next Metrics/ParameterLists,Metrics/PerceivedComplexity
        def render_payload(messages, tools:, temperature:, model:, stream: false, schema: nil,
                           thinking: nil, citations: false, tool_prefs: nil)
          warn_unsupported_citations(model) if citations && !model.citations?
          tool_prefs ||= {}
          payload = {
            model: model.id,
            input: format_input(messages, gpt6: gpt6_model?(model.id)),
            instructions: format_instructions(messages),
            stream: stream,
            store: false
          }.compact

          payload[:include] = ['reasoning.encrypted_content'] if reasoning_model?(model.id)
          payload[:temperature] = temperature unless temperature.nil?

          if tools.any?
            payload[:tools] = tools.map { |_, tool| tool_for(tool) }
            payload[:tool_choice] = build_tool_choice(tool_prefs[:choice]) unless tool_prefs[:choice].nil?
            payload[:parallel_tool_calls] = tool_prefs[:calls] == :many unless tool_prefs[:calls].nil?
          end

          payload[:text] = { format: schema_format(schema) } if schema

          effort = resolve_effort(thinking)
          payload[:reasoning] = { effort: effort } if effort

          payload
        end

        def parse_completion_response(response)
          parse_completion_body(response.body, raw: response)
        end

        def parse_completion_body(data, raw:)
          return if data.nil? || data.empty?

          raise Error.new(raw, data.dig('error', 'message')) if data.dig('error', 'message')

          output = data['output'] || []
          gpt6 = gpt6_model?(model&.id || data['model'])
          content = parse_reply_content(output, gpt6:)

          Message.new(
            role: :assistant,
            content: content,
            citations: parse_output_citations(output, reply_text(content)),
            thinking: parse_thinking(output, gpt6:),
            tool_calls: parse_function_calls(output),
            model_id: data['model'],
            raw: raw,
            finish_reason: data.dig('incomplete_details', 'reason'),
            **parse_usage(data['usage'] || {}, gpt6:)
          )
        end

        def gpt6_model?(model_id)
          model_id.to_s.match?(GPT6_MODEL_PATTERN)
        end

        def parse_thinking(output, gpt6:)
          text = parse_reasoning_summary(output)
          signature = parse_reasoning_signature(output)
          return Thinking.build(text:, signature:) unless gpt6

          Thinking.build(text:, signature:, blocks: parse_reasoning_items(output))
        end

        def parse_output_citations(output, content)
          annotations = output.select { |item| item['type'] == 'message' }.flat_map do |message|
            Array(message['content']).flat_map { |part| Array(part['annotations']) }
          end

          parse_annotations(annotations, content)
        end

        def reasoning_model?(model_id)
          model_id.match?(/\A(?:openai\.)?(?:o\d|gpt-5|gpt-6)/)
        end

        def render_compaction_payload(messages)
          {
            model: model.id,
            input: format_input(messages, gpt6: true),
            instructions: format_instructions(messages)
          }.compact
        end

        def parse_compaction_response(response)
          body = response.body
          unless body.is_a?(Hash) && body['object'] == 'response.compaction'
            raise Error.new(response, 'The provider returned an invalid compaction response')
          end

          Message.new(
            role: :assistant,
            content: OutputItems.new(body),
            model_id: body['model'] || model.id,
            raw: response,
            finish_reason: 'stop',
            **parse_usage(body['usage'] || {}, gpt6: true)
          )
        end

        def parse_reply_content(output, gpt6:)
          gpt6 && output_items?(output) ? OutputItems.new(output) : parse_output_text(output)
        end

        def reply_text(content)
          content.is_a?(OutputItems) ? content.text : content
        end

        # A reply keeps its full item list when it holds an item type RubyLLM does not model,
        # or interim commentary, so neither is lost or merged into the answer on replay.
        def output_items?(output)
          output.any? do |item|
            !CLIENT_OUTPUT_ITEM_TYPES.include?(item['type']) ||
              (item['type'] == 'message' && item['phase'] == 'commentary')
          end
        end

        # GPT-6 input_tokens includes both cache reads and cache writes; each is billed at its own
        # rate, so both come out of the plain input count.
        def parse_usage(usage, gpt6: false)
          return parse_legacy_usage(usage) unless gpt6

          details = usage['input_tokens_details'] || {}
          cached = details['cached_tokens']
          cache_writes = details['cache_write_tokens']
          input = usage['input_tokens']

          {
            input_tokens: input && [input.to_i - cached.to_i - cache_writes.to_i, 0].max,
            output_tokens: usage['output_tokens'],
            cached_tokens: cached,
            cache_creation_tokens: cache_writes,
            thinking_tokens: usage.dig('output_tokens_details', 'reasoning_tokens')
          }
        end

        def parse_legacy_usage(usage)
          cached = usage.dig('input_tokens_details', 'cached_tokens')
          input = usage['input_tokens']

          {
            input_tokens: input && [input.to_i - cached.to_i, 0].max,
            output_tokens: usage['output_tokens'],
            cached_tokens: cached,
            thinking_tokens: usage.dig('output_tokens_details', 'reasoning_tokens')
          }
        end

        def schema_format(schema)
          {
            type: 'json_schema',
            name: schema[:name],
            schema: schema[:schema],
            strict: schema[:strict]
          }
        end

        def format_instructions(messages)
          instructions = messages.select { |msg| msg.role == :system }.map do |msg|
            msg.content.is_a?(Content) ? msg.content.text : msg.content.to_s
          end

          instructions.empty? ? nil : instructions.join("\n\n")
        end

        # A compaction stands in for all history before it, so the input built so far is replaced
        # by the compacted items: a /compact response.compaction object's output, or a reply's own
        # items from its last compaction item on. Under store: false the server does not drop that
        # history itself. System messages travel as instructions, not input.
        def format_input(messages, gpt6: false)
          conversation = messages.reject { |msg| msg.role == :system }
          return conversation.flat_map { |msg| format_item(msg) } unless gpt6

          conversation.each_with_object([]) do |msg, input|
            compacted = compaction_items(msg.content)
            if compacted
              input.replace(compacted)
            else
              input.concat([format_item(msg, gpt6: true)].flatten(1))
            end
          end
        end

        def compaction_items(content)
          return unless content.is_a?(RubyLLM::Content::Raw)

          value = content.value
          return value['output'] if value.is_a?(Hash) && value['object'] == 'response.compaction'
          return unless value.is_a?(Array)

          last = value.rindex { |item| item.is_a?(Hash) && item['type'] == 'compaction' }
          value[last..] if last
        end

        def format_item(msg, gpt6: false)
          case msg.role
          when :tool
            {
              type: 'function_call_output',
              call_id: msg.tool_call_id,
              output: format_content(msg.content)
            }
          when :assistant
            gpt6 ? format_gpt6_assistant_items(msg) : format_assistant_items(msg)
          else
            { role: 'user', content: format_content(msg.content) }
          end
        end

        def format_assistant_items(msg)
          items = []
          items << format_reasoning_item(msg.thinking) if msg.thinking&.signature
          items << { role: 'assistant', content: format_output_content(msg) } unless empty_content?(msg.content)
          items.concat(format_function_call_items(msg.tool_calls)) if msg.tool_call?
          items
        end

        # A reply kept as raw output items replays them verbatim, in order. The check takes the
        # base Content::Raw, which is what ActiveRecord-backed chats read stored raw content as.
        def format_gpt6_assistant_items(msg)
          return msg.content.value if msg.content.is_a?(RubyLLM::Content::Raw) && msg.content.value.is_a?(Array)

          items = format_reasoning_items(msg.thinking)
          items << { role: 'assistant', content: format_output_content(msg) } unless empty_content?(msg.content)
          items.concat(format_function_call_items(msg.tool_calls)) if msg.tool_call?
          items
        end

        # Every reasoning item from the reply replays as received; a message carrying only a
        # signature (e.g. restored without its blocks) replays a single rebuilt item as before.
        def format_reasoning_items(thinking)
          return [] unless thinking

          blocks = Array(thinking.blocks).select { |block| reasoning_block?(block) }
          return blocks unless blocks.empty?

          thinking.signature ? [format_reasoning_item(thinking)] : []
        end

        def reasoning_block?(block)
          block.is_a?(Hash) && (block['type'] || block[:type]).to_s == 'reasoning'
        end

        def format_reasoning_item(thinking)
          {
            type: 'reasoning',
            summary: thinking.text ? [{ type: 'summary_text', text: thinking.text }] : [],
            encrypted_content: thinking.signature
          }
        end

        def format_function_call_items(tool_calls)
          tool_calls.map do |_, tc|
            {
              type: 'function_call',
              call_id: tc.id,
              name: tc.name,
              arguments: JSON.generate(tc.arguments)
            }
          end
        end

        def format_output_content(msg)
          text = msg.content.is_a?(Content) ? msg.content.text : msg.content
          text = text.to_json if text.is_a?(Hash) || text.is_a?(Array)

          [{ type: 'output_text', text: text }]
        end

        def empty_content?(content)
          content.nil? || (content.is_a?(String) && content.strip.empty?) ||
            (content.is_a?(Content) && content.text.nil?)
        end

        def parse_output_text(output)
          texts = output.select { |item| item['type'] == 'message' }.flat_map do |message|
            Array(message['content']).filter_map { |part| part['text'] if part['type'] == 'output_text' }
          end

          texts.empty? ? nil : texts.join
        end

        def parse_function_calls(output)
          calls = output.select { |item| item['type'] == 'function_call' }
          return nil if calls.empty?

          calls.to_h do |call|
            arguments = call['arguments']

            [
              call['call_id'],
              ToolCall.new(
                id: call['call_id'],
                name: call['name'],
                arguments: arguments.nil? || arguments.empty? ? {} : JSON.parse(arguments)
              )
            ]
          end
        end

        def parse_reasoning_summary(output)
          texts = output.select { |item| item['type'] == 'reasoning' }.flat_map do |item|
            Array(item['summary']).filter_map { |part| part['text'] }
          end

          texts.empty? ? nil : texts.join("\n")
        end

        def parse_reasoning_signature(output)
          output.find { |item| item['type'] == 'reasoning' }&.dig('encrypted_content')
        end

        # Reasoning items with encrypted_content, exactly as received. Under store: false an
        # item without it cannot be replayed, so it is not kept.
        def parse_reasoning_items(output)
          output.select { |item| item['type'] == 'reasoning' && item['encrypted_content'] }
        end

        def supports_provider_file_references?
          true
        end

        def default_large_file_upload_threshold
          OPENAI_INLINE_FILE_LIMIT
        end

        def provider_file_upload_limit
          OPENAI_FILE_UPLOAD_LIMIT
        end

        def provider_file_attachable?(attachment)
          attachment.pdf?
        end

        def provider_file_upload_options(_attachment)
          { purpose: 'user_data' }
        end
      end
    end
  end
end
