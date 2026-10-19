# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Tool do
  describe '.description' do
    it 'allows desc as a shorter alias' do
      stub_const('DescribedTool', Class.new(described_class) do
        desc 'Does a thing'
      end)

      expect(DescribedTool.description).to eq('Does a thing')
      expect(DescribedTool.new.description).to eq('Does a thing')
    end
  end

  describe '.param' do
    it 'accepts description as an alias for desc' do
      stub_const('ParamDescriptionTool', Class.new(described_class) do
        param :city, description: 'City name'

        def execute(city:)
          city
        end
      end)

      expect(ParamDescriptionTool.parameters[:city].description).to eq('City name')
    end
  end

  describe '#name' do
    it 'converts class name to snake_case and removes _tool suffix' do
      stub_const('SampleTool', Class.new(described_class))
      expect(SampleTool.new.name).to eq('sample')
    end

    # rubocop:disable Naming/AsciiIdentifiers

    it 'normalizes class name Unicode characters to ASCII' do
      stub_const('SàmpleTòol', Class.new(described_class))
      expect(SàmpleTòol.new.name).to eq('sample')
    end

    it 'handles class names with unsupported characters' do
      stub_const('SampleΨTool', Class.new(described_class))
      expect(SampleΨTool.new.name).to eq('sample')
    end

    # rubocop:enable Naming/AsciiIdentifiers

    it 'handles class names without Tool suffix' do
      stub_const('AnotherSample', Class.new(described_class))
      expect(AnotherSample.new.name).to eq('another_sample')
    end

    it 'strips :: for class in module namespace' do
      stub_const('TestModule::SampleTool', Class.new(described_class))
      expect(TestModule::SampleTool.new.name).to eq('test_module--sample')
    end

    it 'handles ASCII-8BIT encoded class names without raising Encoding::CompatibilityError' do
      # This simulates a class name that is ASCII-8BIT encoded
      tool_class = Class.new(described_class)
      ascii_8bit_name = 'SampleTool'.dup.force_encoding('ASCII-8BIT')
      allow(tool_class).to receive(:name).and_return(ascii_8bit_name)
      expect(tool_class.new.name).to eq('sample')
    end
  end

  describe '#call' do
    it 'returns an error hash for unknown keyword arguments' do
      stub_const('SignatureTool', Class.new(described_class) do
        def execute(questions:)
          questions
        end
      end)

      result = SignatureTool.new.call({ 'questions' => [], 'isOther' => true })

      expect(result).to eq({ error: 'Invalid tool arguments: unknown keyword: isOther' })
    end

    it 'returns an error hash for missing required keyword arguments' do
      stub_const('RequiredTool', Class.new(described_class) do
        def execute(questions:)
          questions
        end
      end)

      result = RequiredTool.new.call({})

      expect(result).to eq({ error: 'Invalid tool arguments: missing keyword: questions' })
    end

    it 'allows extra keyword arguments when execute accepts keyrest' do
      stub_const('FlexibleTool', Class.new(described_class) do
        def execute(questions:, **extra)
          { questions:, extra: }
        end
      end)

      result = FlexibleTool.new.call({ 'questions' => [1], 'isOther' => true })

      expect(result).to eq({ questions: [1], extra: { isOther: true } })
    end

    it 're-raises unrelated ArgumentError from inside execute' do
      stub_const('ManualArgumentErrorTool', Class.new(described_class) do
        def execute(questions:)
          _ = questions
          raise ArgumentError, 'bad value provided'
        end
      end)

      expect { ManualArgumentErrorTool.new.call({ 'questions' => [] }) }
        .to raise_error(ArgumentError, 'bad value provided')
    end

    it 'returns an error hash for unknown arguments when execute takes no keywords' do
      stub_const('NoArgumentTool', Class.new(described_class) do
        def execute
          'ok'
        end
      end)

      result = NoArgumentTool.new.call({ 'unexpected' => true })

      expect(result).to eq({ error: 'Invalid tool arguments: unknown keyword: unexpected' })
    end

    it 'preserves legacy positional argument handling' do
      stub_const('RestArgumentTool', Class.new(described_class) do
        def execute(*args)
          args
        end
      end)

      result = RestArgumentTool.new.call({ 'legacy' => true })

      expect(result).to eq([{ legacy: true }])
    end
  end

  describe '#params_schema' do
    it 'infers flat string parameters from the execute keyword signature' do
      stub_const('InferredSignatureTool', Class.new(described_class) do
        def execute(query:, limit: '5')
          [query, limit]
        end
      end)

      expect(InferredSignatureTool.new.params_schema).to eq(
        'type' => 'object',
        'properties' => {
          'query' => { 'type' => 'string' },
          'limit' => { 'type' => 'string' }
        },
        'required' => ['query'],
        'additionalProperties' => false,
        'strict' => true
      )
    end

    it 'uses an empty object schema for tools without keyword arguments' do
      stub_const('EmptySignatureTool', Class.new(described_class) do
        def execute
          'ok'
        end
      end)

      expect(EmptySignatureTool.new.params_schema).to eq(
        'type' => 'object',
        'properties' => {},
        'required' => [],
        'additionalProperties' => false,
        'strict' => true
      )
    end

    it 'keeps explicit param declarations authoritative' do
      stub_const('ExplicitParamTool', Class.new(described_class) do
        param :query, desc: 'Search query'

        def execute(query:, internal:)
          [query, internal]
        end
      end)

      expect(ExplicitParamTool.new.params_schema).to eq(
        'type' => 'object',
        'properties' => {
          'query' => { 'type' => 'string', 'description' => 'Search query' }
        },
        'required' => ['query'],
        'additionalProperties' => false,
        'strict' => true
      )
    end
  end

  describe 'RubyLLM 2.0 tool DSL names' do
    def expected_schema
      {
        'type' => 'object',
        'properties' => {
          'city' => { 'type' => 'string', 'description' => 'City name' },
          'days' => { 'type' => 'integer', 'description' => 'Forecast days' },
          'tags' => { 'type' => 'array', 'description' => 'Tags', 'items' => { 'type' => 'string' } }
        },
        'required' => %w[city tags],
        'additionalProperties' => false,
        'strict' => true
      }
    end

    def fixed_schema
      { 'type' => 'object', 'properties' => { 'q' => { 'type' => 'string' } }, 'required' => ['q'] }
    end

    def gemini_fixed_schema
      { type: 'OBJECT', properties: { 'q' => { type: 'STRING' } }, required: ['q'] }
    end

    def base_tool
      Class.new(described_class) do
        parameter :base, description: 'Base value'

        def execute(base:)
          base
        end
      end
    end

    def merged_hash
      base_tool.declared_parameters.merge(extra: RubyLLM::Parameter.new(:extra, description: 'Extra value'))
    end

    def merged_schema
      {
        'type' => 'object',
        'properties' => {
          'base' => { 'type' => 'string', 'description' => 'Base value' },
          'extra' => { 'type' => 'string', 'description' => 'Extra value' }
        },
        'required' => %w[base extra],
        'additionalProperties' => false,
        'strict' => true
      }
    end

    def build_tool(const_name, &)
      klass = Class.new(described_class, &)
      stub_const(const_name, klass)
      klass.new
    end

    it 'builds the same schema from param/desc and parameter/description' do
      old_tool = build_tool('OldNamesTool') do
        param :city, desc: 'City name'
        param :days, type: :integer, desc: 'Forecast days', required: false
        param :tags, type: :array, desc: 'Tags'
      end
      new_tool = build_tool('NewNamesTool') do
        parameter :city, description: 'City name'
        parameter :days, type: :integer, description: 'Forecast days', required: false
        parameter :tags, type: :array, description: 'Tags'
      end

      [old_tool, new_tool].each do |tool|
        expect(tool.params_schema).to eq(expected_schema)
        expect(tool.parameters_schema).to eq(expected_schema)
      end
    end

    it 'accepts params and parameters with a schema hash' do
      schema = { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] }
      old_tool = build_tool('OldHashTool') { params schema }
      new_tool = build_tool('NewHashTool') { parameters schema }

      expect(old_tool.parameters_schema).to eq(fixed_schema)
      expect(new_tool.parameters_schema).to eq(old_tool.parameters_schema)
    end

    it 'accepts params and parameters with a RubyLLM::Schema block' do
      old_tool = build_tool('OldBlockTool') do
        params do
          string :q
        end
      end
      new_tool = build_tool('NewBlockTool') do
        parameters do
          string :q
        end
      end

      expect(old_tool.parameters_schema).to eq(new_tool.parameters_schema)
      expect(new_tool.parameters_schema['properties']).to eq('q' => { 'type' => 'string' })
      expect(new_tool.params_schema).to eq(new_tool.parameters_schema)
    end

    it 'returns the parameter hash from parameters without arguments and the class from params' do
      klass = Class.new(described_class) { parameter :a }

      expect(klass.parameters).to be(klass.declared_parameters)
      expect(klass.parameters.keys).to eq([:a])
      expect(klass.params).to be(klass)
    end

    it 'honors a class-level parameters override through both names' do
      merged = merged_hash
      tool = build_tool('ClassParametersOverrideTool') do
        define_singleton_method(:parameters) { |*| merged }
      end

      expect(tool.params_schema).to eq(merged_schema)
      expect(tool.parameters_schema).to eq(merged_schema)
      expect(tool.parameters).to eq(merged)
      expect(tool.declared_parameters).to eq(merged)
    end

    it 'honors a class-level declared_parameters override through both names' do
      merged = merged_hash
      tool = build_tool('ClassDeclaredOverrideTool') do
        define_singleton_method(:declared_parameters) { merged }
      end

      expect(tool.params_schema).to eq(merged_schema)
      expect(tool.parameters_schema).to eq(merged_schema)
      expect(tool.parameters).to eq(merged)
      expect(tool.declared_parameters).to eq(merged)
    end

    it 'honors instance-level parameters and params_schema overrides at the 1.x names' do
      schema = fixed_schema
      tool = build_tool('InstanceOldOverrideTool') do
        define_method(:parameters) { {} }
        define_method(:params_schema) { schema }
      end

      # parameters_schema and declared_parameters sit further down the chains than these overrides
      # and so deliberately do not see them. The protocols call params_schema and parameters.
      expect(tool.params_schema).to eq(fixed_schema)
      expect(tool.parameters).to eq({})
    end

    it 'honors instance-level declared_parameters and parameters_schema overrides at the 2.0 names' do
      schema = fixed_schema
      tool = build_tool('InstanceNewOverrideTool') do
        define_method(:declared_parameters) { {} }
        define_method(:parameters_schema) { schema }
      end

      expect(tool.params_schema).to eq(fixed_schema)
      expect(tool.parameters_schema).to eq(fixed_schema)
      expect(tool.parameters).to eq({})
      expect(tool.declared_parameters).to eq({})
    end

    %w[InstanceOldOverrideTool InstanceNewOverrideTool].each do |const_name|
      context "with #{const_name}" do
        let(:tool) do
          schema = fixed_schema
          if const_name == 'InstanceOldOverrideTool'
            build_tool(const_name) do
              define_method(:parameters) { {} }
              define_method(:params_schema) { schema }
            end
          else
            build_tool(const_name) do
              define_method(:declared_parameters) { {} }
              define_method(:parameters_schema) { schema }
            end
          end
        end

        it 'reaches Anthropic' do
          definition = RubyLLM::Protocols::Anthropic::Tools.function_for(tool)
          expect(definition[:input_schema]).to eq(fixed_schema)
        end

        it 'reaches Chat Completions' do
          definition = RubyLLM::Protocols::ChatCompletions::Tools.tool_for(tool)
          expect(definition[:function][:parameters]).to eq(fixed_schema)
        end

        it 'reaches Converse' do
          definition = RubyLLM::Protocols::Converse::Chat.format_tool(tool)
          expect(definition[:toolSpec][:inputSchema][:json]).to eq(fixed_schema)
        end

        it 'reaches Gemini' do
          formatter = Object.new.extend(RubyLLM::Protocols::Gemini::Tools)
          definitions = formatter.format_tools({ tool.name.to_sym => tool })
          expect(definitions[0][:functionDeclarations][0][:parameters]).to eq(gemini_fixed_schema)
        end

        it 'reaches Responses' do
          definition = RubyLLM::Protocols::Responses::Tools.tool_for(tool)
          expect(definition[:parameters]).to eq(fixed_schema)
        end
      end
    end

    it 'does not copy declared parameters into subclasses' do
      parent = Class.new(described_class) { parameter :query }
      child = Class.new(parent)

      # RubyLLM 2.0 copies a parent's parameters and schema definition into each subclass.
      # 1.16 does not, and Supernova handles the difference at the gem bump.
      expect(child.declared_parameters).to eq({})
      expect(parent.declared_parameters.keys).to eq([:query])
    end
  end
end
