# Apiana AI Development Plan

## Current State Assessment

### Critical Issues Found

1. **Circular Import Dependencies**
   - `chatgpt_export_loader.py` and `chatgpt_export_processor.py` have circular imports
   - Tests cannot run due to these import errors

2. **Type System Inconsistency**
   - Mixed use of dataclasses and Pydantic models for the same entities
   - `AIConversation` exists in both `types/common.py` (Pydantic) and `types/chatgpt_conversation.py` (dataclass as `ChatGPTConversation`)
   - Import errors due to naming mismatches

3. **Syntax and Implementation Errors**
   - `summary_generator.py` has incorrect method signature: `generateOnce(self, AIConversation: convo)`
   - Missing imports (json) in several files
   - Undefined variables in parsing functions

4. **Incomplete Implementations**
   - Local embedder has no actual embedding logic
   - Summary generator async method references non-existent variables
   - Several TODO items remain unimplemented

## Comprehensive Development Plan

### Phase 1: Fix Critical Issues (Immediate)

#### 1.1 Resolve Circular Imports
- [ ] Remove import of `chatgpt_export_processor` from `chatgpt_export_loader.py`
- [ ] Move shared functionality to a separate module if needed
- [ ] Ensure clean dependency hierarchy

#### 1.2 Standardize Type System
- [ ] Complete migration to Pydantic for all data models
- [ ] Remove duplicate model definitions
- [ ] Update all imports to use consistent naming:
  - Use `AIConversation` from `types/common.py`
  - Remove `ChatGPTConversation` from `types/chatgpt_conversation.py`
  - Update all references throughout the codebase

#### 1.3 Fix Syntax Errors
- [ ] Fix `summary_generator.py` method signatures
- [ ] Add missing imports
- [ ] Fix undefined variables in parsing functions

### Phase 2: Complete Core Functionality

#### 2.1 Implement Embedding System
- [ ] Complete `local_embedder.py` implementation
- [ ] Add support for multiple embedding providers:
  - [ ] Local (Sentence Transformers)
  - [ ] OpenAI
  - [ ] Cohere
  - [ ] Custom providers via plugin system
- [ ] Create embedding provider interface/protocol
- [ ] Add embedding caching mechanism

#### 2.2 Complete LLM Provider System
- [ ] Enhance `openai_client.py` to support multiple providers:
  - [ ] OpenAI
  - [ ] Anthropic
  - [ ] Ollama (local)
  - [ ] Azure OpenAI
  - [ ] Custom providers
- [ ] Create provider configuration system
- [ ] Add retry logic and fallback mechanisms

#### 2.3 Finish Batch Processing
- [ ] Complete async batch summary generation
- [ ] Implement proper error handling and recovery
- [ ] Add progress tracking and reporting
- [ ] Create batch job management system

### Phase 3: Configuration and Deployment

#### 3.1 Complete Pydantic Migration
- [ ] Convert all configuration classes to Pydantic Settings
- [ ] Add environment variable support
- [ ] Create configuration validation
- [ ] Add configuration file formats (YAML, TOML)

#### 3.2 Storage Enhancements
- [ ] Complete Neo4j implementation
- [ ] Add vector database support:
  - [ ] ChromaDB
  - [ ] Weaviate
  - [ ] Pinecone
  - [ ] Qdrant
- [ ] Create storage adapter interface
- [ ] Implement storage migration tools

### Phase 4: Testing and Quality

#### 4.1 Comprehensive Test Suite
- [ ] Fix existing test imports
- [ ] Add unit tests for all modules
- [ ] Create integration tests
- [ ] Add end-to-end tests
- [ ] Implement test fixtures and factories

#### 4.2 Documentation
- [ ] Complete API documentation
- [ ] Create user guides
- [ ] Add architecture diagrams
- [ ] Write deployment guides

### Phase 5: Advanced Features

#### 5.1 Memory Management
- [ ] Implement reflection scheduling
- [ ] Add memory versioning
- [ ] Create memory search and retrieval APIs
- [ ] Implement memory compression strategies

#### 5.2 Multi-Agent Support
- [ ] Design inter-agent communication protocol
- [ ] Implement shared memory pools
- [ ] Add agent discovery mechanisms
- [ ] Create agent coordination systems

## Implementation Order

1. **Week 1: Critical Fixes**
   - Fix all import errors and circular dependencies
   - Standardize on Pydantic models
   - Get tests running successfully

2. **Week 2: Core Functionality**
   - Implement embedding generation
   - Complete batch processing
   - Fix all syntax errors

3. **Week 3: Provider System**
   - Multiple LLM provider support
   - Multiple embedding provider support
   - Configuration management

4. **Week 4: Storage and Testing**
   - Complete storage implementations
   - Comprehensive test coverage
   - Documentation

5. **Week 5+: Advanced Features**
   - Memory management enhancements
   - Multi-agent support
   - Performance optimization

## Technical Decisions

### Architecture Principles
1. **Plugin-based Architecture**: Use interfaces/protocols for providers
2. **Dependency Injection**: Leverage existing DI system for flexibility
3. **Async-First**: Design APIs with async support from the start
4. **Type Safety**: Use Pydantic throughout for validation
5. **Testability**: Design with testing in mind

### Technology Stack
- **Type System**: Pydantic (remove dataclasses for models)
- **Async**: asyncio with proper task management
- **Testing**: pytest with async support
- **Documentation**: mkdocs with autodoc
- **Packaging**: uv for dependency management

## Migration Path

### From Current State to Target State

1. **Data Model Migration**
   ```python
   # Current (mixed)
   @dataclass
   class ChatGPTConversation: ...
   
   # Target (Pydantic everywhere)
   class AIConversation(BaseModel): ...
   ```

2. **Provider System**
   ```python
   # Current (hardcoded)
   llm_client = LLMClient(model="gpt-4", ...)
   
   # Target (configurable)
   llm_provider = LLMProviderFactory.create(config.llm_provider)
   ```

3. **Storage System**
   ```python
   # Current (Neo4j only)
   store = Neo4jStore(...)
   
   # Target (pluggable)
   store = StorageFactory.create(config.storage)
   ```

## Success Metrics

1. **Code Quality**
   - All tests passing
   - No linting errors
   - 80%+ test coverage

2. **Functionality**
   - Can process ChatGPT exports end-to-end
   - Supports at least 3 LLM providers
   - Supports at least 2 embedding providers
   - Supports at least 2 storage backends

3. **Performance**
   - Can process 1000 conversations in under 10 minutes
   - Embedding generation < 100ms per conversation
   - Storage operations < 50ms per conversation

## Next Steps

1. **Immediate Actions**
   - Fix circular imports in `chatgpt_export_loader.py`
   - Standardize model names to use `AIConversation`
   - Fix syntax errors in `summary_generator.py`
   - Get tests running

2. **Short-term Goals** (1-2 weeks)
   - Complete Pydantic migration
   - Implement basic embedding functionality
   - Fix batch processing

3. **Medium-term Goals** (3-4 weeks)
   - Multi-provider support
   - Comprehensive testing
   - Documentation

4. **Long-term Goals** (1-2 months)
   - Advanced memory features
   - Multi-agent support
   - Production deployment

This plan provides a clear path forward to transform Apiana AI from its current partially-implemented state to a fully-functional, production-ready system with support for multiple providers and deployment scenarios.