# Parallel Inference Design

## Overview

This document outlines the design for adding parallel inference capabilities to the Apiana AI system, supporting both vLLM and transformers backends while maintaining a clean, provider-agnostic architecture.

## Goals

1. **Achieve <1 hour processing time** for 200MB of conversation data
2. **Support multiple inference backends** (vLLM, transformers, future providers)
3. **Maintain backward compatibility** with existing pipelines
4. **Maximize hardware utilization** (2x 18GB GPUs, 128GB RAM)
5. **Keep parallelism logic decoupled** from provider implementations

## Architecture

### Layer Separation

```
┌─────────────────────────────────────┐
│         Pipeline Layer              │
│  (Orchestration & Data Flow)        │
├─────────────────────────────────────┤
│     Parallel Transform Layer        │
│  (Parallelism & Batch Management)   │
├─────────────────────────────────────┤
│        Provider Interface           │
│    (invoke, batch_invoke)           │
├─────────────────────────────────────┤
│     Provider Implementations        │
│ (VLLMProvider, TransformersProvider)│
└─────────────────────────────────────┘
```

### Key Components

#### 1. Provider Interface Extension

```python
# apiana/core/providers/base.py
class LLMProvider(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Single inference - required"""
        pass
    
    def batch_invoke(self, prompts: List[str]) -> List[str]:
        """Batch inference - optional, defaults to sequential"""
        return [self.invoke(prompt) for prompt in prompts]
    
    @property
    def supports_batch(self) -> bool:
        """Check if provider has native batch support"""
        return type(self).batch_invoke != LLMProvider.batch_invoke
```

#### 2. Parallel Transform Component

```python
# apiana/core/components/transform/parallel_summarizer.py
class ParallelSummarizerTransform(SummarizerTransform):
    """
    Provider-agnostic parallel processing transform.
    Automatically adapts to provider capabilities.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        max_workers: int = 2,
        batch_size: int = 16,
        **kwargs
    ):
        super().__init__(provider=provider, **kwargs)
        self.max_workers = max_workers
        self.batch_size = batch_size
```

#### 3. Provider Implementations

##### VLLMProvider
- **Location**: `apiana/core/providers/vllm_provider.py`
- **Features**: Native batch processing, tensor parallelism, continuous batching
- **Configuration**: Model name, tensor parallel size, GPU memory utilization

##### Enhanced LocalTransformersLLM
- **Location**: `apiana/core/providers/local.py` (existing, enhanced)
- **Features**: Optional batch support, multi-GPU via DataParallel
- **Configuration**: Device placement, quantization options

## Implementation Plan

### Phase 1: Provider Interface & Base Implementation
1. Extend `LLMProvider` base class with `batch_invoke` method
2. Add `supports_batch` property
3. Update existing `LocalTransformersLLM` to implement new interface

### Phase 2: VLLMProvider Implementation
1. Create new `VLLMProvider` class
2. Implement vLLM initialization with tensor parallelism
3. Add configuration for memory management and batching

### Phase 3: Parallel Transform Component
1. Create `ParallelSummarizerTransform` extending `SummarizerTransform`
2. Implement adaptive parallel processing logic
3. Add progress tracking and error handling

### Phase 4: Pipeline Integration
1. Create factory functions for parallel pipelines
2. Add CLI options for backend selection
3. Update existing pipelines to use parallel transforms where appropriate

## Configuration

### Example Pipeline Configuration

```python
# pipelines.py
def chatgpt_parallel_processing_pipeline(
    backend: str = "vllm",
    max_workers: int = 2,
    batch_size: int = 16,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    **kwargs
) -> Pipeline:
    """
    High-throughput parallel processing pipeline.
    
    Args:
        backend: "vllm" or "transformers"
        max_workers: Number of parallel workers
        batch_size: Batch size for inference
        model_name: Model to use for summarization
    """
    
    # Provider selection
    if backend == "vllm":
        provider = VLLMProvider(
            model_name=model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            **kwargs
        )
    else:
        provider = LocalTransformersLLM(
            model_name=model_name,
            device="auto",
            load_in_8bit=True,
            **kwargs
        )
    
    return Pipeline(
        name="chatgpt_parallel_processing",
        components=[
            ChatGPTExportReader(),
            ConversationChunkerComponent(
                chunk_method="tokens",
                chunk_size=3500
            ),
            ParallelSummarizerTransform(
                provider=provider,
                max_workers=max_workers,
                batch_size=batch_size,
                system_prompt=load_prompt("self-reflective-system-message.txt"),
                user_template=load_prompt("self-reflective-prompt-template.txt"),
                max_length=300
            ),
            ChatFragmentWriter(store=get_application_store())
        ]
    )
```

### CLI Integration

```bash
# Using vLLM backend (fast)
uv run chatgpt-export-v2 -i export.json -o output/ \
    --backend vllm \
    --batch-size 16 \
    --max-workers 2

# Using transformers backend (compatible)
uv run chatgpt-export-v2 -i export.json -o output/ \
    --backend transformers \
    --max-workers 4
```

## Performance Targets

### With vLLM Backend
- **Throughput**: ~100-300 tokens/sec
- **Batch size**: 16-24
- **Processing time**: 35-45 minutes for 200MB

### With Transformers Backend
- **Throughput**: ~40-60 tokens/sec  
- **Batch size**: 1-8
- **Processing time**: 60-70 minutes for 200MB

## Memory Management

### GPU Memory Allocation
```python
# vLLM: Automatic with PagedAttention
gpu_memory_utilization = 0.85  # 15% buffer

# Transformers: Manual management
per_gpu_batch_size = estimate_batch_size(
    model_size_gb=14,
    available_vram_gb=18,
    sequence_length=3500
)
```

### System RAM Usage
- Model weights: ~14-28GB (depending on model)
- Conversation data: ~200MB (input)
- KV cache: Dynamic based on batch size
- Buffer: 20% of total RAM

## Error Handling

1. **OOM Protection**: Adaptive batch size reduction
2. **Provider Failures**: Automatic retry with exponential backoff
3. **Partial Progress**: Save completed batches periodically
4. **Graceful Degradation**: Fall back to smaller batch sizes

## Testing Strategy

### Unit Tests
- Test provider interface compliance
- Test parallel transform with mock providers
- Test adaptive batching logic

### Integration Tests
- Test vLLM provider with real models
- Test transformers provider with batching
- Test pipeline end-to-end with both backends

### Performance Tests
- Benchmark throughput with different batch sizes
- Measure memory usage patterns
- Validate <1 hour target for 200MB input

## Future Enhancements

1. **Dynamic Load Balancing**: Distribute work based on GPU utilization
2. **Streaming Support**: Process results as they complete
3. **Multi-Node Support**: Distribute across multiple machines
4. **Adaptive Batch Sizing**: Automatically tune based on latency/throughput
5. **Provider Plugins**: Easy addition of new inference backends

## Dependencies

### Required Packages
```toml
# pyproject.toml additions
[project.optional-dependencies]
vllm = [
    "vllm>=0.4.0",
    "ray>=2.9.0",  # Required by vLLM
]

parallel = [
    "vllm>=0.4.0",
    "ray>=2.9.0",
    "torch>=2.0.0",
]
```

### Installation
```bash
# Install with vLLM support
uv sync --extra parallel

# Install base only (transformers)
uv sync
```

## Migration Path

1. **Existing pipelines continue to work** without modification
2. **Opt-in to parallel processing** via new pipeline variants
3. **Gradual migration** as users test and validate results
4. **Feature flags** for A/B testing between implementations