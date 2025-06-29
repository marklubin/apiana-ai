# Apiana AI

A comprehensive system for building memory-enabled AI agents with persistent, experiential memory. This monorepo contains the reference implementation for creating AI systems that can remember, reflect, and evolve with users over time.

## Overview

Apiana AI addresses the challenge of fragmented, stateless AI agents by providing a semantic memory layer that enables:
- **Persistent Identity**: Agents maintain continuity across sessions
- **Experiential Learning**: Systems that evolve through interaction
- **Reflective Capabilities**: Self-narrated insights and integration
- **User Data Sovereignty**: Memory ownership remains with the end user

## Project Structure

```
apiana/
├── applications/         # User-facing applications
│   ├── batch/           # Batch processing pipelines
│   ├── chatgpt_export/  # ChatGPT export CLI and TUI
│   └── gradio/          # Web UI for pipeline execution
├── core/                # Core components and pipelines
│   ├── components/      # Reusable pipeline components
│   ├── pipelines/       # Pipeline system implementation
│   └── providers/       # LLM and embedding providers
├── stores/              # Storage backends
│   └── neo4j/          # Neo4j graph storage
└── types/              # Data models and types
```

## Core Components

### Memory Types
- **Experiential**: Experiences, conversations, temporal sequences
- **Conceptual**: Ideas, beliefs, abstractions  
- **Reference**: Static documents or structured data
- **Reflective**: Self-narrated journal entries and insights
- **Task State**: Operational records and task threads

### Key Features
- **Component-Based Pipeline System**: Flexible, composable data processing
- **Dynamic Pipeline Discovery**: Automatic UI generation for pipelines
- **Graph-Based Storage**: Neo4j backend with vector index support
- **Agent Memory Isolation**: Each agent's memories are automatically tagged
- **Local LLM Support**: Privacy-preserving processing with local models
- **Comprehensive Testing**: Unit, integration, and UI automation tests

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/apiana-ai.git
cd apiana-ai

# Install with uv (recommended)
uv sync

# Start Neo4j database
docker-compose up -d
```

## Quick Start

### 1. ChatGPT Export Processing (CLI)

Process your ChatGPT conversation history into experiential memories:

```bash
# Run the new modular CLI (recommended)
uv run chatgpt-export-v2 -i export.json -o output/

# Or run the Terminal UI
uv run chatgpt-export-tui
```

### 2. Web UI (Gradio)

Launch the web interface for visual pipeline execution:

```bash
# Start the Gradio web UI
uv run python launch_gradio.py

# Open http://localhost:7860
```

### 3. Python API

Use the pipeline system programmatically:

```python
from pipelines import chatgpt_full_processing_pipeline
from apiana.types.configuration import Neo4jConfig
from pathlib import Path

# Configure Neo4j connection
neo4j_config = Neo4jConfig(
    username="neo4j",
    password="password",
    host="localhost",
    port=7687
)

# Create and run pipeline
pipeline = chatgpt_full_processing_pipeline(
    neo4j_config=neo4j_config,
    agent_id="my_agent",
    input_file=Path("export.json"),
    run_name="My Processing Run"
)

result = pipeline.run(Path("export.json"))
```

## Testing

```bash
# Run unit tests only (default)
make test

# Run integration tests
make test-integ

# Run UI automation tests
make test-ui

# Run comprehensive test suite with environment checks
make test-comprehensive

# Run linting
uv run ruff check --fix
```

## Architecture

### Component-Based Pipeline System

The system uses a flexible pipeline architecture where components can be composed:

```python
from apiana.core.pipelines.base import PipelineBuilder
from apiana.core.components import ChatGPTExportReader, ValidationTransform

pipeline = (
    PipelineBuilder("My Pipeline")
    .add_component(ChatGPTExportReader())
    .add_component(ValidationTransform(config={"min_messages": 2}))
    .add_component(ChatFragmentWriter(store=app_store))
    .build()
)
```

### Storage Architecture

- **ApplicationStore**: Shared storage for ChatFragments and pipeline metadata
- **AgentMemoryStore**: Agent-specific memory storage with automatic agent_id tagging
  - Works with Neo4j Community Edition (single database)
  - Each memory is tagged with agent_id for filtering
  - Vector indices supported for similarity search

### Pipeline Types

1. **Full Processing Pipeline**: Complete ChatGPT export processing with memory generation
2. **Fragment Storage Pipeline**: Simple storage without memory generation  
3. **Fragment to Memory Pipeline**: Post-process stored fragments into memories
4. **Custom Pipelines**: Define your own in `pipelines.py`

## Configuration

### Neo4j Setup

The system works with Neo4j Community Edition:
- Single database ("neo4j")
- Vector index support for embeddings
- Agent memories separated by agent_id property

### Environment Variables

Create a `.env` file:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## Development

### Prerequisites
- Python 3.11+
- Neo4j Database (Community Edition works)
- Docker (for Neo4j)
- uv package manager

### Adding New Pipelines

1. Define your pipeline factory in `pipelines.py`:
```python
def my_custom_pipeline(neo4j_config: Neo4jConfig, **kwargs) -> Pipeline:
    """My custom pipeline description."""
    return (
        PipelineBuilder("My Pipeline")
        .add_component(MyComponent())
        .build()
    )
```

2. Add metadata to `PIPELINE_REGISTRY` for UI discovery
3. The Gradio UI will automatically display your pipeline

### Development Commands

```bash
# Install dependencies
uv sync

# Run with hot reload (TUI)
uv run textual run --dev apiana/applications/chatgpt_export/tui.py

# Check code quality
uv run ruff check .
uv run mypy .
```

## Roadmap

- [x] Component-based pipeline system
- [x] Dynamic UI generation for pipelines
- [x] Neo4j Community Edition support
- [x] Comprehensive test coverage
- [ ] Additional vector database backends
- [ ] Reflection scheduling system
- [ ] Memory export/import tools
- [ ] Multi-agent memory protocols

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with inspiration from biological memory systems and the vision of creating AI that truly remembers and grows with its users.