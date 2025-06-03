# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Apiana AI is a comprehensive system for building memory-enabled AI agents. This monorepo contains the reference implementation for creating AI systems with persistent, experiential memory that can remember, reflect, and evolve with users over time.

## Common Development Commands

```bash
# Install dependencies (using uv package manager)
uv sync

# Run the ChatGPT Export TUI application
uv run gpt-export-tui
# or
uv run python main.py

# Run the Gradio Web UI for ChatGPT Export processing
uv run python launch_ui.py
# Opens web interface at http://localhost:7860

# Run with hot reload during development
uv run textual run --dev apiana/applications/chatgpt_export/tui.py

# Run linting
uv run ruff check .

# Run type checking
uv run mypy .
```

## Code Architecture

### Module Structure

```
apiana/
├── applications/          # User-facing applications
│   └── chatgpt_export_tui/  # Terminal UI for processing ChatGPT exports
├── configuration.py       # System-wide configuration schemas
├── embeddings/           # Embedding generation (local models)
├── flows/                # Processing workflows (Prefect-based)
├── llm/                  # LLM interfaces (OpenAI-compatible)
└── storage/              # Storage backends (Neo4j, future: vector DBs)
```

### Core Components

1. **Configuration** (`apiana/configuration.py`):
   - `ProcessorConfig`: Main configuration container
   - `Neo4jConfig`: Database connection settings
   - `LLMProviderConfig`: LLM service configuration
   - `EmbedderConfig`: Embedding model settings

2. **Memory Processing** (`apiana/flows/chatgpt_export_processor.py`):
   - Prefect-based workflow orchestration
   - Batch processing with concurrent execution
   - Progress tracking and error handling
   - Memory storage in Neo4j with embeddings

3. **Storage Layer** (`apiana/storage/neo4j_store.py`):
   - Graph-based memory storage
   - Experiential memory nodes with relationships
   - Vector similarity search capabilities
   - ProcessorRun tracking for batch jobs

4. **LLM Integration** (`apiana/llm/openai_agent.py`):
   - OpenAI-compatible client (works with Ollama, OpenAI, etc.)
   - Structured output support
   - System prompts and temperature control

5. **Embeddings** (`apiana/embeddings/local_embedder.py`):
   - Local model support via Transformers
   - Privacy-preserving vector generation
   - Configurable model selection

### Key Implementation Details

- **Memory Types**: Experiential, Conceptual, Reference, Reflective, Task State
- **Data Models**: Using dataclasses (no Pydantic dependency)
- **Async Support**: Currently synchronous, async can be added as needed
- **Error Handling**: Comprehensive error handling with retries
- **Testing**: Use pytest for unit and integration tests

### Development Notes

- Always use `uv` for package management (not pip)
- Import modules unconditionally at the top of files
- Use dataclasses instead of Pydantic models
- Follow existing code patterns and conventions
- Neo4j required for storage (future: pluggable backends)
- The project is designed to be extensible and modular

### Future Considerations

- Vector database support (Chroma, Weaviate)
- Multi-agent memory sharing protocols
- Reflection scheduling and automated journaling
- Fine-tuning pipelines from memory corpus
- Web-based memory browser interface