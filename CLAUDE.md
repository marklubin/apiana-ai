# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ChatGPT Export Processor - a Python application for processing and analyzing ChatGPT conversation exports. It provides a Terminal User Interface (TUI) for browsing conversations and preparing them for summarization.

## Common Development Commands

```bash
# Install dependencies (using uv package manager)
uv sync

# Run the TUI application
uv run processor-tui
# or
uv run python main.py

# Run with hot reload during development
uv run textual run --dev export_processor/chatgpt/tui.py

# Run with debugging enabled
uv run python main.py  # debugpy is already imported
```

## Code Architecture

### Core Components

1. **Data Models** (`export_processor/chatgpt/schema.py`):
   - `Message`: Individual message with role, content, timestamps
   - `OpenAIConversation`: Full conversation with title, messages, metadata
   - Handles ChatGPT's nested export format with content parts

2. **TUI Application** (`export_processor/chatgpt/tui.py`):
   - Built with Textual framework
   - State machine workflow: new → loading → loaded → prompt_selection → confirming
   - Components: FilePicker, DataTable, TextArea
   - Handles file selection, validation, and prompt selection

### Key Implementation Details

- The app expects ChatGPT JSON exports in the `data/` directory
- Prompt files should be placed in the `prompts/` directory
- Output would go to the `output/` directory (not yet implemented)
- The summarization logic itself is not implemented - the app currently only sets up the UI workflow

### Dependencies

- Python 3.13+ required
- Core: `pydantic` (data validation), `textual` (TUI framework)
- Development: `ruff` (linting), `textual-dev`, `debugpy`

### Development Notes

- Always use `uv` for package management (not pip)
- The project uses dataclasses for data models
- Error handling is implemented for file parsing operations
- The TUI uses Textual's logging system for debugging