# Configuration Guide

Apiana AI supports multiple configuration environments to handle different deployment scenarios.

## Environment-Based Configuration

The system uses the `APIANA_ENVIRONMENT_STAGE` environment variable to determine which configuration file to load.

### Available Environments

1. **local** (default)
   - Everything runs on the local machine
   - Uses local Ollama instance for LLM
   - Uses local Neo4j database
   - Configuration file: `configs/local.toml`

2. **dev**
   - Uses remote development services
   - Can connect to shared Ollama/Neo4j instances
   - Configuration file: `configs/dev.toml`

3. **production**
   - Uses 3rd party APIs (OpenAI)
   - Uses hosted services (Neo4j Aura)
   - Configuration file: `configs/production.toml`

### Setting the Environment

```bash
# Use local configuration (default)
chatgpt-export-process export.json

# Use development configuration
export APIANA_ENVIRONMENT_STAGE=dev
chatgpt-export-process export.json

# Use production configuration
export APIANA_ENVIRONMENT_STAGE=production
chatgpt-export-process export.json

# Or inline for a single command
APIANA_ENVIRONMENT_STAGE=production chatgpt-export-process export.json
```

## Configuration Files

Configuration files are stored in the `configs/` directory and use TOML format.

### Environment Variable Substitution

Configuration files support environment variable substitution using the `${VAR_NAME}` syntax:

```toml
[llm_provider]
api_key = "${OPENAI_API_KEY}"

[neo4j]
uri = "${NEO4J_URI}"
username = "${NEO4J_USERNAME}"
password = "${NEO4J_PASSWORD}"
```

### Configuration Structure

```toml
[general]
environment = "local"
batch_size = 10
output_base_dir = "output"

[prompt]
name = "default"
system_prompt_file = "prompts/self-reflective-system-message.txt"
user_prompt_template_file = "prompts/self-reflective-prompt-template.txt"

[llm_provider]
model = "llama3.2"
base_url = "http://localhost:11434/v1"
api_key = "ollama"
temperature = 0.7
max_tokens = 4096

[embedder]
type = "local"
model = "nomic-ai/nomic-embed-text-v1"
dimension = 768

[neo4j]
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"
```

## Command-Line Interface

The `chatgpt-export-process` command provides direct access to the export processor:

### Basic Usage

```bash
chatgpt-export-process export.json
```

### Options

- `-c, --config PATH`: Use a custom configuration file
- `-o, --output-dir DIR`: Override the output directory
- `-b, --batch-size N`: Override the batch size
- `-v, --verbose`: Enable verbose logging
- `--dry-run`: Show configuration without processing

### Examples

```bash
# Process with custom config
chatgpt-export-process export.json --config my-config.toml

# Override output directory
chatgpt-export-process export.json --output-dir ./results

# Dry run to check configuration
chatgpt-export-process export.json --dry-run

# Verbose mode for debugging
chatgpt-export-process export.json --verbose
```

## Programmatic Usage

You can also load configurations programmatically:

```python
from apiana.configuration import ChatGPTExportProcessorConfiguration

# Load from environment
config = ChatGPTExportProcessorConfiguration.load_from_environment()

# Load from specific TOML file
config = ChatGPTExportProcessorConfiguration.from_toml("configs/production.toml")

# Load from JSON file
config = ChatGPTExportProcessorConfiguration.from_file("saved-config.json")

# Save configuration
config.save_to_file("my-config.json")
```