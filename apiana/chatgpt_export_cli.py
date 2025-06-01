"""
Command-line interface for ChatGPT Export Processor.

Provides direct access to the export processor without the TUI.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from apiana.configuration import ProcessorConfig
from apiana.flows.chatgpt_export_processor import ChatGPTExportProcessor


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def process_export(
    export_path: Path,
    config: ProcessorConfig,
    verbose: bool = False
) -> None:
    """Process a ChatGPT export file."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing export: {export_path}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Output directory: {config.get_output_dir()}")
    
    processor = ChatGPTExportProcessor(config)
    
    try:
        # Run the processor
        stats = await processor.process_export(export_path)
        
        logger.info("\nProcessing complete!")
        logger.info(f"Total conversations: {stats['total_conversations']}")
        logger.info(f"Successful summaries: {stats['successful_summaries']}")
        logger.info(f"Failed summaries: {stats['failed_summaries']}")
        logger.info(f"Output directory: {config.get_output_dir()}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if verbose:
            logger.exception("Full error details:")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='ChatGPT Export Processor - Process ChatGPT exports and build memory graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Configuration:
  Set APIANA_ENVIRONMENT_STAGE to control which config to use:
    - local (default): Use local services (Ollama, local Neo4j)
    - dev: Use development services
    - production: Use production services (OpenAI, hosted Neo4j)

Examples:
  # Process with local configuration (default)
  chatgpt-export-process export.json

  # Process with development configuration
  APIANA_ENVIRONMENT_STAGE=dev chatgpt-export-process export.json

  # Process with production configuration
  APIANA_ENVIRONMENT_STAGE=production chatgpt-export-process export.json

  # Override output directory
  chatgpt-export-process export.json --output-dir ./my-output

  # Use a specific config file
  chatgpt-export-process export.json --config ./custom-config.toml
        """
    )
    
    parser.add_argument(
        'export_file',
        type=str,
        help='Path to the ChatGPT export JSON file'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to a custom configuration file (overrides environment)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Override the output directory'
    )
    
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        help='Override the batch size for processing'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without processing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate export file
    export_path = Path(args.export_file)
    if not export_path.exists():
        logger.error(f"Export file not found: {export_path}")
        sys.exit(1)
    
    # Load configuration
    try:
        if args.config:
            # Use custom config file
            config_path = Path(args.config)
            if config_path.suffix == '.toml':
                config = ProcessorConfig.from_toml(config_path)
            else:
                config = ProcessorConfig.from_file(config_path)
            logger.info(f"Loaded configuration from: {config_path}")
        else:
            # Use environment-based config
            config = ProcessorConfig.load_from_environment()
            logger.info(f"Loaded configuration for environment: {config.environment}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Apply command-line overrides
    if args.output_dir:
        config.output_base_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Show configuration if dry run
    if args.dry_run:
        logger.info("\nConfiguration (dry run):")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  LLM Model: {config.llm_provider.model}")
        logger.info(f"  LLM Base URL: {config.llm_provider.base_url}")
        logger.info(f"  Embedder: {config.embedder.model}")
        logger.info(f"  Neo4j URI: {config.neo4j.uri}")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  Output Directory: {config.get_output_dir()}")
        sys.exit(0)
    
    # Run the processor
    asyncio.run(process_export(export_path, config, args.verbose))


if __name__ == '__main__':
    main()