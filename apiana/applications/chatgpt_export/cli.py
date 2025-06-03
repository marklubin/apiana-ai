import argparse
import json
import logging
import os
from typing import List

from neo4j_graphrag.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings,
)
from neo4j_graphrag.llm.openai_llm import OpenAILLM

from apiana import runtime_config
from apiana.batch.chatgpt.chatgpt_export_parsing import convo_from_export_format_dict
from apiana.storage.neo4j_store import Neo4jMemoryStore
from apiana.types.common import Conversation

log = logging.getLogger(__name__)


def get_dependencies():
    """Initialize and return dependencies. This allows for easier testing."""
    memory_store = Neo4jMemoryStore(runtime_config.neo4j)
    summarizer = OpenAILLM(
        runtime_config.summarizer.model_name,
        {
            "temperature": runtime_config.summarizer.temperature,
            "max_tokens": runtime_config.summarizer.max_tokens,
        },
        base_url=runtime_config.summarizer.inference_provider_config.base_url,
    )
    embedder = SentenceTransformerEmbeddings(
        runtime_config.embedding_model_name, trust_remote_code=True
    )
    return memory_store, summarizer, embedder


def write_convos_in_apiana_format(convos: List[Conversation], output_dir: str):
    parsed_dir = os.path.join(output_dir, "parsed")
    os.makedirs(parsed_dir, exist_ok=True)
    for i, c in enumerate(convos):
        # Sanitize filename - replace problematic characters
        safe_title = c.title.replace(" ", "_").replace("/", "_").replace(":", "_")
        safe_title = "".join(
            char for char in safe_title if char.isalnum() or char in ("_", "-", ".")
        )
        output_file_name = f"{i}_{safe_title.lower()}.json"
        output_path = os.path.join(parsed_dir, output_file_name)
        output = c.to_json(indent=2)
        with open(output_path, "w") as f:
            f.write(output)
            log.info(f"Wrote {output_path} - {i + 1}/{len(convos)}")


def process_one_conversation(
    convo: Conversation,
    output_dir: str,
    memory_store=None,
    summarizer=None,
    embedder=None,
):
    """Process a single conversation. Dependencies can be injected for testing."""
    if memory_store is None or summarizer is None or embedder is None:
        memory_store, summarizer, embedder = get_dependencies()

    log.info(f"Embedding convo for {convo}, calling llm.")
    prompt = f"{runtime_config.summarizer.prompt_config.userprompt_template}\n\n{convo.to_json(indent=2)}"
    system = runtime_config.summarizer.prompt_config.system_prompt
    summary = summarizer.invoke(prompt, system_instruction=system).content

    # Sanitize filename - same logic as write_convos_in_apiana_format
    safe_title = convo.title.replace(" ", "_").replace("/", "_").replace(":", "_")
    safe_title = "".join(
        char for char in safe_title if char.isalnum() or char in ("_", "-", ".")
    )
    output_file_name = f"{safe_title.lower()}.txt"
    output_path = os.path.join(output_dir, output_file_name)

    with open(output_path, "w") as f:
        f.write(summary)
        log.info(f"Wrote {output_path}")

    log.info("Calling embedder.")
    vector = embedder.embed_query(summary)
    log.info("Received emedding vector, persisting to graph")
    memory_store.store_convo(convo, summary, vector, [])


def embedded_chatgpt_export_summaries(
    input_file: str, output_dir: str, memory_store=None, summarizer=None, embedder=None
):
    """Main processing function. Dependencies can be injected for testing."""
    log.info("Starting...")
    apiana_convos = []
    with open(input_file) as f:
        data = json.load(f)
    log.info("Read json file, starting to convert to Apiana format.")

    for c in data:
        apiana_convo = convo_from_export_format_dict(c)
        apiana_convos.append(apiana_convo)

    log.info("Writing out Apiana convos.")
    write_convos_in_apiana_format(apiana_convos, output_dir)
    log.info("Wrote convos.")

    summaries_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)

    for i, c in enumerate(apiana_convos):
        process_one_conversation(c, summaries_dir, memory_store, summarizer, embedder)

    log.info("Extraction finished!")


def main():
    parser = argparse.ArgumentParser(
        description="Process ChatGPT export files and generate embeddings"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input file path (ChatGPT export JSON)"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for parsed conversations and summaries",
    )

    args = parser.parse_args()

    # Invoke the function with parsed arguments
    embedded_chatgpt_export_summaries(args.input, args.output)


if __name__ == "__main__":
    main()
