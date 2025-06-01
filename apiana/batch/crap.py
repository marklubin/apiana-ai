# """
# Idempotent operations for processing individual conversations.
# Split into two operations: summary generation and enrichment/storage.
# """
#
# import asyncio
# import logging
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Any, List, Optional
# import json
#
# from apiana.decorators import (
#     retry,
#     idempotent,
#     conversation_summary_key,
#     summary_enrichment_key,
# )
# from apiana.applications.chatgpt_export.schema import OpenAIConversation
# from apiana.configuration import ChatGPTExportProcessorConfiguration
# from apiana.llm.openai_client import LLMClient
# from apiana.embeddings.local_embedder import Embedder
# from apiana.storxage.neo4j_store import Neo4jMemoryStore
# from apiana.lease_manager import LeaseManager
#
# logger = logging.getLogger(__name__)
#
# # Global instances that will be initialized once
# llm_client: Optional[LLMClient] = None
# embedder: Optional[Embedder] = None
# lease_manager = LeaseManager(["llm", "embedder"], max_concurrent=1)
#
#
# def init_models(config: ChatGPTExportProcessorConfiguration):
#     """Initialize global model instances."""
#     global llm_client, embedder
#
#     if llm_client is None:
#         llm_client = LLMClient(
#             model=config.llm_provider.model,
#             base_url=config.llm_provider.base_url,
#             api_key=config.llm_provider.api_key,
#             temperature=config.llm_provider.temperature,
#             max_tokens=config.llm_provider.max_tokens,
#         )
#
#     if embedder is None:
#         embedder = Embedder(model_name=config.embedder.model)
#
#
# @retry(attempts=3, delay=1.0, backoff=2.0)
# @idempotent(key_generator=conversation_summary_key)
# async def generate_conversation_summary(
#     conversation: OpenAIConversation,
#     config: ChatGPTExportProcessorConfiguration,
#     prompts: Dict[str, str],
# ) -> Dict[str, Any]:
#     """
#     Generate summary for a single conversation and save to disk.
#     This is the first idempotent operation.
#
#     Returns:
#         Dict containing summary metadata and file path
#     """
#     logger.info(f"Generating summary for conversation: {conversation.title}")
#
#     # Ensure models are initialized
#     init_models(config)
#
#     # Use lease manager to ensure only one model type at a time
#     with lease_manager.lease("llm"):
#         # Set system prompt if needed
#         llm_client.system_prompt = prompts["system_prompt"]
#
#         # Format conversation messages as JSON structure
#         formatted_messages = []
#         for msg in conversation.messages:
#             formatted_messages.append({"role": msg.role, "content": msg.content})
#
#         # Convert to JSON string for the prompt
#         messages_json = json.dumps(formatted_messages, indent=2)
#
#         # Format the prompt with conversation data
#         formatted_prompt = prompts["user_prompt_template"].format(
#             conversation_date=conversation.created_at.strftime("%B %d, %Y")
#             if conversation.created_at
#             else "unknown date",
#             formatted_messages=messages_json,
#         )
#
#         # Generate summary using async LLM
#         summary_content = await llm_client.generate(formatted_prompt)
#
#     # Create output directory
#     summaries_dir = config.get_summaries_dir()
#     summaries_dir.mkdir(parents=True, exist_ok=True)
#
#     # Generate filename
#     created_date = (
#         conversation.created_at.strftime("%Y-%m-%d")
#         if conversation.created_at
#         else "unknown"
#     )
#     safe_title = "".join(
#         c for c in conversation.title if c.isalnum() or c in (" ", "-", "_")
#     )[:50].strip()
#     filename = f"{created_date}_{safe_title}.txt"
#     summary_file_path = summaries_dir / filename
#
#     # Write summary to disk asynchronously
#     async with aiofiles.open(summary_file_path, "w", encoding="utf-8") as f:
#         await f.write(summary_content)
#
#     # Return summary metadata
#     result = {
#         "conversation_id": conversation.id,
#         "title": conversation.title,
#         "summary_content": summary_content,
#         "message_count": len(conversation.messages),
#         "created_at": conversation.created_at.isoformat()
#         if conversation.created_at
#         else None,
#         "summary_timestamp": datetime.now().isoformat(),
#         "summary_file_path": str(summary_file_path),
#         "filename": filename,
#     }
#
#     logger.info(f"Summary generated and saved to: {summary_file_path}")
#     return result
#
#
# @retry(attempts=3, delay=1.0, backoff=2.0)
# @idempotent(key_generator=summary_enrichment_key)
# async def enrich_and_store_summary(
#     summary_content: str,
#     summary_file_path: str,
#     config: ChatGPTExportProcessorConfiguration,
# ) -> Dict[str, Any]:
#     """
#     Generate embeddings for summary and store in Neo4j.
#     This is the second idempotent operation.
#
#     Args:
#         summary_content: The summary text to enrich
#         summary_file_path: Path to the summary file (for metadata)
#         config: Processing configuration
#
#     Returns:
#         Dict containing enrichment metadata
#     """
#     logger.info(f"Enriching and storing summary from: {summary_file_path}")
#
#     # Ensure models are initialized
#     init_models(config)
#
#     # Use lease manager to ensure only one model type at a time
#     with lease_manager.lease("embedder"):
#         # Generate embedding asynchronously
#         logger.info("Generating embedding for summary")
#         embedding = await embedder.embed_async(summary_content)
#
#     # Initialize Neo4j store
#     neo4j_store = Neo4jMemoryStore(
#         uri=config.neo4j.uri,
#         auth=config.neo4j.auth,
#         embedding_model=config.embedder.model,
#         embedding_dimension=config.embedder.dimension,
#     )
#
#     # Extract conversation ID from filename for storage
#     filename = Path(summary_file_path).name
#     conversation_id = f"file_{filename}"  # Simplified for now
#
#     # TODO: Extract contextual tags using LLM
#     contextual_tags = {
#         "emotional_context": None,
#         "environmental_context": None,
#         "activity_context": None,
#         "social_context": None,
#     }
#
#     # Store in Neo4j
#     logger.info("Storing enriched summary in Neo4j")
#     # TODO: Implement actual storage method in Neo4jMemoryStore
#     # memory_id = neo4j_store.create_experiential_summary(
#     #     conversation_id=conversation_id,
#     #     summary_content=summary_content,
#     #     embedding=embedding,
#     #     contextual_tags=contextual_tags,
#     #     file_path=summary_file_path
#     # )
#
#     # For now, just return the enrichment metadata
#     result = {
#         "conversation_id": conversation_id,
#         "summary_file_path": summary_file_path,
#         "embedding_dimension": len(embedding),
#         "contextual_tags": contextual_tags,
#         "enrichment_timestamp": datetime.now().isoformat(),
#         "neo4j_stored": False,  # Will be True when storage is implemented
#     }
#
#     logger.info(f"Summary enriched and stored for: {conversation_id}")
#     return result
#
#
# async def process_single_conversation(
#     conversation: OpenAIConversation,
#     config: ChatGPTExportProcessorConfiguration,
#     prompts: Dict[str, str],
# ) -> Dict[str, Any]:
#     """
#     Process a single conversation through both idempotent operations.
#
#     Returns:
#         Combined results from both operations
#     """
#     try:
#         # Operation 1: Generate summary and save to disk
#         summary_result = await generate_conversation_summary(
#             conversation, config, prompts
#         )
#
#         # Operation 2: Enrich summary and store in Neo4j (pass summary content)
#         enrichment_result = await enrich_and_store_summary(
#             summary_result["summary_content"],
#             summary_result["summary_file_path"],
#             config,
#         )
#
#         # Combine results
#         return {
#             "status": "success",
#             "conversation_id": conversation.id,
#             "summary": summary_result,
#             "enrichment": enrichment_result,
#         }
#
#     except Exception as e:
#         logger.error(f"Failed to process conversation {conversation.id}: {e}")
#         return {"status": "failed", "conversation_id": conversation.id, "error": str(e)}
#
#
# async def process_conversations_batch(
#     conversations: List[OpenAIConversation],
#     config: ChatGPTExportProcessorConfiguration,
#     prompts: Dict[str, str],
#     progress_callback=None,
# ) -> List[Dict[str, Any]]:
#     """
#     Process multiple conversations concurrently using TaskGroup.
#
#     Args:
#         conversations: List of conversations to process
#         config: Processing configuration
#         prompts: Prompt templates
#         progress_callback: Optional progress callback
#
#     Returns:
#         List of processing results
#     """
#     results = []
#
#     # Use TaskGroup for concurrent processing
#     async with asyncio.TaskGroup() as tg:
#         conversations.map(
#             lambda convo: tg.create_task(
#                 process_single_conversation(convo, config, prompts)
#             )
#         )
#
#         tasks = []
#         for i, conversation in enumerate(conversations):
#             # Create task for each conversation
#             task = tg.create_task(
#                 process_single_conversation(conversation, config, prompts)
#             )
#             tasks.append((i, conversation.id, task))
#
#     # Collect results in order
#     for i, conv_id, task in tasks:
#         try:
#             result = task.result()
#             results.append(result)
#
#             if progress_callback:
#                 progress_callback(
#                     completed=i + 1,
#                     total=len(conversations),
#                     current_item=conv_id,
#                     status=result["status"],
#                 )
#         except Exception as e:
#             logger.error(f"Task failed for conversation {conv_id}: {e}")
#             results.append(
#                 {"status": "failed", "conversation_id": conv_id, "error": str(e)}
#             )
#
#     return results
