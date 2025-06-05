"""
Batched versions of local providers for efficient batch processing.
"""

import logging
from typing import List, Dict
import torch

from apiana.core.providers.base import LLMResponse
from apiana.core.providers.local import LocalTransformersLLM
from apiana.core.providers.local_embedding import LocalTransformersEmbedding
from apiana.core.components.transform.batch_inference import BatchProvider

logger = logging.getLogger(__name__)


class BatchedLocalTransformersLLM(BatchProvider[str, LLMResponse]):
    """
    Batched version of LocalTransformersLLM for efficient batch inference.
    
    This provider processes multiple prompts in a batch, improving throughput
    for local transformer models.
    """
    
    def __init__(self, llm_provider: LocalTransformersLLM):
        """Initialize with an existing LocalTransformersLLM instance.
        
        Args:
            llm_provider: The underlying LLM provider to use
        """
        self.llm_provider = llm_provider
        self.model = llm_provider.model
        self.tokenizer = llm_provider.tokenizer
        self.generation_config = llm_provider.generation_config
        self.max_length = llm_provider.max_length
    
    def process_batch(self, items: List[str]) -> List[LLMResponse]:
        """Process a batch of prompts and return responses.
        
        Args:
            items: List of prompt strings
            
        Returns:
            List of LLMResponse objects in the same order as inputs
        """
        if not items:
            return []
        
        # For single item, use regular invoke
        if len(items) == 1:
            return [self.llm_provider.invoke(items[0])]
        
        logger.debug(f"Processing batch of {len(items)} prompts")
        
        # Tokenize all prompts
        batch_inputs = self.tokenizer(
            items,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length - 500  # Leave room for response
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        
        # Generate responses for the batch
        with torch.no_grad():
            outputs = self.model.generate(
                **batch_inputs,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode responses
        responses = []
        for i, (input_ids, output_ids) in enumerate(zip(batch_inputs["input_ids"], outputs)):
            # Get the length of the input
            input_length = len(input_ids)
            
            # Extract only the generated tokens
            response_tokens = output_ids[input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Create LLMResponse
            response = LLMResponse(
                content=response_text,
                model=self.llm_provider.model_name,
                usage={
                    "prompt_tokens": input_length,
                    "completion_tokens": len(response_tokens),
                    "total_tokens": input_length + len(response_tokens),
                },
                metadata={
                    "batch_index": i,
                    "batch_size": len(items),
                    "device": str(device),
                }
            )
            responses.append(response)
        
        return responses


class BatchedLocalTransformersEmbedding(BatchProvider[str, List[float]]):
    """
    Batched version of LocalTransformersEmbedding for efficient batch embedding.
    
    This provider is already optimized for batch processing internally,
    so this is mainly a wrapper to conform to the BatchProvider interface.
    """
    
    def __init__(self, embedding_provider: LocalTransformersEmbedding):
        """Initialize with an existing LocalTransformersEmbedding instance.
        
        Args:
            embedding_provider: The underlying embedding provider to use
        """
        self.embedding_provider = embedding_provider
    
    def process_batch(self, items: List[str]) -> List[List[float]]:
        """Process a batch of texts and return embeddings.
        
        Args:
            items: List of text strings to embed
            
        Returns:
            List of embedding vectors in the same order as inputs
        """
        if not items:
            return []
        
        # The embedding provider already has efficient batch processing
        return self.embedding_provider.embed_documents(items)


class BatchedLLMWithSystemPrompt(BatchProvider[Dict[str, str], LLMResponse]):
    """
    Batched LLM provider that handles prompts with system instructions.
    
    Expects input items as dictionaries with 'prompt' and optional 'system_instruction'.
    """
    
    def __init__(self, llm_provider: LocalTransformersLLM):
        """Initialize with an existing LocalTransformersLLM instance.
        
        Args:
            llm_provider: The underlying LLM provider to use
        """
        self.llm_provider = llm_provider
        self.batched_provider = BatchedLocalTransformersLLM(llm_provider)
    
    def process_batch(self, items: List[Dict[str, str]]) -> List[LLMResponse]:
        """Process a batch of prompts with system instructions.
        
        Args:
            items: List of dicts with 'prompt' and optional 'system_instruction'
            
        Returns:
            List of LLMResponse objects in the same order as inputs
        """
        if not items:
            return []
        
        # Format prompts with system instructions
        formatted_prompts = []
        for item in items:
            prompt = item.get('prompt', '')
            system_instruction = item.get('system_instruction')
            
            if system_instruction:
                formatted_prompt = f"System: {system_instruction}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = prompt
            
            formatted_prompts.append(formatted_prompt)
        
        # Use the batched provider
        return self.batched_provider.process_batch(formatted_prompts)