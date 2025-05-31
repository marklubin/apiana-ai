"""
OpenAI-compatible agent for LLM interactions.

Uses the OpenAI Python client to communicate with Ollama's OpenAI-compatible endpoint.
Supports structured outputs using dataclass models.
"""

import logging
from typing import Optional, List, Union, Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Simple LLM agent using OpenAI client with Ollama endpoint.
    
    Supports both text generation and structured outputs.
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        system_prompt: str = "",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0
    ):
        """
        Initialize LLM agent.
        
        Args:
            model: Ollama model name
            system_prompt: System prompt for the agent
            base_url: Ollama OpenAI-compatible endpoint
            api_key: API key (not required for Ollama)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(
        self,
        prompt: str,
        response_model: Optional[Any] = None,
        temperature: Optional[float] = None
    ) -> Union[Any, str]:
        """
        Generate response with optional structured output.
        
        Args:
            prompt: The prompt text
            response_model: Optional model class for structured output
            temperature: Override default temperature
            
        Returns:
            Instance of response_model if provided, otherwise string
        """
        messages: List[ChatCompletionMessageParam] = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        try:
            if response_model:
                # Use structured output
                response = await self.client.beta.chat.completions.parse(
                    **params,
                    response_format=response_model
                )
                return response.choices[0].message.parsed
            else:
                # Regular text output
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise