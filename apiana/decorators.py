"""
Retry and idempotent decorators.
"""

import asyncio
import hashlib
import json
import logging
from functools import wraps
from typing import Callable, Optional

from apiana.storage.task_cache import Neo4jTaskCache

logger = logging.getLogger(__name__)

# Global cache instance (will be initialized when decorators are used)
_cache: Optional[Neo4jTaskCache] = None


def init_cache(task_cache: Neo4jTaskCache):
    """Initialize the global cache."""
    global _cache
    _cache = task_cache


def retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,)):
    """
    Retry decorator with exponential backoff.
    
    Args:
        attempts: Number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exception types to retry on
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    if attempt < attempts - 1:
                        sleep_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {sleep_time:.1f}s"
                        )
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(f"All {attempts} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator


def idempotent(key_generator: Callable[..., str]):
    """
    Idempotent decorator that caches results in Neo4j.
    
    Args:
        key_generator: Function to generate cache key from function arguments
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if _cache is None:
                raise RuntimeError("Task cache not initialized. Call init_cache() first.")
            
            # Generate cache key
            cache_key = key_generator(*args, **kwargs)
            
            # Check cache first
            cached_result = _cache.get_result(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached result for {func.__name__} key: {cache_key}")
                return cached_result
            
            # Execute function (either first time or retrying after failure)
            try:
                logger.info(f"Executing {func.__name__} with key: {cache_key}")
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache successful result
                _cache.set_result(cache_key, result, 'completed')
                logger.info(f"Cached successful result for {func.__name__} key: {cache_key}")
                return result
            
            except Exception as e:
                # Cache failure (but don't throw on retrieval)
                _cache.mark_failed(cache_key, str(e))
                logger.error(f"Cached failure for {func.__name__} key: {cache_key}: {e}")
                raise
        
        return wrapper
    return decorator


def generate_key_from_args(*args, **kwargs) -> str:
    """Generate a deterministic key from function arguments."""
    key_data = json.dumps([args, kwargs], sort_keys=True, default=str)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def conversation_summary_key(conversation, config, prompts) -> str:
    """Generate key for conversation summary operation."""
    content_hash = hashlib.md5(str(conversation.id).encode()).hexdigest()[:8]
    config_hash = hashlib.md5(str(config.run_id).encode()).hexdigest()[:8]
    prompt_hash = hashlib.md5(str(prompts).encode()).hexdigest()[:8]
    return f"summary_{content_hash}_{config_hash}_{prompt_hash}"


def summary_enrichment_key(summary_file_path, config) -> str:
    """Generate key for summary enrichment operation."""
    file_hash = hashlib.md5(str(summary_file_path).encode()).hexdigest()[:8]
    config_hash = hashlib.md5(str(config.run_id).encode()).hexdigest()[:8]
    return f"enrichment_{file_hash}_{config_hash}"