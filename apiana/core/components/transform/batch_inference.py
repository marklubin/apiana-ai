"""Batch inference transform component for efficient LLM API usage."""

import logging
import time
from typing import TypeVar, Generic, Callable, Protocol, List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from apiana.core.components.common.base import Component, ComponentResult

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


class BatchStore(Protocol[T, R]):
    """Storage interface for batch processing results."""
    
    def get_by_hash(self, hash_key: str) -> Optional[R]:
        """Retrieve cached result by hash key."""
        ...
    
    def store_result(self, hash_key: str, input_item: T, result: R) -> None:
        """Store computation result with hash key."""
        ...
    
    def batch_get_by_hashes(self, hash_keys: List[str]) -> Dict[str, Optional[R]]:
        """Batch retrieve multiple results by hash keys."""
        ...


class BatchProvider(Protocol[T, R]):
    """Provider interface that supports batch processing."""
    
    def process_batch(self, items: List[T]) -> List[R]:
        """Process a batch of items and return results in order."""
        ...


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int
    hash_function: Callable[[T], str]
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None


class BatchInferenceTransform(Component, Generic[T, R]):
    """
    Generic batch inference transform that:
    1. Accepts collections of inputs
    2. Processes in batches of size N
    3. Handles retries for failed batches
    4. Returns complete results in order
    
    Note: Hash-based deduplication is planned but not yet implemented.
    """
    
    def __init__(
        self,
        provider: BatchProvider[T, R],
        store: BatchStore[T, R],
        config: BatchConfig,
        name: Optional[str] = None
    ):
        """Initialize batch inference transform.
        
        Args:
            provider: Provider that supports batch processing
            store: Storage backend for results
            config: Batch processing configuration
            name: Optional component name
        """
        self.provider = provider
        self.store = store
        self.batch_config = config
        # Convert BatchConfig to dict for base class
        config_dict = {
            "batch_size": config.batch_size,
            "max_retries": config.max_retries,
            "retry_delay": config.retry_delay,
            "timeout": config.timeout
        }
        super().__init__(name=name or "BatchInferenceTransform", config=config_dict)
    
    @property
    def input_types(self) -> List[type]:
        """Return supported input types."""
        return [list]  # List[T]
    
    @property
    def output_types(self) -> List[type]:
        """Return output types."""
        return [list]  # List[R]
    
    def process(self, inputs: List[T]) -> ComponentResult:
        """Process input items in batches.
        
        Args:
            inputs: List of items to process
            
        Returns:
            ComponentResult containing list of results in same order as inputs
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        if not inputs:
            return ComponentResult(
                data=[],
                execution_time_ms=0,
                timestamp=datetime.utcnow()
            )
        
        # Partition inputs into batches
        batches = self._partition_into_batches(inputs)
        results = []
        
        # Process each batch with retry logic
        for i, batch in enumerate(batches):
            try:
                batch_results = self._process_batch_with_retry(batch)
                results.extend(batch_results)
            except Exception as e:
                error_msg = f"Failed to process batch {i+1}/{len(batches)}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue with other batches or raise?
                raise  # For now, fail fast
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return ComponentResult(
            data=results,
            metadata={
                "total_items": len(inputs),
                "batch_count": len(batches),
                "batch_size": self.batch_config.batch_size
            },
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.utcnow()
        )
    
    def _partition_into_batches(self, items: List[T]) -> List[List[T]]:
        """Partition items into batches of configured size.
        
        Args:
            items: Items to partition
            
        Returns:
            List of batches
        """
        batch_size = self.batch_config.batch_size
        batches = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        logger.debug(f"Partitioned {len(items)} items into {len(batches)} batches")
        return batches
    
    def _process_batch_with_retry(self, batch: List[T]) -> List[R]:
        """Process a single batch with retry logic.
        
        Args:
            batch: Items to process
            
        Returns:
            Results for the batch
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_error = None
        
        for attempt in range(self.batch_config.max_retries):
            try:
                logger.debug(f"Processing batch of {len(batch)} items (attempt {attempt + 1})")
                results = self.provider.process_batch(batch)
                
                if len(results) != len(batch):
                    raise ValueError(
                        f"Provider returned {len(results)} results for {len(batch)} inputs"
                    )
                
                return results
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Batch processing failed (attempt {attempt + 1}/{self.batch_config.max_retries}): {e}"
                )
                
                if attempt < self.batch_config.max_retries - 1:
                    # Exponential backoff
                    delay = self.batch_config.retry_delay * (2 ** attempt)
                    logger.debug(f"Waiting {delay}s before retry")
                    time.sleep(delay)
        
        # All retries exhausted
        raise RuntimeError(
            f"Batch processing failed after {self.batch_config.max_retries} attempts: {last_error}"
        ) from last_error