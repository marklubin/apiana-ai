"""
Neo4j-based task result cache for idempotency.
"""

import json
import logging
from typing import Any, Optional

from apiana.storage.neo4j_store import Neo4jMemoryStore

logger = logging.getLogger(__name__)


class Neo4jTaskCache:
    """Neo4j-based task result cache for idempotency."""
    
    def __init__(self, neo4j_store: Neo4jMemoryStore):
        self.store = neo4j_store
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create task cache schema in Neo4j."""
        with self.store.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT task_cache_key IF NOT EXISTS
                FOR (t:TaskCache) REQUIRE t.key IS UNIQUE
            """)
    
    def get_result(self, key: str) -> Optional[Any]:
        """Get cached result by key. Returns None for both missing and failed results."""
        with self.store.driver.session() as session:
            result = session.run("""
                MATCH (t:TaskCache {key: $key})
                RETURN t.status, t.result, t.error, t.created_at
            """, key=key).single()
            
            if result:
                status, result_data, error, created_at = result
                if status == 'completed':
                    return json.loads(result_data) if result_data else None
                elif status == 'failed':
                    # Don't throw exception, just return None to allow retry
                    logger.info(f"Found cached failure for {key}, will retry: {error}")
                    return None
            
            return None
    
    def set_result(self, key: str, result: Any, status: str = 'completed', error: str = None):
        """Cache a result."""
        with self.store.driver.session() as session:
            session.run("""
                MERGE (t:TaskCache {key: $key})
                SET t.status = $status,
                    t.result = $result,
                    t.error = $error,
                    t.created_at = datetime(),
                    t.updated_at = datetime()
            """, 
                key=key,
                status=status,
                result=json.dumps(result, default=str) if result is not None else None,
                error=error
            )
    
    def mark_failed(self, key: str, error: str):
        """Mark a task as failed."""
        self.set_result(key, None, 'failed', error)
    
    def clear_cache(self, pattern: str = None):
        """Clear cache entries, optionally by pattern."""
        with self.store.driver.session() as session:
            if pattern:
                session.run("""
                    MATCH (t:TaskCache)
                    WHERE t.key CONTAINS $pattern
                    DELETE t
                """, pattern=pattern)
            else:
                session.run("MATCH (t:TaskCache) DELETE t")