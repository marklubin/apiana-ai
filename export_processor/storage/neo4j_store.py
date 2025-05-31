"""
Neo4j storage implementation using neomodel ORM for experiential summaries.
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import os

from neomodel import (
    config, StructuredNode, StringProperty, DateTimeProperty,
    ArrayProperty, FloatProperty, IntegerProperty, JSONProperty, db
)
from neomodel.exceptions import UniqueProperty

# Configure neomodel connection
def setup_connection(uri: str, auth: Tuple[str, str]):
    """Setup neomodel database connection."""
    username, password = auth
    config.DATABASE_URL = f"bolt://{username}:{password}@{uri.split('://')[-1]}"


class ExperientialSummary(StructuredNode):
    """
    Node model for experiential summaries.
    
    This model represents first-person narrative summaries of conversations,
    designed to be extended with other memory types in the future.
    """
    # Required properties
    conversation_id = StringProperty(required=True, index=True)
    title = StringProperty(required=True)
    content = StringProperty(required=True)
    
    # Embedding stored as array of floats
    embedding = ArrayProperty(FloatProperty(), required=True)
    
    # Metadata
    created_at = DateTimeProperty(default_now=True)
    embedding_model = StringProperty(required=True)
    message_count = IntegerProperty(default=0)
    word_count = IntegerProperty()
    
    # Contextual tags (to be populated by separate agent)
    emotional_context = JSONProperty()  # {"primary": "curious", "secondary": ["frustrated", "excited"], "intensity": 0.8}
    environmental_context = JSONProperty()  # {"location": "home", "time_of_day": "evening", "weather": "rainy", "inferred": true}
    activity_context = JSONProperty()  # {"activity": "debugging", "tools": ["python", "docker"], "domain": "software development"}
    social_context = JSONProperty()  # {"interaction_type": "learning", "formality": "casual", "audience": "ai_assistant"}
    
    def save(self):
        """Override save to auto-calculate word count."""
        self.word_count = len(self.content.split()) if self.content else 0
        return super().save()
    
    @classmethod
    def create_from_summary(
        cls,
        conversation_id: str,
        title: str,
        content: str,
        embedding: List[float],
        embedding_model: str,
        message_count: int = 0
    ) -> 'ExperientialSummary':
        """
        Factory method to create an experiential summary.
        
        Args:
            conversation_id: ID of source conversation
            title: Conversation title
            content: First-person summary
            embedding: Vector embedding
            embedding_model: Model used for embedding
            message_count: Messages in original conversation
            
        Returns:
            Created ExperientialSummary instance
        """
        summary = cls(
            conversation_id=conversation_id,
            title=title,
            content=content,
            embedding=embedding,
            embedding_model=embedding_model,
            message_count=message_count
        )
        summary.save()
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.element_id,
            'conversation_id': self.conversation_id,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'message_count': self.message_count,
            'word_count': self.word_count,
            'emotional_context': self.emotional_context,
            'environmental_context': self.environmental_context,
            'activity_context': self.activity_context,
            'social_context': self.social_context
        }


class Neo4jMemoryStore:
    """
    High-level store for managing experiential summaries using neomodel.
    """
    
    def __init__(
        self,
        uri: str,
        auth: Tuple[str, str],
        embedding_model: str = "nomic-embed-text",
        embedding_dimension: int = 768
    ):
        """
        Initialize store with Neo4j connection.
        
        Args:
            uri: Neo4j URI (e.g., "bolt://localhost:7687")
            auth: (username, password) tuple
            embedding_model: Name of embedding model
            embedding_dimension: Vector dimension
        """
        setup_connection(uri, auth)
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.vector_index_name = "experiential_summary_embeddings"
        
    def setup_schema(self):
        """
        Create vector index for similarity search.
        
        Note: Constraints are handled automatically by neomodel.
        """
        # Install all constraints and indexes from models
        db.install_all_labels()
        
        # Create vector index using raw Cypher
        # (neomodel doesn't support vector indexes yet)
        query = f"""
        CREATE VECTOR INDEX {self.vector_index_name} IF NOT EXISTS
        FOR (n:ExperientialSummary) 
        ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        try:
            db.cypher_query(query)
        except Exception as e:
            # Index might already exist
            pass
    
    def create_experiential_summary(
        self,
        conversation_id: str,
        title: str,
        content: str,
        embedding: List[float],
        message_count: int = 0
    ) -> str:
        """
        Create an experiential summary.
        
        Args:
            conversation_id: Source conversation ID
            title: Conversation title
            content: First-person narrative summary
            embedding: Vector embedding
            message_count: Number of messages
            
        Returns:
            The element ID of created summary
        """
        summary = ExperientialSummary.create_from_summary(
            conversation_id=conversation_id,
            title=title,
            content=content,
            embedding=embedding,
            embedding_model=self.embedding_model,
            message_count=message_count
        )
        return summary.element_id
    
    def search_similar_summaries(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar summaries using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of summaries with similarity scores
        """
        query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node, score
        MATCH (es:ExperientialSummary)
        WHERE elementId(es) = elementId(node)
        RETURN es, score
        ORDER BY score DESC
        """
        
        results, _ = db.cypher_query(
            query,
            {
                'index_name': self.vector_index_name,
                'top_k': top_k,
                'query_embedding': query_embedding
            }
        )
        
        return [
            {
                'summary': ExperientialSummary.inflate(row[0]).to_dict(),
                'score': row[1]
            }
            for row in results
        ]
    
    def update_contextual_tags(
        self,
        summary_id: str,
        emotional_context: Optional[Dict[str, Any]] = None,
        environmental_context: Optional[Dict[str, Any]] = None,
        activity_context: Optional[Dict[str, Any]] = None,
        social_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update contextual tags for an existing summary.
        
        This allows the contextual tagging agent to run separately
        and update summaries with extracted context.
        
        Args:
            summary_id: Element ID of the summary to update
            emotional_context: Emotional state data
            environmental_context: Location/time/weather data
            activity_context: Activity and domain data
            social_context: Interaction type data
            
        Returns:
            True if update successful
        """
        # TODO: Implement update logic
        # This will be used by the separate contextual tagging agent
        pass