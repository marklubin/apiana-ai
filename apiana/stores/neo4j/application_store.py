"""
Application-level data store for common shared data.

This store uses the default Neo4j database to store common application data
including all ChatFragments, pipeline run metadata, embeddings, and shared resources.
"""

import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any

from neomodel import config as neomodel_config
from neomodel import (
    db, StringProperty, DateTimeProperty, JSONProperty, FloatProperty,
    IntegerProperty, StructuredNode, RelationshipTo, RelationshipFrom, ArrayProperty
)

from apiana.types.configuration import Neo4jConfig
from apiana.types.chat_fragment import ChatFragment


class ChatFragmentNode(StructuredNode):
    """Neo4j node representing a ChatFragment."""
    
    # Core identifiers
    fragment_id = StringProperty(unique_index=True, required=True)
    openai_conversation_id = StringProperty(index=True)
    
    # Metadata
    title = StringProperty()
    create_time = DateTimeProperty()
    update_time = DateTimeProperty()
    
    # Content stored as JSON
    messages = JSONProperty()
    message_metadata = JSONProperty()
    
    # Processing metadata
    processed_by_runs = ArrayProperty(StringProperty())  # List of PipelineRun IDs
    tags = ArrayProperty(StringProperty())
    
    # Summary tracking
    summary_agent_id = StringProperty(index=True)  # Agent that created the summary
    summary_hash = StringProperty(index=True)  # Hash of summary content
    
    # Timestamps
    stored_at = DateTimeProperty(default=datetime.utcnow)
    updated_at = DateTimeProperty(default=datetime.utcnow)
    
    # Relationships
    embeddings = RelationshipTo('EmbeddingNode', 'HAS_EMBEDDING')


class EmbeddingNode(StructuredNode):
    """Neo4j node representing an embedding vector."""
    
    # Unique identifier
    embedding_id = StringProperty(unique_index=True, required=True)
    
    # Source information
    source_type = StringProperty(required=True)  # "summary", "fragment", etc.
    source_id = StringProperty(required=True, index=True)  # ID of source (fragment_id, etc.)
    source_hash = StringProperty(required=True, index=True)  # Hash of source content
    
    # Model information
    model_provider = StringProperty(required=True)  # "openai", "cohere", etc.
    model_name = StringProperty(required=True)  # "text-embedding-ada-002", etc.
    model_dimensions = IntegerProperty()  # Vector dimensions
    
    # Embedding data
    vector = ArrayProperty(FloatProperty(), required=True)  # The actual embedding vector
    
    # Metadata
    created_at = DateTimeProperty(default=datetime.utcnow)
    metadata = JSONProperty()  # Additional metadata (token count, etc.)
    
    # Relationships
    source_fragment = RelationshipFrom('ChatFragmentNode', 'HAS_EMBEDDING')


class PipelineRunNode(StructuredNode):
    """Neo4j node representing a pipeline run."""
    
    run_id = StringProperty(unique_index=True, required=True)
    name = StringProperty(required=True)
    started_at = DateTimeProperty(default=datetime.utcnow)
    completed_at = DateTimeProperty()
    status = StringProperty(default="running")  # running, completed, failed
    
    # Configuration and metadata
    config = JSONProperty()
    stats = JSONProperty()
    errors = JSONProperty()
    
    # Relationships
    processed_fragments = RelationshipTo(ChatFragmentNode, "PROCESSED")


def create_embedding_hash(
    summary_text: str,
    model_provider: str,
    model_name: str
) -> str:
    """
    Create a hash key for an embedding based on summary text and model.
    
    Args:
        summary_text: The summary text that was embedded
        model_provider: The embedding model provider
        model_name: The specific model name
        
    Returns:
        A hash string for the embedding
    """
    hasher = hashlib.sha256()
    
    # Hash the summary text
    hasher.update(summary_text.encode('utf-8'))
    
    # Add provider and model (normalized to lowercase)
    hasher.update(model_provider.lower().encode('utf-8'))
    hasher.update(model_name.lower().encode('utf-8'))
    
    return hasher.hexdigest()[:32]  # Use first 32 chars for reasonable length


class ApplicationStore:
    """
    Application-level data store using the default Neo4j database.
    
    This store handles:
    - Storage and retrieval of ChatFragments
    - Storage and retrieval of embeddings
    - Tracking pipeline runs and their relationships to fragments
    - Querying fragments by various criteria
    - Managing application-level metadata
    """
    
    def __init__(self, config: Neo4jConfig):
        """
        Initialize the application store using the default database.
        
        Args:
            config: Neo4j connection configuration
        """
        self.config = config
        
        # Configure connection to default database (no database suffix)
        config_url = (
            f"bolt://{config.username}:{config.password}@{config.host}:{config.port}"
        )
        
        neomodel_config.DATABASE_URL = config_url
        
        # Install labels for application-level data
        db.install_all_labels()
    
    def store_fragment(
        self, 
        fragment: ChatFragment, 
        tags: Optional[List[str]] = None,
        summary_agent_id: Optional[str] = None,
        summary_text: Optional[str] = None
    ) -> ChatFragmentNode:
        """
        Store a ChatFragment in the default database.
        
        Args:
            fragment: The ChatFragment to store
            tags: Optional list of tags to associate with the fragment
            summary_agent_id: Optional ID of the agent that created a summary
            summary_text: Optional summary text (used to create summary hash)
            
        Returns:
            The created ChatFragmentNode
        """
        # Check if fragment already exists
        existing = ChatFragmentNode.nodes.filter(fragment_id=fragment.fragment_id).first()
        
        # Calculate summary hash if summary text provided
        summary_hash = None
        if summary_text:
            summary_hash = hashlib.sha256(summary_text.encode('utf-8')).hexdigest()[:32]
        
        if existing:
            # Update existing fragment
            existing.title = fragment.title
            existing.openai_conversation_id = fragment.openai_conversation_id
            existing.create_time = fragment.create_time
            existing.update_time = fragment.update_time
            existing.messages = fragment.messages
            existing.message_metadata = fragment.message_metadata or []
            existing.updated_at = datetime.utcnow()
            
            if summary_agent_id:
                existing.summary_agent_id = summary_agent_id
            
            if summary_hash:
                existing.summary_hash = summary_hash
            
            if tags:
                # Merge tags (avoid duplicates)
                existing_tags = set(existing.tags or [])
                new_tags = set(tags)
                existing.tags = list(existing_tags | new_tags)
            
            existing.save()
            return existing
        else:
            # Create new fragment
            node = ChatFragmentNode(
                fragment_id=fragment.fragment_id,
                openai_conversation_id=fragment.openai_conversation_id,
                title=fragment.title,
                create_time=fragment.create_time,
                update_time=fragment.update_time,
                messages=fragment.messages,
                message_metadata=fragment.message_metadata or [],
                tags=tags or [],
                summary_agent_id=summary_agent_id,
                summary_hash=summary_hash
            ).save()
            
            return node
    
    def store_embedding(
        self,
        source_id: str,
        source_type: str,
        source_text: str,
        embedding_vector: List[float],
        model_provider: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingNode:
        """
        Store an embedding vector.
        
        Args:
            source_id: ID of the source (e.g., fragment_id)
            source_type: Type of source ("summary", "fragment", etc.)
            source_text: The text that was embedded
            embedding_vector: The embedding vector
            model_provider: Provider of the embedding model
            model_name: Name of the embedding model
            metadata: Optional additional metadata
            
        Returns:
            The created EmbeddingNode
        """
        # Create hash of source content and model
        source_hash = create_embedding_hash(source_text, model_provider, model_name)
        
        # Check if this exact embedding already exists
        existing = EmbeddingNode.nodes.filter(
            source_id=source_id,
            source_hash=source_hash
        ).first()
        
        if existing:
            # Update existing embedding
            existing.vector = embedding_vector
            existing.metadata = metadata or {}
            existing.save()
            return existing
        
        # Create new embedding
        embedding_id = f"{source_id}:{source_hash}"
        
        embedding_node = EmbeddingNode(
            embedding_id=embedding_id,
            source_type=source_type,
            source_id=source_id,
            source_hash=source_hash,
            model_provider=model_provider,
            model_name=model_name,
            model_dimensions=len(embedding_vector),
            vector=embedding_vector,
            metadata=metadata or {}
        ).save()
        
        # Link to source fragment if it exists
        if source_type == "summary":
            fragment_node = ChatFragmentNode.nodes.filter(fragment_id=source_id).first()
            if fragment_node:
                fragment_node.embeddings.connect(embedding_node)
        
        return embedding_node
    
    def get_embedding(
        self,
        source_id: str,
        model_provider: str,
        model_name: str,
        source_text: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an embedding by source and model.
        
        Args:
            source_id: ID of the source
            model_provider: Provider of the embedding model
            model_name: Name of the embedding model
            source_text: Optional source text to verify exact match
            
        Returns:
            Dictionary with embedding data if found, None otherwise
        """
        query = EmbeddingNode.nodes.filter(
            source_id=source_id,
            model_provider=model_provider,
            model_name=model_name
        )
        
        if source_text:
            # If source text provided, match exact content hash
            source_hash = create_embedding_hash(source_text, model_provider, model_name)
            query = query.filter(source_hash=source_hash)
        
        embedding_node = query.first()
        
        if not embedding_node:
            return None
        
        return {
            "embedding_id": embedding_node.embedding_id,
            "vector": embedding_node.vector,
            "model_provider": embedding_node.model_provider,
            "model_name": embedding_node.model_name,
            "model_dimensions": embedding_node.model_dimensions,
            "source_hash": embedding_node.source_hash,
            "created_at": embedding_node.created_at,
            "metadata": embedding_node.metadata
        }
    
    def list_embeddings(
        self,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        model_provider: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List embeddings with optional filtering.
        
        Args:
            source_id: Filter by source ID
            source_type: Filter by source type
            model_provider: Filter by model provider
            limit: Maximum number of results
            
        Returns:
            List of embedding dictionaries
        """
        query = EmbeddingNode.nodes
        
        if source_id:
            query = query.filter(source_id=source_id)
        
        if source_type:
            query = query.filter(source_type=source_type)
        
        if model_provider:
            query = query.filter(model_provider=model_provider)
        
        query = query.order_by('-created_at')[:limit]
        
        return [
            {
                "embedding_id": node.embedding_id,
                "source_id": node.source_id,
                "source_type": node.source_type,
                "model_provider": node.model_provider,
                "model_name": node.model_name,
                "model_dimensions": node.model_dimensions,
                "created_at": node.created_at,
                "metadata": node.metadata
            }
            for node in query
        ]
    
    def get_fragment(self, fragment_id: str) -> Optional[ChatFragment]:
        """
        Retrieve a ChatFragment by its ID.
        
        Args:
            fragment_id: The unique fragment identifier
            
        Returns:
            ChatFragment if found, None otherwise
        """
        node = ChatFragmentNode.nodes.filter(fragment_id=fragment_id).first()
        if not node:
            return None
        
        return self._node_to_fragment(node)
    
    def list_fragments(
        self, 
        limit: Optional[int] = None, 
        offset: int = 0,
        tags: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        summary_agent_id: Optional[str] = None
    ) -> List[ChatFragment]:
        """
        List ChatFragments with optional filtering.
        
        Args:
            limit: Maximum number of fragments to return
            offset: Number of fragments to skip
            tags: Filter by tags (fragments must have ALL specified tags)
            conversation_id: Filter by OpenAI conversation ID
            summary_agent_id: Filter by summary agent ID
            
        Returns:
            List of ChatFragments matching the criteria
        """
        query = ChatFragmentNode.nodes
        
        # Apply filters
        if conversation_id:
            query = query.filter(openai_conversation_id=conversation_id)
        
        if summary_agent_id:
            query = query.filter(summary_agent_id=summary_agent_id)
        
        if tags:
            # Filter fragments that contain all specified tags
            for tag in tags:
                query = query.filter(tags__contains=tag)
        
        # Apply ordering, offset, and limit
        query = query.order_by('-create_time')
        
        if offset > 0:
            query = query[offset:]
        
        if limit:
            query = query[:limit]
        
        return [self._node_to_fragment(node) for node in query]
    
    def search_fragments(self, search_text: str, limit: int = 50) -> List[ChatFragment]:
        """
        Search fragments by text content in title or messages.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching ChatFragments
        """
        # This is a simple implementation. For production, you'd want
        # to use Neo4j's full-text search capabilities
        query = """
        MATCH (f:ChatFragmentNode)
        WHERE f.title CONTAINS $search_text
        RETURN f
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {"search_text": search_text, "limit": limit})
        
        fragments = []
        for row in results:
            node = ChatFragmentNode.inflate(row[0])
            fragments.append(self._node_to_fragment(node))
        
        return fragments
    
    def delete_fragment(self, fragment_id: str) -> bool:
        """
        Delete a ChatFragment and its relationships.
        
        Args:
            fragment_id: The fragment ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        node = ChatFragmentNode.nodes.filter(fragment_id=fragment_id).first()
        if node:
            # Delete associated embeddings
            for embedding in node.embeddings.all():
                embedding.delete()
            
            # Delete the fragment
            node.delete()
            return True
        
        return False
    
    def create_pipeline_run(
        self, 
        run_id: str, 
        name: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> PipelineRunNode:
        """
        Create a new pipeline run record.
        
        Args:
            run_id: Unique identifier for the run
            name: Human-readable name for the run
            config: Optional configuration used for the run
            
        Returns:
            The created PipelineRunNode
        """
        run = PipelineRunNode(
            run_id=run_id,
            name=name,
            config=config or {}
        ).save()
        
        return run
    
    def complete_pipeline_run(
        self, 
        run_id: str, 
        stats: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None
    ) -> Optional[PipelineRunNode]:
        """
        Mark a pipeline run as completed.
        
        Args:
            run_id: The run ID to complete
            stats: Optional statistics from the run
            errors: Optional list of errors that occurred
            
        Returns:
            The updated PipelineRunNode if found, None otherwise
        """
        run = PipelineRunNode.nodes.filter(run_id=run_id).first()
        if run:
            run.completed_at = datetime.utcnow()
            run.status = "completed" if not errors else "failed"
            run.stats = stats or {}
            run.errors = errors or []
            run.save()
        
        return run
    
    def link_fragments_to_run(self, run_id: str, fragment_ids: List[str]) -> int:
        """
        Link processed fragments to a pipeline run.
        
        Args:
            run_id: The pipeline run ID
            fragment_ids: List of fragment IDs that were processed
            
        Returns:
            Number of fragments successfully linked
        """
        run = PipelineRunNode.nodes.filter(run_id=run_id).first()
        if not run:
            return 0
        
        linked_count = 0
        for fragment_id in fragment_ids:
            fragment_node = ChatFragmentNode.nodes.filter(fragment_id=fragment_id).first()
            if fragment_node:
                # Create relationship from run to fragment
                run.processed_fragments.connect(fragment_node)
                
                # Add run ID to fragment's processed_by_runs list
                if not fragment_node.processed_by_runs:
                    fragment_node.processed_by_runs = []
                
                if run_id not in fragment_node.processed_by_runs:
                    fragment_node.processed_by_runs.append(run_id)
                    fragment_node.save()
                
                linked_count += 1
        
        return linked_count
    
    def get_pipeline_runs(self, limit: int = 50) -> List[PipelineRunNode]:
        """
        Get recent pipeline runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of PipelineRunNode objects
        """
        return list(PipelineRunNode.nodes.order_by('-started_at')[:limit])
    
    def get_run_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about the stored data.
        
        Returns:
            Dictionary with statistics
        """
        total_fragments = len(ChatFragmentNode.nodes)
        total_runs = len(PipelineRunNode.nodes)
        total_embeddings = len(EmbeddingNode.nodes)
        
        # Get recent activity
        recent_fragments = len(ChatFragmentNode.nodes.filter(
            stored_at__gte=datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        ))
        
        return {
            "total_fragments": total_fragments,
            "total_pipeline_runs": total_runs,
            "total_embeddings": total_embeddings,
            "fragments_added_today": recent_fragments,
        }
    
    def _node_to_fragment(self, node: ChatFragmentNode) -> ChatFragment:
        """Convert a ChatFragmentNode to a ChatFragment."""
        return ChatFragment(
            fragment_id=node.fragment_id,
            openai_conversation_id=node.openai_conversation_id,
            title=node.title,
            create_time=node.create_time,
            update_time=node.update_time,
            messages=node.messages or [],
            message_metadata=node.message_metadata or []
        )