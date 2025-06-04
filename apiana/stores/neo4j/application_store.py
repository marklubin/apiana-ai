"""
Application-level data store for common shared data.

This store uses the default Neo4j database to store common application data
including all ChatFragments, pipeline run metadata, and shared resources.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from neomodel import config as neomodel_config
from neomodel import db, StringProperty, DateTimeProperty, JSONProperty, StructuredNode, RelationshipTo, ArrayProperty

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
    
    # Timestamps
    stored_at = DateTimeProperty(default=datetime.utcnow)
    updated_at = DateTimeProperty(default=datetime.utcnow)


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


class ApplicationStore:
    """
    Application-level data store using the default Neo4j database.
    
    This store handles:
    - Storage and retrieval of ChatFragments
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
    
    def store_fragment(self, fragment: ChatFragment, tags: Optional[List[str]] = None) -> ChatFragmentNode:
        """
        Store a ChatFragment in the default database.
        
        Args:
            fragment: The ChatFragment to store
            tags: Optional list of tags to associate with the fragment
            
        Returns:
            The created ChatFragmentNode
        """
        # Check if fragment already exists
        existing = ChatFragmentNode.nodes.filter(fragment_id=fragment.fragment_id).first()
        
        if existing:
            # Update existing fragment
            existing.title = fragment.title
            existing.openai_conversation_id = fragment.openai_conversation_id
            existing.create_time = fragment.create_time
            existing.update_time = fragment.update_time
            existing.messages = fragment.messages
            existing.message_metadata = fragment.message_metadata or []
            existing.updated_at = datetime.utcnow()
            
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
                tags=tags or []
            ).save()
            
            return node
    
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
        conversation_id: Optional[str] = None
    ) -> List[ChatFragment]:
        """
        List ChatFragments with optional filtering.
        
        Args:
            limit: Maximum number of fragments to return
            offset: Number of fragments to skip
            tags: Filter by tags (fragments must have ALL specified tags)
            conversation_id: Filter by OpenAI conversation ID
            
        Returns:
            List of ChatFragments matching the criteria
        """
        query = ChatFragmentNode.nodes
        
        # Apply filters
        if conversation_id:
            query = query.filter(openai_conversation_id=conversation_id)
        
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
        # Note: This is a simple implementation. For better full-text search,
        # you'd want to use Neo4j's full-text search capabilities
        
        # Search in titles
        title_matches = ChatFragmentNode.nodes.filter(
            title__icontains=search_text
        ).order_by('-create_time')[:limit]
        
        results = [self._node_to_fragment(node) for node in title_matches]
        
        # If we haven't hit the limit, search in message content
        if len(results) < limit:
            # This is a simplified approach - in practice you'd want better indexing
            all_nodes = ChatFragmentNode.nodes.order_by('-create_time')
            
            for node in all_nodes:
                if len(results) >= limit:
                    break
                
                # Skip if already in results
                if any(r.fragment_id == node.fragment_id for r in results):
                    continue
                
                # Search in message content
                messages = node.messages or []
                if any(search_text.lower() in str(msg.get('content', '')).lower() for msg in messages):
                    results.append(self._node_to_fragment(node))
        
        return results[:limit]
    
    def delete_fragment(self, fragment_id: str) -> bool:
        """
        Delete a ChatFragment by ID.
        
        Args:
            fragment_id: The fragment ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        node = ChatFragmentNode.nodes.filter(fragment_id=fragment_id).first()
        if node:
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
        
        # Get recent activity
        recent_fragments = len(ChatFragmentNode.nodes.filter(
            stored_at__gte=datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        ))
        
        return {
            "total_fragments": total_fragments,
            "total_pipeline_runs": total_runs,
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