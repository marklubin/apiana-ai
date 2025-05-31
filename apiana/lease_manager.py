"""
Generic lease manager for controlling concurrent access to named resources.
"""

import atomics
from typing import Set, Dict, List
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class LeaseInfo:
    """Track lease state for a resource using atomic counter."""
    name: str
    ref_count: atomics.atomic = None
    
    def __post_init__(self):
        if self.ref_count is None:
            self.ref_count = atomics.atomic(width=4, atype=atomics.INT)
            self.ref_count.store(0)

class LeaseManager:
    """
    Generic lease manager for named resources with concurrency limits.
    
    Args:
        resource_names: List of resource names that can be leased
        max_concurrent: Maximum number of different resources that can have active leases
    
    Example:
        # Only 1 GPU model at a time, but unlimited concurrent access to same model
        lease_manager = LeaseManager(["embedder_model", "llm_model"], max_concurrent=1)
        
        with lease_manager.lease("embedder_model"):
            # Do embedding work - lease is held for duration of context
            pass
    """
    
    def __init__(self, resource_names: List[str], max_concurrent: int = 1):
        self.resource_names = set(resource_names)
        self.max_concurrent = max_concurrent
        
        # Track which resources currently have active leases
        self.active_resources: Set[str] = set()
        self.lease_info: Dict[str, LeaseInfo] = {
            name: LeaseInfo(name) for name in resource_names
        }
    
    def _can_acquire_lease(self, resource_name: str) -> bool:
        """Check if we can acquire a lease for this resource."""
        if resource_name not in self.resource_names:
            raise ValueError(f"Unknown resource: {resource_name}")
        
        # If resource already has active leases, we can always add more
        if resource_name in self.active_resources:
            return True
        
        # If we haven't hit the concurrent limit, we can start a new resource
        if len(self.active_resources) < self.max_concurrent:
            return True
        
        return False
    
    def _acquire_lease(self, resource_name: str) -> int:
        """Acquire a lease for the resource. Returns new ref count."""
        lease_info = self.lease_info[resource_name]
        new_count = lease_info.ref_count.fetch_inc() + 1  # fetch_inc returns old value
        
        if new_count == 1:
            # First lease for this resource
            self.active_resources.add(resource_name)
            print(f"Started leasing resource: {resource_name}")
        
        print(f"Acquired lease for {resource_name} (ref_count: {new_count})")
        return new_count
    
    def _release_lease(self, resource_name: str) -> int:
        """Release a lease for the resource. Returns new ref count."""
        lease_info = self.lease_info[resource_name]
        new_count = lease_info.ref_count.fetch_dec() - 1  # fetch_dec returns old value
        print(f"Released lease for {resource_name} (ref_count: {new_count})")
        
        if new_count <= 0:
            # No more leases for this resource
            self.active_resources.discard(resource_name)
            print(f"Stopped leasing resource: {resource_name}")
            return 0
        
        return new_count
    
    @contextmanager
    def lease(self, resource_name: str):
        """
        Context manager to lease a resource. Fails immediately if not available.
        
        Args:
            resource_name: Name of resource to lease
            
        Raises:
            RuntimeError: If lease cannot be acquired
        """
        # Check if we can acquire the lease - fail fast
        if not self._can_acquire_lease(resource_name):
            active_list = list(self.active_resources)
            raise RuntimeError(
                f"Cannot acquire lease for {resource_name}. "
                f"Active resources: {active_list} "
                f"(max_concurrent: {self.max_concurrent})"
            )
        
        # Acquire the lease
        self._acquire_lease(resource_name)
        
        try:
            yield
        finally:
            # Release the lease
            self._release_lease(resource_name)
    
    def get_status(self) -> Dict[str, any]:
        """Get current lease status."""
        return {
            'active_resources': list(self.active_resources),
            'lease_counts': {
                name: info.ref_count.load() 
                for name, info in self.lease_info.items()
                if info.ref_count.load() > 0
            },
            'max_concurrent': self.max_concurrent,
            'available_slots': self.max_concurrent - len(self.active_resources)
        }