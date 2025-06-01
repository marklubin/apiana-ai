"""Dependency injection container and module selection."""

import os
from typing import Type, Optional
from injector import Injector, Module

from apiana.configuration import ChatGPTExportProcessorConfiguration
from .modules import (
    LocalConfigModule,
    DevConfigModule,
    ProductionConfigModule
)


_injector: Optional[Injector] = None


def get_module_for_environment(environment: Optional[str] = None) -> Type[Module]:
    """Get the appropriate configuration module based on environment.
    
    Args:
        environment: Environment name. If None, reads from APIANA_ENVIRONMENT_STAGE
        
    Returns:
        Configuration module class for the environment
        
    Raises:
        ValueError: If environment is not recognized
    """
    env = environment or os.getenv("APIANA_ENVIRONMENT_STAGE", "local")
    
    modules = {
        "local": LocalConfigModule,
        "dev": DevConfigModule,
        "production": ProductionConfigModule,
        "prod": ProductionConfigModule,  # Alias
    }
    
    module_class = modules.get(env.lower())
    if not module_class:
        raise ValueError(
            f"Unknown environment: {env}. "
            f"Valid options: {', '.join(modules.keys())}"
        )
    
    return module_class


def create_injector(environment: Optional[str] = None) -> Injector:
    """Create a new injector for the specified environment.
    
    Args:
        environment: Environment name. If None, reads from APIANA_ENVIRONMENT_STAGE
        
    Returns:
        Configured Injector instance
    """
    module_class = get_module_for_environment(environment)
    return Injector([module_class()])


def get_injector(environment: Optional[str] = None, force_new: bool = False) -> Injector:
    """Get the singleton injector instance.
    
    By default, returns a cached injector instance. This ensures consistent
    configuration throughout the application lifetime.
    
    Args:
        environment: Environment name. If None, reads from APIANA_ENVIRONMENT_STAGE
        force_new: If True, creates a new injector instead of using cached one
        
    Returns:
        Configured Injector instance
    """
    global _injector
    
    if force_new or _injector is None:
        _injector = create_injector(environment)
    
    return _injector


def get_configuration(
    environment: Optional[str] = None,
    force_new: bool = False
) -> ChatGPTExportProcessorConfiguration:
    """Convenience function to get the processor configuration.
    
    Args:
        environment: Environment name. If None, reads from APIANA_ENVIRONMENT_STAGE
        force_new: If True, creates a new injector instead of using cached one
        
    Returns:
        Configured ChatGPTExportProcessorConfiguration instance
    """
    injector = get_injector(environment, force_new)
    return injector.get(ChatGPTExportProcessorConfiguration)


def reset_injector() -> None:
    """Reset the global injector instance.
    
    Useful for testing or when you need to switch environments at runtime.
    """
    global _injector
    _injector = None