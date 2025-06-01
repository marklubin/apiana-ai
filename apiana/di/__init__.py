"""Dependency injection configuration for Apiana."""

from .container import get_injector, get_configuration

__all__ = ["get_injector", "get_configuration"]