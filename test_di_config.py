#!/usr/bin/env python3
"""Test the dependency injection configuration system."""

import os
from apiana.di import get_configuration, get_injector, reset_injector
from apiana.di.container import get_module_for_environment


def test_di_configuration():
    """Test different environment configurations."""
    
    print("Testing Dependency Injection Configuration System\n")
    
    # Test local configuration (default)
    print("1. Testing LOCAL configuration (default):")
    config = get_configuration()
    print(f"   Environment: {config.environment}")
    print(f"   LLM Model: {config.llm_provider.model}")
    print(f"   LLM Base URL: {config.llm_provider.base_url}")
    print(f"   Neo4j URI: {config.neo4j.uri}")
    print(f"   Embedder: {config.embedder.model}\n")
    
    # Test environment switching
    print("2. Testing environment module selection:")
    for env in ["local", "dev", "production"]:
        try:
            module = get_module_for_environment(env)
            print(f"   {env}: {module.__name__}")
        except ValueError as e:
            print(f"   {env}: ERROR - {e}")
    print()
    
    # Test getting same instance
    print("3. Testing singleton behavior:")
    injector1 = get_injector()
    injector2 = get_injector()
    print(f"   Same injector instance: {injector1 is injector2}")
    
    config1 = get_configuration()
    config2 = get_configuration()
    print(f"   Same config instance: {config1 is config2}\n")
    
    # Test force_new
    print("4. Testing force_new parameter:")
    reset_injector()
    config3 = get_configuration(force_new=True)
    config4 = get_configuration()
    print(f"   Different config after force_new: {config3 is not config4}\n")
    
    # Test environment variable switching
    print("5. Testing environment variable switching:")
    reset_injector()
    os.environ["APIANA_ENVIRONMENT_STAGE"] = "dev"
    config_dev = get_configuration()
    print(f"   Environment: {config_dev.environment}")
    print(f"   LLM Base URL: {config_dev.llm_provider.base_url}")
    
    # Reset environment
    os.environ.pop("APIANA_ENVIRONMENT_STAGE", None)
    reset_injector()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    test_di_configuration()