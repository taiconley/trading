"""
Strategy Registry

This module provides dynamic strategy loading, registration, and validation.
It allows strategies to be loaded from the database configuration and
instantiated at runtime with proper parameter validation.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import sys
import os

from pydantic import BaseModel, ValidationError

from .base import BaseStrategy, StrategyConfig, StrategyState


logger = logging.getLogger(__name__)


class StrategyRegistrationError(Exception):
    """Raised when strategy registration fails."""
    pass


class StrategyLoadError(Exception):
    """Raised when strategy loading fails."""
    pass


class StrategyInfo(BaseModel):
    """Information about a registered strategy class."""
    name: str
    class_name: str
    module_path: str
    description: Optional[str] = None
    parameters_schema: Dict[str, Any] = {}
    default_config: Dict[str, Any] = {}
    
    class Config:
        extra = "allow"


class StrategyRegistry:
    """
    Registry for managing strategy classes and instances.
    
    This class handles:
    - Dynamic loading of strategy classes from modules
    - Strategy registration and validation
    - Parameter schema management
    - Strategy instantiation with configuration
    """
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._strategy_info: Dict[str, StrategyInfo] = {}
        self._strategy_instances: Dict[str, BaseStrategy] = {}
        self._loaded_modules: set = set()
    
    def register_strategy(self, strategy_class: Type[BaseStrategy], 
                         name: Optional[str] = None,
                         description: Optional[str] = None,
                         default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_class: The strategy class to register
            name: Strategy name (defaults to class name)
            description: Strategy description
            default_config: Default configuration parameters
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise StrategyRegistrationError(
                f"Strategy class {strategy_class.__name__} must inherit from BaseStrategy"
            )
        
        strategy_name = name or strategy_class.__name__
        
        # Extract parameter schema from the strategy class if it has one
        parameters_schema = {}
        if hasattr(strategy_class, 'get_parameters_schema'):
            try:
                parameters_schema = strategy_class.get_parameters_schema()
            except Exception as e:
                logger.warning(f"Failed to get parameter schema for {strategy_name}: {e}")
        
        # Store strategy class and info
        self._strategies[strategy_name] = strategy_class
        self._strategy_info[strategy_name] = StrategyInfo(
            name=strategy_name,
            class_name=strategy_class.__name__,
            module_path=strategy_class.__module__,
            description=description or strategy_class.__doc__,
            parameters_schema=parameters_schema,
            default_config=default_config or {}
        )
        
        logger.info(f"Registered strategy: {strategy_name}")
    
    def unregister_strategy(self, name: str) -> None:
        """Unregister a strategy."""
        if name in self._strategies:
            del self._strategies[name]
            del self._strategy_info[name]
            logger.info(f"Unregistered strategy: {name}")
    
    def load_strategies_from_directory(self, directory: Union[str, Path]) -> int:
        """
        Load all strategy classes from a directory.
        
        Args:
            directory: Directory path containing strategy modules
            
        Returns:
            Number of strategies loaded
        """
        directory = Path(directory)
        if not directory.exists():
            raise StrategyLoadError(f"Strategy directory does not exist: {directory}")
        
        # Add directory to Python path if not already there
        directory_str = str(directory.parent)
        if directory_str not in sys.path:
            sys.path.insert(0, directory_str)
        
        loaded_count = 0
        
        # Find all Python files in the directory
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name.startswith("test_"):
                continue  # Skip private modules and test files
                
            module_name = py_file.stem
            module_path = f"{directory.name}.{module_name}"
            
            try:
                loaded_count += self._load_module(module_path)
            except Exception as e:
                logger.error(f"Failed to load strategies from {py_file}: {e}")
        
        return loaded_count
    
    def load_strategy_from_module(self, module_path: str) -> int:
        """
        Load strategies from a specific module.
        
        Args:
            module_path: Python module path (e.g., 'strategy_lib.examples.sma_cross')
            
        Returns:
            Number of strategies loaded
        """
        return self._load_module(module_path)
    
    def _load_module(self, module_path: str) -> int:
        """Load strategies from a module."""
        if module_path in self._loaded_modules:
            return 0  # Already loaded
        
        try:
            module = importlib.import_module(module_path)
            self._loaded_modules.add(module_path)
            
            loaded_count = 0
            
            # Find all strategy classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (obj != BaseStrategy and 
                    issubclass(obj, BaseStrategy) and 
                    obj.__module__ == module_path):
                    
                    # Auto-register the strategy
                    try:
                        self.register_strategy(obj)
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to register strategy {name}: {e}")
            
            logger.info(f"Loaded {loaded_count} strategies from {module_path}")
            return loaded_count
            
        except ImportError as e:
            raise StrategyLoadError(f"Failed to import module {module_path}: {e}")
        except Exception as e:
            raise StrategyLoadError(f"Error loading strategies from {module_path}: {e}")
    
    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """Get a registered strategy class by name."""
        return self._strategies.get(name)
    
    def get_strategy_info(self, name: str) -> Optional[StrategyInfo]:
        """Get information about a registered strategy."""
        return self._strategy_info.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    def get_all_strategy_info(self) -> Dict[str, StrategyInfo]:
        """Get information about all registered strategies."""
        return self._strategy_info.copy()
    
    def create_strategy_instance(self, name: str, config: Union[StrategyConfig, Dict[str, Any]]) -> BaseStrategy:
        """
        Create an instance of a registered strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration (StrategyConfig or dict)
            
        Returns:
            Strategy instance
        """
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            raise StrategyLoadError(f"Strategy not found: {name}")
        
        # Convert dict to StrategyConfig if needed
        if isinstance(config, dict):
            try:
                config = StrategyConfig(**config)
            except ValidationError as e:
                raise StrategyLoadError(f"Invalid configuration for {name}: {e}")
        
        # Validate configuration against strategy schema if available
        strategy_info = self.get_strategy_info(name)
        if strategy_info and strategy_info.parameters_schema:
            self._validate_strategy_parameters(config.parameters, strategy_info.parameters_schema)
        
        try:
            instance = strategy_class(config)
            instance_id = f"{name}_{config.strategy_id}"
            self._strategy_instances[instance_id] = instance
            logger.info(f"Created strategy instance: {instance_id}")
            return instance
        except Exception as e:
            raise StrategyLoadError(f"Failed to create strategy instance {name}: {e}")
    
    def get_strategy_instance(self, instance_id: str) -> Optional[BaseStrategy]:
        """Get a strategy instance by ID."""
        return self._strategy_instances.get(instance_id)
    
    def remove_strategy_instance(self, instance_id: str) -> None:
        """Remove a strategy instance."""
        if instance_id in self._strategy_instances:
            del self._strategy_instances[instance_id]
            logger.info(f"Removed strategy instance: {instance_id}")
    
    def list_strategy_instances(self) -> List[str]:
        """List all active strategy instance IDs."""
        return list(self._strategy_instances.keys())
    
    def _validate_strategy_parameters(self, parameters: Dict[str, Any], 
                                    schema: Dict[str, Any]) -> None:
        """Validate strategy parameters against schema."""
        # This is a basic validation - could be enhanced with jsonschema
        for param_name, param_info in schema.items():
            if param_info.get('required', False) and param_name not in parameters:
                raise ValidationError(f"Required parameter missing: {param_name}")
            
            if param_name in parameters:
                param_value = parameters[param_name]
                param_type = param_info.get('type')
                
                if param_type and not isinstance(param_value, param_type):
                    try:
                        # Try to convert the type
                        parameters[param_name] = param_type(param_value)
                    except (ValueError, TypeError):
                        raise ValidationError(
                            f"Parameter {param_name} must be of type {param_type.__name__}"
                        )
    
    def reload_strategies(self) -> int:
        """
        Reload all strategies from their modules.
        
        Returns:
            Number of strategies reloaded
        """
        # Clear current registrations
        old_strategies = list(self._strategies.keys())
        self._strategies.clear()
        self._strategy_info.clear()
        
        # Reload modules
        loaded_modules = list(self._loaded_modules)
        self._loaded_modules.clear()
        
        total_loaded = 0
        for module_path in loaded_modules:
            try:
                # Force reload the module
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                total_loaded += self._load_module(module_path)
            except Exception as e:
                logger.error(f"Failed to reload module {module_path}: {e}")
        
        logger.info(f"Reloaded {total_loaded} strategies (was {len(old_strategies)})")
        return total_loaded


# Global registry instance
_global_registry: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry


def register_strategy(strategy_class: Type[BaseStrategy], 
                     name: Optional[str] = None,
                     description: Optional[str] = None,
                     default_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a strategy class with the global registry.
    
    This is a convenience function for the most common use case.
    """
    registry = get_strategy_registry()
    registry.register_strategy(strategy_class, name, description, default_config)


def load_strategies_from_directory(directory: Optional[Union[str, Path]] = None) -> int:
    """
    Load all strategies from the specified directory or strategy_lib directory.
    
    Args:
        directory: Directory to load from (defaults to strategy_lib directory)
        
    Returns:
        Number of strategies loaded
    """
    registry = get_strategy_registry()
    
    if directory is None:
        # Get the strategy_lib directory (this file's parent)
        directory = Path(__file__).parent
    
    return registry.load_strategies_from_directory(directory)


def create_strategy_from_db_config(db_strategy_config: Dict[str, Any]) -> BaseStrategy:
    """
    Create a strategy instance from database configuration.
    
    Args:
        db_strategy_config: Configuration from database (strategies table)
        
    Returns:
        Strategy instance
    """
    registry = get_strategy_registry()
    
    # Extract strategy name from config
    strategy_name = db_strategy_config.get('name')
    if not strategy_name:
        raise StrategyLoadError("Strategy name not found in configuration")
    
    # Build StrategyConfig from database data
    config_data = {
        'strategy_id': db_strategy_config.get('strategy_id'),
        'name': strategy_name,
        'enabled': db_strategy_config.get('enabled', False),
        'symbols': db_strategy_config.get('symbols', []),
        'parameters': db_strategy_config.get('params_json', {})
    }
    
    # Add any additional configuration from params_json
    params_json = db_strategy_config.get('params_json', {})
    if isinstance(params_json, dict):
        config_data.update(params_json)
    
    return registry.create_strategy_instance(strategy_name, config_data)


# Decorator for easy strategy registration
def strategy(name: Optional[str] = None, 
            description: Optional[str] = None,
            default_config: Optional[Dict[str, Any]] = None):
    """
    Decorator to automatically register a strategy class.
    
    Usage:
        @strategy(name="MyStrategy", description="A sample strategy")
        class MyStrategy(BaseStrategy):
            ...
    """
    def decorator(cls: Type[BaseStrategy]):
        register_strategy(cls, name, description, default_config)
        return cls
    return decorator
