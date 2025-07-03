"""
Model analysis utilities for DeepSpeed AutoZero.

Single responsibility: Load PyTorch models and extract parameter information.
Simplified for Phase 2 with evidence-based approach.
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import NamedTuple, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelInfo:
    """Container for essential model analysis results."""

    total_params: int
    trainable_params: int
    model_size_mb: float
    largest_layer_params: int
    largest_layer_name: str
    layer_info: List = field(default_factory=list)  # For backward compatibility
    dtype_size: int = 4  # Default to fp32 size


class ModelAnalyzer:
    """Analyzes PyTorch models for DeepSpeed configuration."""

    def __init__(self):
        self.dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }

    def load_model_from_file(self, model_path: Path) -> nn.Module:
        """
        Load a PyTorch model from a Python file.

        Args:
            model_path: Path to Python file containing model definition

        Returns:
            Instantiated PyTorch model

        Raises:
            ValueError: If no suitable model class found
            ImportError: If module cannot be imported
        """
        # Load the module
        spec = importlib.util.spec_from_file_location("dynamic_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {model_path}")

        module = importlib.util.module_from_spec(spec)

        # Add to sys.modules to avoid import issues
        module_name = f"dynamic_model_{id(module)}"
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Error executing module {model_path}: {str(e)}")

        # Find nn.Module subclasses
        model_classes = []
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, nn.Module)
                and obj != nn.Module
                and obj.__module__ == module.__name__
            ):
                model_classes.append((name, obj))

        if not model_classes:
            raise ValueError(f"No nn.Module subclass found in {model_path}")

        # Use first class found (or prefer one with 'model' in name)
        model_class = model_classes[0][1]
        for name, cls in model_classes:
            if "model" in name.lower():
                model_class = cls
                break

        # Try to instantiate the model
        try:
            # Try with no arguments first
            return model_class()
        except TypeError:
            # Try with common default arguments
            try:
                sig = inspect.signature(model_class.__init__)
                kwargs = {}

                for param_name, param in sig.parameters.items():
                    if param_name == "self" or param.default != inspect.Parameter.empty:
                        continue

                    # Common defaults for transformer models
                    if "dim" in param_name or "hidden" in param_name:
                        kwargs[param_name] = 768
                    elif "vocab" in param_name:
                        kwargs[param_name] = 50000
                    elif "layer" in param_name or "num" in param_name:
                        kwargs[param_name] = 12
                    elif "head" in param_name:
                        kwargs[param_name] = 12

                return model_class(**kwargs)
            except Exception as e:
                raise ValueError(
                    f"Cannot instantiate {model_class.__name__}. "
                    f"Ensure the model can be created with default arguments. "
                    f"Error: {str(e)}"
                )

    def analyze_model(
        self, model: nn.Module, dtype: torch.dtype = torch.float32
    ) -> ModelInfo:
        """
        Analyze model to extract parameter information.

        Args:
            model: PyTorch model to analyze
            dtype: Data type for size calculations

        Returns:
            ModelInfo with parameter statistics
        """
        total_params = 0
        trainable_params = 0
        largest_layer_params = 0
        largest_layer_name = ""
        layer_info = []

        # Count parameters in each module
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                # Skip container modules
                continue

            layer_params = sum(p.numel() for p in module.parameters(recurse=False))
            layer_trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            if layer_params > 0:
                total_params += layer_params
                trainable_params += layer_trainable

                # Add layer info
                layer_info.append(
                    {
                        "name": name or module.__class__.__name__,
                        "params": layer_params,
                        "trainable": layer_trainable,
                        "type": module.__class__.__name__,
                    }
                )

                if layer_params > largest_layer_params:
                    largest_layer_params = layer_params
                    largest_layer_name = name or module.__class__.__name__

        # Calculate model size
        dtype_size = self.dtype_sizes.get(dtype, 4)
        model_size_mb = (total_params * dtype_size) / (1024 * 1024)

        return ModelInfo(
            total_params=total_params,
            trainable_params=trainable_params,
            model_size_mb=model_size_mb,
            largest_layer_params=largest_layer_params,
            largest_layer_name=largest_layer_name,
            layer_info=layer_info,
            dtype_size=dtype_size,
        )

    def estimate_activation_memory(
        self,
        model: nn.Module,
        batch_size: int,
        sequence_length: Optional[int] = None,
        seq_length: Optional[int] = None,  # Alias for backward compatibility
        dtype: torch.dtype = torch.float32,
    ) -> float:
        """
        Estimate activation memory for the model.

        Args:
            model: PyTorch model
            batch_size: Batch size
            sequence_length: Sequence length for transformer models
            dtype: Data type

        Returns:
            Estimated activation memory in MB
        """
        # Use seq_length if provided (for backward compatibility)
        effective_seq_length = seq_length or sequence_length

        # Simple heuristic: estimate based on model size and batch size
        model_info = self.analyze_model(model, dtype)

        # Basic estimate: ~2x model size per batch sample
        activation_memory_mb = model_info.model_size_mb * 2 * batch_size

        # Add sequence length factor for transformer models
        if effective_seq_length:
            activation_memory_mb *= (
                effective_seq_length / 512
            )  # Normalize by typical seq len

        return activation_memory_mb

    def get_model_summary(
        self, model_or_info, dtype: torch.dtype = torch.float32
    ) -> dict:
        """
        Get a summary of the model for display.

        Args:
            model_or_info: PyTorch model or ModelInfo object
            dtype: Data type (used only if model is provided)

        Returns:
            Dictionary with formatted model information
        """
        if isinstance(model_or_info, ModelInfo):
            model_info = model_or_info
        else:
            model_info = self.analyze_model(model_or_info, dtype)

        return {
            "Total Parameters": f"{model_info.total_params:,}",
            "Trainable Parameters": f"{model_info.trainable_params:,}",
            "Model Size": f"{model_info.model_size_mb:.2f} MB",
            "Largest Layer": model_info.largest_layer_name,
            "Largest Layer Params": f"{model_info.largest_layer_params:,}",
        }
