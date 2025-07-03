"""
Essential utility functions for DeepSpeed AutoZero.

Simplified for Phase 2 - contains only functions used by other modules.
"""

import torch
from pathlib import Path


def get_dtype_from_string(dtype_str: str) -> torch.dtype:
    """
    Convert string representation to torch dtype.

    Args:
        dtype_str: String representation of dtype

    Returns:
        torch.dtype object

    Raises:
        ValueError: If dtype string is not supported
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    dtype_lower = dtype_str.lower()
    if dtype_lower not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. "
            f"Supported types: {', '.join(dtype_map.keys())}"
        )

    return dtype_map[dtype_lower]


def validate_model_file(model_path: Path) -> None:
    """
    Validate that a model file exists and is a Python file.

    Args:
        model_path: Path to model file

    Raises:
        ValueError: If file is invalid
    """
    if not model_path.exists():
        raise ValueError(f"Model file does not exist: {model_path}")

    if not model_path.is_file():
        raise ValueError(f"Model path is not a file: {model_path}")

    if model_path.suffix != ".py":
        raise ValueError(
            f"Model file must be a Python file (.py), got: {model_path.suffix}"
        )


def get_system_info() -> dict:
    """
    Get system information including GPU details.

    Returns:
        Dictionary with system information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()

        for i in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {"name": props.name, "memory_gb": props.total_memory / (1024**3)}
            )

    return info
