"""
DeepSpeed AutoZero - Automated DeepSpeed ZeRO configuration generator.

This package provides tools to automatically analyze models and hardware
to generate optimal DeepSpeed ZeRO configurations.
"""

__version__ = "0.1.0"
__author__ = "DeepSpeed AutoZero Team"
__email__ = "team@deepspeed-autozero.io"

from .cli import app

__all__ = ["app"]
