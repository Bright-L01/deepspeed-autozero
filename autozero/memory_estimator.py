"""
Memory estimation module for DeepSpeed AutoZero.

This module implements the core requirement from the design document:
"Python module to call DeepSpeed's memory estimation APIs, apply stage-specific
safety factors, and return a recommended ZeRO stage and offload settings."

Evidence-based safety factors from issues #4527 and #5484.
"""

import io
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Tuple, List

import torch
import torch.nn as nn

try:
    from deepspeed.runtime.zero.stage_1_and_2 import (
        estimate_zero2_model_states_mem_needs_all_live,
    )
    from deepspeed.runtime.zero.stage3 import (
        estimate_zero3_model_states_mem_needs_all_live,
    )

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

    # Provide fallback for testing
    def estimate_zero2_model_states_mem_needs_all_live(*_, **__):
        raise ImportError("DeepSpeed is not installed")

    def estimate_zero3_model_states_mem_needs_all_live(*_, **__):
        raise ImportError("DeepSpeed is not installed")


@dataclass
class MemoryEstimate:
    """Container for memory estimation results."""

    stage: int
    gpu_memory_gb: float
    cpu_memory_gb: float
    warnings: List[str]
    # Additional fields for backward compatibility
    total_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    parameter_memory_gb: float = 0.0
    can_fit_gpu: bool = True
    requires_cpu_offload: bool = False
    notes: List[str] = field(default_factory=list)


class MemoryEstimator:
    """
    Memory estimator with evidence-based safety factors.

    Safety factors derived from real DeepSpeed issues:
    - Issue #4527: 43x underestimation (0.49GB → 21GB actual)
    - Issue #5484: 1.8x underestimation (17GB → 30GB actual)
    """

    # Evidence-based safety factors from DeepSpeed issues
    SAFETY_FACTORS = {
        0: 1.5,  # Baseline - no ZeRO
        1: 2.0,  # Stage 1 - conservative
        2: 4.0,  # Stage 2 - moderate underestimation observed
        3: 8.0,  # Stage 3 - severe underestimation (issue #4527)
    }

    # PyTorch reserved vs allocated memory overhead (issue #4527)
    PYTORCH_OVERHEAD = 2.5

    # Base initialization overhead
    INIT_OVERHEAD_GB = 2.0

    def __init__(self, verbose: bool = False, require_deepspeed: bool = True):
        self.verbose = verbose
        self.require_deepspeed = require_deepspeed
        if require_deepspeed and not DEEPSPEED_AVAILABLE:
            raise ImportError(
                "DeepSpeed is not installed. Please install with: pip install deepspeed"
            )

    def get_gpu_memory_gb(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0

        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        # Reserve 10% for CUDA context
        return total_memory_gb * 0.9

    def _calculate_evidence_based_safety_factor(
        self, stage: int, model_params: int, has_offload: bool
    ) -> float:
        """
        Calculate safety factor based on evidence from DeepSpeed issues.

        Args:
            stage: ZeRO stage (0-3)
            model_params: Total model parameters
            has_offload: Whether CPU offloading is enabled

        Returns:
            Safety factor to apply to DeepSpeed estimates
        """
        base_factor = self.SAFETY_FACTORS.get(stage, 2.0)

        # Issue #4527: Stage 3 with offloading shows extreme underestimation
        if stage == 3 and has_offload:
            return 10.0  # Conservative for critical offload scenarios

        # Issue #5484: Large models (>10B params) show additional underestimation
        if model_params > 10_000_000_000:  # >10B parameters
            base_factor += 2.0
        elif model_params > 1_000_000_000:  # >1B parameters
            base_factor += 1.0

        return min(base_factor, 15.0)  # Cap at 15x for safety

    def _parse_deepspeed_output(self, output: str) -> float:
        """
        Parse GPU memory requirement from DeepSpeed estimation output.

        Args:
            output: String output from DeepSpeed estimation functions

        Returns:
            GPU memory requirement in GB, or 0.0 if not found
        """
        # Look for GPU memory requirement in output
        patterns = [
            r"per GPU\s*=\s*([\d.]+)\s*GB",
            r"GPU memory.*?=\s*([\d.]+)\s*GB",
            r"params.*?=\s*([\d.]+)\s*GB",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # If no specific GPU memory found, look for any memory value
        numbers = re.findall(r"([\d.]+)\s*GB", output)
        if numbers:
            return float(numbers[0])  # Use first number found

        return 0.0

    def estimate_stage_memory(
        self, model: nn.Module, stage: int, num_gpus: int = 1, has_offload: bool = False
    ) -> MemoryEstimate:
        """
        Estimate memory for a specific ZeRO stage.

        Args:
            model: PyTorch model
            stage: ZeRO stage (2 or 3)
            num_gpus: Number of GPUs
            has_offload: Whether CPU offloading is enabled

        Returns:
            MemoryEstimate with corrected values
        """
        warnings = []

        # Capture DeepSpeed output
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                if stage == 2:
                    estimate_zero2_model_states_mem_needs_all_live(
                        model, num_gpus_per_node=num_gpus, num_nodes=1
                    )
                elif stage == 3:
                    estimate_zero3_model_states_mem_needs_all_live(
                        model, num_gpus_per_node=num_gpus, num_nodes=1
                    )
                else:
                    raise ValueError(f"Unsupported stage: {stage}")
        except Exception as e:
            warnings.append(f"DeepSpeed estimation failed: {str(e)}")
            # Provide fallback estimate
            total_params = sum(p.numel() for p in model.parameters())
            base_estimate = (total_params * 4) / (1024**3)  # fp32 baseline
        else:
            output = buffer.getvalue()
            if self.verbose:
                print(f"DeepSpeed Stage {stage} Output:")
                print(output)
                print("-" * 40)

            base_estimate = self._parse_deepspeed_output(output)
            if base_estimate == 0.0:
                warnings.append("Could not parse DeepSpeed output")
                # Fallback estimate
                total_params = sum(p.numel() for p in model.parameters())
                base_estimate = (total_params * 4) / (1024**3)

        # Apply evidence-based corrections
        total_params = sum(p.numel() for p in model.parameters())
        safety_factor = self._calculate_evidence_based_safety_factor(
            stage, total_params, has_offload
        )

        # Apply corrections:
        # 1. Safety factor for known underestimation
        corrected_gpu_memory = base_estimate * safety_factor

        # 2. PyTorch overhead (issue #4527)
        corrected_gpu_memory *= self.PYTORCH_OVERHEAD

        # 3. Initialization overhead
        corrected_gpu_memory += self.INIT_OVERHEAD_GB

        # CPU memory for offloading
        cpu_memory = 0.0
        if has_offload:
            # Rough estimate: 2x model size for CPU offloading
            model_size_gb = (total_params * 4) / (1024**3)  # fp32
            cpu_memory = model_size_gb * 2

        # Add warnings based on evidence
        if stage == 3 and not has_offload:
            warnings.append(
                "Stage 3 without offloading may use 8-10x more memory than estimated "
                "(based on DeepSpeed issue #4527)"
            )

        if total_params > 10_000_000_000:
            warnings.append(
                "Large models (>10B params) often exceed memory estimates "
                "(based on DeepSpeed issue #5484)"
            )

        warnings.append(
            f"Applied {safety_factor:.1f}x safety factor based on reported issues"
        )

        # Calculate component breakdown (estimates)
        model_size_gb = (total_params * 4) / (1024**3)
        parameter_memory = model_size_gb / num_gpus if stage >= 2 else model_size_gb
        optimizer_memory = parameter_memory * 2  # Adam optimizer states
        gradient_memory = parameter_memory

        return MemoryEstimate(
            stage=stage,
            gpu_memory_gb=corrected_gpu_memory,
            cpu_memory_gb=cpu_memory,
            warnings=warnings,
            total_memory_gb=corrected_gpu_memory + cpu_memory,
            optimizer_memory_gb=optimizer_memory,
            gradient_memory_gb=gradient_memory,
            parameter_memory_gb=parameter_memory,
            can_fit_gpu=True,  # Will be updated by caller
            requires_cpu_offload=has_offload,
            notes=warnings.copy(),
        )

    def select_optimal_stage(
        self,
        model: nn.Module,
        num_gpus: int = 1,
        batch_size: int = 1,
        force_offload: bool = False,
        prefer_speed: bool = True,
    ) -> Tuple[int, bool, MemoryEstimate]:
        """
        Select optimal ZeRO stage and offload settings.

        Args:
            model: PyTorch model
            num_gpus: Number of GPUs
            batch_size: Batch size per GPU
            force_offload: Force CPU offloading

        Returns:
            Tuple of (recommended_stage, needs_offload, memory_estimate)
        """
        available_gpu_memory = self.get_gpu_memory_gb()

        if available_gpu_memory == 0:
            # No GPU available, force CPU offloading
            estimate = self.estimate_stage_memory(model, 3, num_gpus, has_offload=True)
            return 3, True, estimate

        # Try stages in order of preference
        stages = [2, 3] if prefer_speed else [3, 2]
        for stage in stages:
            for offload in [False, True]:
                if offload and not force_offload and stage == 2:
                    continue  # Skip offload for stage 2 unless forced

                try:
                    estimate = self.estimate_stage_memory(
                        model, stage, num_gpus, has_offload=offload
                    )

                    # Check if it fits
                    if estimate.gpu_memory_gb <= available_gpu_memory or offload:
                        estimate.can_fit_gpu = (
                            estimate.gpu_memory_gb <= available_gpu_memory
                        )
                        return stage, offload, estimate

                except Exception as e:
                    if self.verbose:
                        print(f"Failed to estimate stage {stage}: {e}")
                    continue

        # If nothing works, return conservative estimate
        total_params = sum(p.numel() for p in model.parameters())
        fallback_estimate = MemoryEstimate(
            stage=3,
            gpu_memory_gb=available_gpu_memory * 0.8,  # Use 80% of available
            cpu_memory_gb=(total_params * 4) / (1024**3) * 2,  # 2x model size
            warnings=[
                "Could not determine optimal configuration",
                "Using conservative fallback with CPU offloading",
            ],
        )

        return 3, True, fallback_estimate

    def estimate_zero2_memory(
        self,
        model: nn.Module,
        num_gpus: int = 1,
        batch_size: int = 1,
        cpu_offload: bool = False,
    ) -> MemoryEstimate:
        """
        Estimate memory for ZeRO Stage 2.

        Args:
            model: PyTorch model
            num_gpus: Number of GPUs
            batch_size: Batch size per GPU
            cpu_offload: Whether CPU offloading is enabled

        Returns:
            MemoryEstimate for Stage 2
        """
        return self.estimate_stage_memory(model, 2, num_gpus, cpu_offload)

    def estimate_zero3_memory(
        self,
        model: nn.Module,
        num_gpus: int = 1,
        batch_size: int = 1,
        cpu_offload: bool = False,
    ) -> MemoryEstimate:
        """
        Estimate memory for ZeRO Stage 3.

        Args:
            model: PyTorch model
            num_gpus: Number of GPUs
            batch_size: Batch size per GPU
            cpu_offload: Whether CPU offloading is enabled

        Returns:
            MemoryEstimate for Stage 3
        """
        return self.estimate_stage_memory(model, 3, num_gpus, cpu_offload)
