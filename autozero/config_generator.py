"""
Configuration generator for DeepSpeed AutoZero.

Single responsibility: Generate and validate DeepSpeed JSON configurations
based on model analysis and memory estimation results.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from .model_analyzer import ModelInfo


class ConfigGenerator:
    """Generates DeepSpeed configurations with evidence-based parameters."""

    def __init__(self):
        # Default configuration baseline
        self.base_config = {
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
            "wall_clock_breakdown": False,
        }

    def generate_config(
        self,
        model_info: ModelInfo,
        memory_estimate: Any = None,  # For backward compatibility
        stage: int = None,
        batch_size: int = None,
        num_gpus: int = None,
        dtype: str = "fp16",
        cpu_offload: bool = False,
        additional_options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate DeepSpeed configuration.

        Args:
            model_info: Model analysis results
            memory_estimate: Memory estimate (if provided, stage is extracted from it)
            stage: ZeRO optimization stage
            batch_size: Batch size per GPU
            num_gpus: Number of GPUs
            dtype: Data type (fp16, bf16, fp32)
            cpu_offload: Whether to enable CPU offloading

        Returns:
            DeepSpeed configuration dictionary
        """
        # Handle backward compatibility
        if memory_estimate is not None:
            stage = memory_estimate.stage
            cpu_offload = memory_estimate.requires_cpu_offload
        config = self.base_config.copy()

        # Basic training configuration
        config["train_batch_size"] = batch_size * num_gpus
        config["train_micro_batch_size_per_gpu"] = batch_size

        # Data type configuration - always include both for backward compatibility
        config["fp16"] = {
            "enabled": dtype == "fp16",
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
        config["bf16"] = {"enabled": dtype == "bf16"}

        # ZeRO optimization configuration
        zero_config = {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
        }

        # Stage-specific configurations
        if stage == 2:
            # Stage 2 configuration
            zero_config.update(
                {
                    "reduce_scatter": True,
                    "reduce_bucket_size": min(5e8, model_info.total_params * 4),
                    "allgather_bucket_size": min(5e8, model_info.total_params * 4),
                }
            )

            # Always include offload_optimizer for backward compatibility
            zero_config["offload_optimizer"] = {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True if cpu_offload else False,
                "buffer_count": 4 if cpu_offload else 1,
                "fast_init": False,
            }

        elif stage == 3:
            # Stage 3 configuration with evidence-based parameters
            zero_config.update(
                {
                    "stage3_max_live_parameters": min(
                        1e9, model_info.total_params // 10
                    ),
                    "stage3_max_reuse_distance": min(
                        1e9, model_info.total_params // 10
                    ),
                    "stage3_prefetch_bucket_size": min(
                        5e8, model_info.total_params * 0.9
                    ),
                    "stage3_param_persistence_threshold": min(
                        1e5, model_info.total_params // 100
                    ),
                    "stage3_gather_16bit_weights_on_model_save": True,
                }
            )

            # Always include offload configurations
            zero_config["offload_optimizer"] = {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True if cpu_offload else False,
                "buffer_count": 4 if cpu_offload else 1,
                "fast_init": False,
            }
            zero_config["offload_param"] = {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True if cpu_offload else False,
                "buffer_count": 5 if cpu_offload else 1,
            }

        config["zero_optimization"] = zero_config

        # Apply additional options if provided
        if additional_options:
            config = self.deep_merge(config, additional_options)

        return config

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate DeepSpeed configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        if "train_batch_size" not in config:
            errors.append("Missing required field: train_batch_size")

        if "train_micro_batch_size_per_gpu" not in config:
            errors.append("Missing required field: train_micro_batch_size_per_gpu")

        # Validate batch sizes
        if "train_batch_size" in config and "train_micro_batch_size_per_gpu" in config:
            total_batch = config["train_batch_size"]
            micro_batch = config["train_micro_batch_size_per_gpu"]

            if total_batch % micro_batch != 0:
                errors.append(
                    f"train_batch_size ({total_batch}) must be divisible by "
                    f"train_micro_batch_size_per_gpu ({micro_batch})"
                )

        # Validate ZeRO configuration
        if "zero_optimization" in config:
            zero_config = config["zero_optimization"]

            if "stage" not in zero_config:
                errors.append("Missing ZeRO stage in zero_optimization")
            else:
                stage = zero_config["stage"]
                if stage not in [0, 1, 2, 3]:
                    errors.append(f"Invalid ZeRO stage: {stage}")

                # Stage 3 specific validation
                if stage == 3 and "offload_param" not in zero_config:
                    errors.append("Stage 3 should have offload_param configuration")

            # Validate offload configurations
            for offload_key in ["offload_optimizer", "offload_param"]:
                if offload_key in zero_config:
                    offload_config = zero_config[offload_key]
                    if (
                        isinstance(offload_config, dict)
                        and offload_config.get("device") == "nvme"
                    ):
                        if "nvme_path" not in offload_config:
                            errors.append(
                                f"nvme_path required when using nvme device in {offload_key}"
                            )

        # Validate data type configuration
        fp16_enabled = config.get("fp16", {}).get("enabled", False)
        bf16_enabled = config.get("bf16", {}).get("enabled", False)

        if fp16_enabled and bf16_enabled:
            errors.append("Cannot enable both fp16 and bf16")

        return errors

    def config_to_json(self, config: Dict[str, Any]) -> str:
        """
        Convert configuration to JSON string.

        Args:
            config: Configuration dictionary

        Returns:
            Pretty-formatted JSON string
        """
        # Remove internal fields (those starting with _)
        cleaned_config = {k: v for k, v in config.items() if not k.startswith("_")}
        return json.dumps(cleaned_config, indent=2, sort_keys=False)

    def save_config(self, config: Dict[str, Any], path: Path) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            path: Path to save to
        """
        json_str = self.config_to_json(config)
        path.write_text(json_str)

    def load_config(self, path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            path: Path to load from

        Returns:
            Configuration dictionary
        """
        return json.loads(path.read_text())

    def deep_merge(
        self, base: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries in place.

        Args:
            base: Base dictionary (modified in place)
            update: Dictionary to merge into base
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
