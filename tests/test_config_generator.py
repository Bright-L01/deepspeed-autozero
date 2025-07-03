"""
Tests for config generator module.
"""

import pytest
import json
from pathlib import Path

from autozero.config_generator import ConfigGenerator
from autozero.model_analyzer import ModelInfo
from autozero.memory_estimator import MemoryEstimate


@pytest.fixture
def config_generator():
    """Create a ConfigGenerator instance."""
    return ConfigGenerator()


@pytest.fixture
def model_info():
    """Create mock model info."""
    return ModelInfo(
        total_params=1000000,
        trainable_params=1000000,
        model_size_mb=4.0,
        largest_layer_params=100000,
        largest_layer_name="output",
        layer_info=[
            {"name": "embedding", "params": 500000},
            {"name": "linear1", "params": 250000},
            {"name": "linear2", "params": 250000},
        ],
        dtype_size=4
    )


@pytest.fixture
def memory_estimate():
    """Create mock memory estimate."""
    return MemoryEstimate(
        stage=2,
        gpu_memory_gb=2.0,
        cpu_memory_gb=0.0,
        warnings=[],  # Required field
        total_memory_gb=2.0,
        optimizer_memory_gb=1.0,
        gradient_memory_gb=0.5,
        parameter_memory_gb=0.5,
        can_fit_gpu=True,
        requires_cpu_offload=False,
        notes=["Test note"]
    )


class TestConfigGenerator:
    """Test cases for ConfigGenerator."""
    
    def test_generate_config_basic(self, config_generator, model_info, memory_estimate):
        """Test basic config generation."""
        config = config_generator.generate_config(
            model_info=model_info,
            memory_estimate=memory_estimate,
            batch_size=8,
            num_gpus=4,
            dtype="fp16",
            stage=2,
            cpu_offload=False
        )
        
        assert config["train_batch_size"] == 32  # 8 * 4
        assert config["train_micro_batch_size_per_gpu"] == 8
        assert config["fp16"]["enabled"] is True
        assert config["bf16"]["enabled"] is False
        assert config["zero_optimization"]["stage"] == 2
        assert config["zero_optimization"]["offload_optimizer"]["device"] == "none"
    
    def test_generate_config_bf16(self, config_generator, model_info, memory_estimate):
        """Test config with bf16."""
        config = config_generator.generate_config(
            model_info=model_info,
            memory_estimate=memory_estimate,
            batch_size=16,
            num_gpus=2,
            dtype="bf16",
            stage=2,
            cpu_offload=False
        )
        
        assert config["fp16"]["enabled"] is False
        assert config["bf16"]["enabled"] is True
    
    def test_generate_config_fp32(self, config_generator, model_info, memory_estimate):
        """Test config with fp32."""
        config = config_generator.generate_config(
            model_info=model_info,
            memory_estimate=memory_estimate,
            batch_size=4,
            num_gpus=1,
            dtype="fp32",
            stage=2,
            cpu_offload=False
        )
        
        assert config["fp16"]["enabled"] is False
        assert config["bf16"]["enabled"] is False
    
    def test_generate_config_stage3(self, config_generator, model_info, memory_estimate):
        """Test config for ZeRO stage 3."""
        memory_estimate.stage = 3
        
        config = config_generator.generate_config(
            model_info=model_info,
            memory_estimate=memory_estimate,
            batch_size=8,
            num_gpus=4,
            dtype="fp16",
            stage=3,
            cpu_offload=False
        )
        
        assert config["zero_optimization"]["stage"] == 3
        assert "stage3_max_live_parameters" in config["zero_optimization"]
        assert "stage3_prefetch_bucket_size" in config["zero_optimization"]
        assert config["zero_optimization"]["offload_param"]["device"] == "none"
    
    def test_generate_config_with_offload(self, config_generator, model_info):
        """Test config with CPU offloading."""
        # Create memory estimate that requires offload
        offload_memory_estimate = MemoryEstimate(
            stage=3,
            gpu_memory_gb=2.0,
            cpu_memory_gb=1.0,
            warnings=[],
            requires_cpu_offload=True  # This is the key difference
        )
        
        config = config_generator.generate_config(
            model_info=model_info,
            memory_estimate=offload_memory_estimate,
            batch_size=8,
            num_gpus=4,
            dtype="fp16",
            stage=3,
            cpu_offload=True
        )
        
        assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
        assert config["zero_optimization"]["offload_param"]["device"] == "cpu"
        assert config["zero_optimization"]["offload_optimizer"]["pin_memory"] is True
    
    def test_generate_config_additional_options(self, config_generator, model_info, memory_estimate):
        """Test config with additional options."""
        additional = {
            "gradient_clipping": 0.5,
            "zero_optimization": {
                "overlap_comm": False
            }
        }
        
        config = config_generator.generate_config(
            model_info=model_info,
            memory_estimate=memory_estimate,
            batch_size=8,
            num_gpus=4,
            dtype="fp16",
            stage=2,
            cpu_offload=False,
            additional_options=additional
        )
        
        assert config["gradient_clipping"] == 0.5
        assert config["zero_optimization"]["overlap_comm"] is False
    
    def test_validate_config_valid(self, config_generator):
        """Test validation of valid config."""
        config = {
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 8,
            "zero_optimization": {
                "stage": 2
            }
        }
        
        errors = config_generator.validate_config(config)
        assert len(errors) == 0
    
    def test_validate_config_invalid_stage(self, config_generator):
        """Test validation with invalid stage."""
        config = {
            "zero_optimization": {
                "stage": 5  # Invalid stage
            }
        }
        
        errors = config_generator.validate_config(config)
        assert len(errors) > 0
    
    def test_validate_config_batch_size_mismatch(self, config_generator):
        """Test validation with batch size mismatch."""
        config = {
            "train_batch_size": 33,
            "train_micro_batch_size_per_gpu": 8  # 33 not divisible by 8
        }
        
        errors = config_generator.validate_config(config)
        assert any("divisible" in error for error in errors)
    
    def test_validate_config_nvme_without_path(self, config_generator):
        """Test validation with NVMe offload but no path."""
        config = {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "nvme"
                    # Missing nvme_path
                }
            }
        }
        
        errors = config_generator.validate_config(config)
        assert any("nvme_path" in error for error in errors)
    
    def test_config_to_json(self, config_generator):
        """Test JSON conversion."""
        config = {
            "train_batch_size": 32,
            "_internal_field": "should be removed",
            "zero_optimization": {"stage": 2}
        }
        
        json_str = config_generator.config_to_json(config)
        parsed = json.loads(json_str)
        
        assert "_internal_field" not in parsed
        assert parsed["train_batch_size"] == 32
    
    def test_save_and_load_config(self, config_generator, tmp_path):
        """Test saving and loading config."""
        config = {
            "train_batch_size": 32,
            "zero_optimization": {"stage": 2}
        }
        
        config_path = tmp_path / "test_config.json"
        config_generator.save_config(config, config_path)
        
        loaded = config_generator.load_config(config_path)
        assert loaded == config
    
    def test_deep_merge(self, config_generator):
        """Test deep merge functionality."""
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2, 3]
        }
        update = {
            "a": 10,
            "b": {"c": 20, "f": 4},
            "g": 5
        }
        
        config_generator._deep_merge(base, update)
        
        assert base["a"] == 10
        assert base["b"]["c"] == 20
        assert base["b"]["d"] == 3
        assert base["b"]["f"] == 4
        assert base["g"] == 5
        assert base["e"] == [1, 2, 3]
    
    def test_validate_config_missing_zero_stage(self, config_generator):
        """Test validation with missing ZeRO stage."""
        config = {
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 8,
            "zero_optimization": {}  # Missing stage
        }
        
        errors = config_generator.validate_config(config)
        assert "Missing ZeRO stage in zero_optimization" in errors
    
    def test_validate_config_stage3_missing_offload_param(self, config_generator):
        """Test validation for Stage 3 missing offload_param."""
        config = {
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 8,
            "zero_optimization": {
                "stage": 3
                # Missing offload_param
            }
        }
        
        errors = config_generator.validate_config(config)
        assert "Stage 3 should have offload_param configuration" in errors
    
    def test_validate_config_both_fp16_and_bf16(self, config_generator):
        """Test validation when both fp16 and bf16 are enabled."""
        config = {
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 8,
            "fp16": {"enabled": True},
            "bf16": {"enabled": True},  # Both enabled - should error
            "zero_optimization": {"stage": 2}
        }
        
        errors = config_generator.validate_config(config)
        assert "Cannot enable both fp16 and bf16" in errors