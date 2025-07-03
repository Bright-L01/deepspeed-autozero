"""
Tests for memory estimator module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn

from autozero.memory_estimator import MemoryEstimator, MemoryEstimate


@pytest.fixture
def mock_deepspeed():
    """Mock DeepSpeed imports."""
    with patch('autozero.memory_estimator.DEEPSPEED_AVAILABLE', True):
        with patch('autozero.memory_estimator.estimate_zero2_model_states_mem_needs_all_live') as mock_zero2:
            with patch('autozero.memory_estimator.estimate_zero3_model_states_mem_needs_all_live') as mock_zero3:
                # Mock the output that would be printed by DeepSpeed
                mock_zero2.side_effect = lambda *args, **kwargs: print(
                    "Estimated memory needed for params, optim states and gradients for a:\n"
                    "HW: Setup with 1 node, 1 GPU per node.\n"
                    "SW: Model with 1000000 total params.\n"
                    "  per CPU  |  per GPU |   Options\n"
                    "   70.00GB |   0.50GB | offload_optimizer=cpu\n"
                    "   70.00GB |   2.00GB | offload_optimizer=none\n"
                )
                
                mock_zero3.side_effect = lambda *args, **kwargs: print(
                    "Estimated memory needed for params, optim states and gradients for a:\n"
                    "HW: Setup with 1 node, 1 GPU per node.\n"
                    "SW: Model with 1000000 total params, 100000 largest layer params.\n"
                    "  per CPU  |  per GPU |   Options\n"  
                    "  100.00GB |   0.30GB | offload_param=cpu, offload_optimizer=cpu\n"
                    "   70.00GB |   0.80GB | offload_param=none, offload_optimizer=cpu\n"
                    "    0.00GB |   1.50GB | offload_param=none, offload_optimizer=none\n"
                )
                
                yield mock_zero2, mock_zero3


@pytest.fixture
def tiny_model():
    """Create a tiny test model."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    return TinyModel()


@pytest.fixture
def mock_cuda():
    """Mock CUDA availability."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_device = MagicMock()
            mock_device.total_memory = 16 * 1024**3  # 16 GB
            mock_props.return_value = mock_device
            yield


class TestMemoryEstimator:
    """Test cases for MemoryEstimator."""
    
    def test_init_without_deepspeed(self):
        """Test initialization when DeepSpeed is not available."""
        with patch('autozero.memory_estimator.DEEPSPEED_AVAILABLE', False):
            with pytest.raises(ImportError, match="DeepSpeed is not installed"):
                MemoryEstimator()
    
    def test_get_gpu_memory(self, mock_cuda):
        """Test GPU memory detection."""
        with patch('autozero.memory_estimator.DEEPSPEED_AVAILABLE', True):
            estimator = MemoryEstimator()
            memory_gb = estimator.get_gpu_memory_gb()
            
            # Should be 90% of 16GB
            assert memory_gb == pytest.approx(16 * 0.9, rel=0.01)
    
    def test_get_gpu_memory_no_cuda(self):
        """Test GPU memory when CUDA is not available."""
        with patch('autozero.memory_estimator.DEEPSPEED_AVAILABLE', True):
            with patch('torch.cuda.is_available', return_value=False):
                estimator = MemoryEstimator()
                memory_gb = estimator.get_gpu_memory_gb()
                assert memory_gb == 0.0
    
    def test_parse_deepspeed_output(self):
        """Test parsing DeepSpeed output."""
        with patch('autozero.memory_estimator.DEEPSPEED_AVAILABLE', True):
            estimator = MemoryEstimator()
            
            output = """
            Estimated memory needed for params, optim states and gradients for a:
            HW: Setup with 1 node, 4 GPUs per node.
            SW: Model with 1000000 total params.
              per CPU  |  per GPU |   Options
               10.50GB |   2.25GB | offload_optimizer=cpu
                0.00GB |   4.50GB | offload_optimizer=none
            """
            
            parsed = estimator._parse_deepspeed_output(output)
            
            assert parsed == 10.5  # Should find first number value found by regex
    
    def test_estimate_zero2_memory(self, mock_deepspeed, tiny_model, mock_cuda):
        """Test ZeRO-2 memory estimation."""
        estimator = MemoryEstimator()
        result = estimator.estimate_zero2_memory(tiny_model, num_gpus=1)
        
        assert isinstance(result, MemoryEstimate)
        assert result.stage == 2
        assert result.gpu_memory_gb > 0
        assert result.can_fit_gpu is not None
        assert len(result.notes) > 0
    
    def test_estimate_zero3_memory(self, mock_deepspeed, tiny_model, mock_cuda):
        """Test ZeRO-3 memory estimation."""
        estimator = MemoryEstimator()
        result = estimator.estimate_zero3_memory(tiny_model, num_gpus=1)
        
        assert isinstance(result, MemoryEstimate)
        assert result.stage == 3
        assert result.gpu_memory_gb > 0
        assert result.can_fit_gpu is not None
        assert len(result.notes) > 0
    
    def test_estimate_with_cpu_offload(self, mock_deepspeed, tiny_model, mock_cuda):
        """Test memory estimation with CPU offload."""
        estimator = MemoryEstimator()
        
        # Test ZeRO-2 with offload
        result_2 = estimator.estimate_zero2_memory(
            tiny_model, num_gpus=1, cpu_offload=True
        )
        assert result_2.requires_cpu_offload is True
        assert result_2.cpu_memory_gb >= 0
        
        # Test ZeRO-3 with offload
        result_3 = estimator.estimate_zero3_memory(
            tiny_model, num_gpus=1, cpu_offload=True
        )
        assert result_3.requires_cpu_offload is True
        assert result_3.cpu_memory_gb >= 0
    
    def test_select_optimal_stage(self, mock_deepspeed, tiny_model, mock_cuda):
        """Test optimal stage selection."""
        estimator = MemoryEstimator()
        
        stage, offload, estimate = estimator.select_optimal_stage(
            tiny_model, num_gpus=1, batch_size=8
        )
        
        assert stage in [2, 3]
        assert isinstance(offload, bool)
        assert isinstance(estimate, MemoryEstimate)
    
    def test_select_optimal_stage_prefer_memory(self, mock_deepspeed, tiny_model, mock_cuda):
        """Test stage selection preferring memory efficiency."""
        estimator = MemoryEstimator()
        
        stage, offload, estimate = estimator.select_optimal_stage(
            tiny_model, num_gpus=1, batch_size=8, prefer_speed=False
        )
        
        # Should prefer higher stages when optimizing for memory
        assert stage in [2, 3]
    
    def test_select_optimal_stage_no_viable_config(self, mock_deepspeed, tiny_model):
        """Test when no viable configuration is found."""
        # Mock very small GPU memory
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_device = MagicMock()
                mock_device.total_memory = 1 * 1024**3  # 1 GB (very small)
                mock_props.return_value = mock_device
                
                # Mock the estimation to fail  
                with patch.object(MemoryEstimator, 'estimate_stage_memory') as mock_est:
                    mock_est.side_effect = Exception("Estimation failed")
                    
                    estimator = MemoryEstimator()
                    stage, offload, estimate = estimator.select_optimal_stage(tiny_model, num_gpus=1)
                    
                    # Should return conservative fallback
                    assert stage == 3
                    assert offload is True
                    assert "Could not determine optimal configuration" in estimate.warnings
    
    def test_safety_factor_large_model(self):
        """Test safety factor calculation for large models."""
        estimator = MemoryEstimator(require_deepspeed=False)
        
        # Test >10B parameter model
        factor = estimator._calculate_evidence_based_safety_factor(
            stage=2, model_params=15_000_000_000, has_offload=False
        )
        expected = 4.0 + 2.0  # base stage 2 factor + large model adjustment
        assert factor == expected
        
        # Test >1B parameter model 
        factor = estimator._calculate_evidence_based_safety_factor(
            stage=2, model_params=2_000_000_000, has_offload=False
        )
        expected = 4.0 + 1.0  # base stage 2 factor + medium model adjustment
        assert factor == expected
        
        # Test stage 3 with offload (extreme case)
        factor = estimator._calculate_evidence_based_safety_factor(
            stage=3, model_params=100_000_000, has_offload=True
        )
        assert factor == 10.0  # Conservative for critical offload scenarios
    
    def test_parse_deepspeed_output_patterns(self):
        """Test parsing various DeepSpeed output patterns."""
        estimator = MemoryEstimator(require_deepspeed=False)
        
        # Test specific pattern matches
        output1 = "GPU memory per GPU = 15.23 GB"
        assert estimator._parse_deepspeed_output(output1) == 15.23
        
        # Test fallback to any GB value
        output2 = "Some other text with 8.45 GB mentioned"
        assert estimator._parse_deepspeed_output(output2) == 8.45
        
        # Test no match returns 0.0
        output3 = "No memory information here"
        assert estimator._parse_deepspeed_output(output3) == 0.0
    
    def test_estimate_stage_memory_with_exception(self):
        """Test estimate_stage_memory when DeepSpeed estimation fails."""
        model = nn.Linear(1000, 1000)  # Larger model for testing
        estimator = MemoryEstimator(require_deepspeed=False)
        
        # Mock the DeepSpeed functions to raise an exception
        with patch('autozero.memory_estimator.estimate_zero2_model_states_mem_needs_all_live') as mock_zero2:
            mock_zero2.side_effect = Exception("Mock DeepSpeed failure")
            
            estimate = estimator.estimate_stage_memory(model, stage=2, num_gpus=1)
            
            assert any("DeepSpeed estimation failed" in warning for warning in estimate.warnings)
            assert estimate.gpu_memory_gb > 0
            assert estimate.stage == 2
    
    def test_estimate_stage_memory_invalid_stage(self):
        """Test estimate_stage_memory with invalid stage."""
        model = nn.Linear(10, 1)
        estimator = MemoryEstimator(require_deepspeed=False)
        
        # Mock the DeepSpeed functions to ensure we test the exception path
        with patch('autozero.memory_estimator.estimate_zero2_model_states_mem_needs_all_live') as mock_zero2:
            with patch('autozero.memory_estimator.estimate_zero3_model_states_mem_needs_all_live') as mock_zero3:
                mock_zero2.side_effect = Exception("Mock failure")
                mock_zero3.side_effect = Exception("Mock failure")
                
                estimate = estimator.estimate_stage_memory(model, stage=5, num_gpus=1)
                
                assert any("DeepSpeed estimation failed" in warning for warning in estimate.warnings)
                assert "Unsupported stage: 5" in str(estimate.warnings)
    
    def test_large_model_warnings(self):
        """Test warnings for large models."""
        # Create a large model (>10B params)
        large_model = nn.Linear(100000, 100000)  # ~10B params
        estimator = MemoryEstimator(require_deepspeed=False)
        
        estimate = estimator.estimate_stage_memory(large_model, stage=3, num_gpus=1)
        
        # Should include large model warning
        warning_found = any("Large models (>10B params)" in warning for warning in estimate.warnings)
        assert warning_found
    
    def test_stage3_without_offload_warning(self):
        """Test warning for Stage 3 without offloading."""
        model = nn.Linear(10, 1)
        estimator = MemoryEstimator(require_deepspeed=False)
        
        estimate = estimator.estimate_stage_memory(model, stage=3, num_gpus=1, has_offload=False)
        
        # Should include Stage 3 warning
        warning_found = any("Stage 3 without offloading may use 8-10x" in warning for warning in estimate.warnings)
        assert warning_found
    
    def test_select_optimal_stage_verbose_error(self):
        """Test verbose error output in select_optimal_stage."""
        model = nn.Linear(10, 1)
        estimator = MemoryEstimator(require_deepspeed=False, verbose=True)
        
        # Mock estimate_stage_memory to raise an exception
        def mock_estimate(*args, **kwargs):
            raise Exception("Mock estimation failure")
        
        with patch.object(estimator, 'estimate_stage_memory', side_effect=mock_estimate):
            # Capture print output
            from io import StringIO
            import sys
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                stage, offload, estimate = estimator.select_optimal_stage(model)
                
                # Restore stdout
                sys.stdout = sys.__stdout__
                output = captured_output.getvalue()
                
                # Should have printed verbose error message
                assert "Failed to estimate stage" in output
                assert "Mock estimation failure" in output
                
            finally:
                sys.stdout = sys.__stdout__