"""
Tests for utils module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from autozero.utils import (
    get_dtype_from_string,
    validate_model_file
)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_dtype_from_string(self):
        """Test dtype string conversion."""
        assert get_dtype_from_string("fp32") == torch.float32
        assert get_dtype_from_string("fp16") == torch.float16
        assert get_dtype_from_string("bf16") == torch.bfloat16
        assert get_dtype_from_string("float32") == torch.float32
        assert get_dtype_from_string("FP32") == torch.float32  # Case insensitive
        
        with pytest.raises(ValueError, match="Unknown dtype"):
            get_dtype_from_string("invalid")
    
    def test_validate_model_file_valid(self, tmp_path):
        """Test model file validation with valid file."""
        model_file = tmp_path / "model.py"
        model_file.write_text("# Model file")
        
        # Should not raise
        validate_model_file(model_file)
    
    def test_validate_model_file_not_exists(self):
        """Test model file validation with non-existent file."""
        with pytest.raises(ValueError, match="Model file does not exist"):
            validate_model_file(Path("nonexistent.py"))
    
    def test_validate_model_file_not_file(self, tmp_path):
        """Test model file validation with directory."""
        with pytest.raises(ValueError, match="Model path is not a file"):
            validate_model_file(tmp_path)
    
    def test_validate_model_file_wrong_extension(self, tmp_path):
        """Test model file validation with wrong extension."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("Not a Python file")
        
        with pytest.raises(ValueError, match="Model file must be a Python file"):
            validate_model_file(model_file)
    
    def test_get_system_info_with_cuda(self):
        """Test get_system_info when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=2):
                with patch('torch.version.cuda', '11.8'):
                    with patch('torch.cuda.get_device_properties') as mock_props:
                        # Mock device properties
                        mock_device = MagicMock()
                        mock_device.name = "NVIDIA RTX 3090"
                        mock_device.total_memory = 24 * 1024**3
                        mock_props.return_value = mock_device
                        
                        from autozero.utils import get_system_info
                        info = get_system_info()
                        
                        assert info["cuda_available"] is True
                        assert info["cuda_version"] == '11.8'
                        assert info["gpu_count"] == 2
                        assert len(info["gpus"]) == 2
                        assert info["gpus"][0]["name"] == "NVIDIA RTX 3090"
                        assert info["gpus"][0]["memory_gb"] == 24.0
    
    def test_get_system_info_without_cuda(self):
        """Test get_system_info when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            from autozero.utils import get_system_info
            info = get_system_info()
            
            assert info["cuda_available"] is False
            assert info["cuda_version"] is None
            assert info["gpu_count"] == 0
            assert info["gpus"] == []