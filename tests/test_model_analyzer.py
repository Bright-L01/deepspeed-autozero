"""
Tests for model analyzer module.
"""

import pytest
from pathlib import Path
import torch
import torch.nn as nn

from autozero.model_analyzer import ModelAnalyzer, ModelInfo


@pytest.fixture
def model_analyzer():
    """Create a ModelAnalyzer instance."""
    return ModelAnalyzer()


@pytest.fixture
def tiny_model():
    """Create a tiny test model."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)
            self.embedding = nn.Embedding(100, 10)
        
        def forward(self, x):
            return self.linear2(self.linear1(x))
    
    return TinyModel()


class TestModelAnalyzer:
    """Test cases for ModelAnalyzer."""
    
    def test_load_model_from_file(self, model_analyzer):
        """Test loading a model from a Python file."""
        model_path = Path(__file__).parent / "fixtures" / "tiny_module.py"
        model = model_analyzer.load_model_from_file(model_path)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, "forward")
    
    def test_load_model_file_not_found(self, model_analyzer):
        """Test loading from non-existent file."""
        with pytest.raises(ImportError):
            model_analyzer.load_model_from_file(Path("nonexistent.py"))
    
    def test_load_model_no_nn_module(self, model_analyzer, tmp_path):
        """Test loading file with no nn.Module."""
        # Create a file without nn.Module
        test_file = tmp_path / "no_module.py"
        test_file.write_text("x = 1")
        
        with pytest.raises(ValueError, match="No nn.Module subclass found"):
            model_analyzer.load_model_from_file(test_file)
    
    def test_analyze_model(self, model_analyzer, tiny_model):
        """Test model analysis."""
        info = model_analyzer.analyze_model(tiny_model)
        
        assert isinstance(info, ModelInfo)
        assert info.total_params > 0
        assert info.trainable_params == info.total_params
        assert info.model_size_mb > 0
        assert info.largest_layer_params > 0
        assert len(info.layer_info) > 0
    
    def test_analyze_model_with_dtype(self, model_analyzer, tiny_model):
        """Test model analysis with different dtypes."""
        # Test fp32
        info_fp32 = model_analyzer.analyze_model(tiny_model, dtype=torch.float32)
        
        # Test fp16
        info_fp16 = model_analyzer.analyze_model(tiny_model, dtype=torch.float16)
        
        # Model size should be half for fp16
        assert info_fp16.model_size_mb == pytest.approx(info_fp32.model_size_mb / 2)
    
    def test_estimate_activation_memory(self, model_analyzer, tiny_model):
        """Test activation memory estimation."""
        batch_size = 32
        
        # Test generic model
        mem_mb = model_analyzer.estimate_activation_memory(
            tiny_model, batch_size=batch_size
        )
        assert mem_mb > 0
        
        # Test with sequence length (for transformers)
        mem_mb_seq = model_analyzer.estimate_activation_memory(
            tiny_model, batch_size=batch_size, seq_length=512
        )
        assert mem_mb_seq > 0
    
    def test_estimate_activation_memory_transformer(self, model_analyzer):
        """Test activation memory for transformer model."""
        # Create a simple transformer
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(256, 8)
                self.hidden_size = 256
            
            def forward(self, x):
                return self.attention(x, x, x)[0]
        
        model = SimpleTransformer()
        mem_mb = model_analyzer.estimate_activation_memory(
            model, batch_size=16, seq_length=128
        )
        
        assert mem_mb > 0
    
    def test_get_model_summary(self, model_analyzer, tiny_model):
        """Test model summary generation."""
        info = model_analyzer.analyze_model(tiny_model)
        summary = model_analyzer.get_model_summary(info)
        
        assert isinstance(summary, dict)
        assert "Total Parameters" in summary
        assert "Model Size" in summary
        assert "Largest Layer" in summary
    
    def test_load_model_invalid_spec(self, model_analyzer, tmp_path):
        """Test loading model with invalid module spec."""
        # Create an invalid Python file 
        model_file = tmp_path / "invalid_model.py"
        model_file.write_text("This is not valid Python syntax ][")
        
        with pytest.raises(ImportError, match="Error executing module"):
            model_analyzer.load_model_from_file(model_file)
    
    def test_load_model_with_arguments(self, model_analyzer, tmp_path):
        """Test loading model that requires constructor arguments."""
        # Create a model that requires arguments
        model_file = tmp_path / "model_with_args.py"
        model_file.write_text("""
import torch.nn as nn

class ModelWithArgs(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.heads = nn.Linear(hidden_dim, num_heads)
    
    def forward(self, x):
        return self.heads(self.layers[0](self.embedding(x)))
""")
        
        model = model_analyzer.load_model_from_file(model_file)
        
        # Should successfully instantiate with default args
        assert model is not None
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'layers') 
        assert hasattr(model, 'heads')
    
    def test_load_model_instantiation_failure(self, model_analyzer, tmp_path):
        """Test loading model that cannot be instantiated."""
        # Create a model that cannot be instantiated even with args
        model_file = tmp_path / "bad_model.py"
        model_file.write_text("""
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self, required_arg):
        super().__init__()
        if required_arg != "exact_value":
            raise ValueError("Invalid argument")
        self.layer = nn.Linear(10, 1)
""")
        
        with pytest.raises(ValueError, match="Cannot instantiate BadModel"):
            model_analyzer.load_model_from_file(model_file)
    
    def test_get_model_summary_with_model_direct(self, model_analyzer):
        """Test get_model_summary with model object directly."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(100, 50)
                self.layer2 = nn.Linear(50, 10)
        
        model = TestModel()
        
        # Test with model object directly  
        summary = model_analyzer.get_model_summary(model, torch.float16)
        
        assert "Total Parameters" in summary
        assert "Trainable Parameters" in summary
        assert "Model Size" in summary
        assert "Largest Layer" in summary
        assert "Largest Layer Params" in summary
    
    def test_estimate_activation_memory_sequence_length_alias(self, model_analyzer, tiny_model):
        """Test activation memory estimation with sequence_length parameter."""
        # Test with sequence_length parameter instead of seq_length
        mem_mb = model_analyzer.estimate_activation_memory(
            tiny_model, batch_size=16, sequence_length=256
        )
        assert mem_mb > 0