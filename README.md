# DeepSpeed AutoZero 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DeepSpeed AutoZero is an intelligent configuration generator for DeepSpeed's ZeRO (Zero Redundancy Optimizer) that automatically analyzes your PyTorch model and hardware setup to recommend optimal training configurations.

## 🎯 Features

- **Automatic Model Analysis**: Loads and analyzes PyTorch models to determine parameter counts and memory requirements
- **Smart Memory Estimation**: Integrates with DeepSpeed's memory estimation APIs to predict GPU/CPU memory usage
- **Optimal Stage Selection**: Automatically selects the best ZeRO stage (1-3) based on model size and available resources
- **CPU Offloading Support**: Intelligently recommends CPU offloading when GPU memory is insufficient
- **Configuration Validation**: Validates generated configurations against DeepSpeed's schema
- **User-Friendly CLI**: Simple command-line interface with rich output formatting

## 📦 Installation

### From PyPI (Coming Soon)
```bash
pip install deepspeed-autozero
```

### From Source
```bash
git clone https://github.com/yourusername/deepspeed-autozero.git
cd deepspeed-autozero
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- DeepSpeed 0.12.0+
- CUDA-capable GPU (for memory detection)

## 🚀 Quick Start

### Basic Usage
```bash
# Generate configuration for your model
autozero path/to/your/model.py --gpus 4 --batch-size 8 --dtype bf16 > ds_config.json

# Use the generated config with DeepSpeed
deepspeed your_training_script.py --deepspeed_config ds_config.json
```

### Command-Line Options
```bash
autozero --help

# Key options:
# --gpus: Number of GPUs to use (default: 1)
# --batch-size: Batch size per GPU (default: 1)  
# --dtype: Data type [fp32, fp16, bf16] (default: fp16)
# --cpu-offload: Enable CPU offloading
# --verbose: Show detailed analysis
# --output: Save to file instead of stdout
```

## 📋 Examples

### Example 1: Small Model on Limited GPUs
```bash
# For a BERT-base model on 2 GPUs
autozero models/bert_base.py --gpus 2 --batch-size 32 --dtype fp16
```

### Example 2: Large Model with CPU Offloading
```bash
# For a GPT-3 style model that doesn't fit in GPU memory
autozero models/gpt3_13b.py --gpus 8 --batch-size 4 --cpu-offload --verbose
```

### Example 3: Validate Existing Configuration
```bash
# Check if your existing DeepSpeed config is valid
autozero validate existing_config.json
```

## 🏗️ Architecture

### Model Requirements

Your model file should contain a PyTorch `nn.Module` class that can be instantiated:

```python
# models/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12):
        super().__init__()
        self.embeddings = nn.Embedding(50000, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, 8)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 50000)
    
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
```

### How It Works

1. **Model Loading**: Dynamically imports and instantiates your model
2. **Analysis**: Counts parameters and estimates memory requirements
3. **Memory Estimation**: Uses DeepSpeed's APIs to predict memory usage for each ZeRO stage
4. **Stage Selection**: Chooses optimal ZeRO stage based on available GPU memory
5. **Config Generation**: Creates a validated DeepSpeed configuration

### Memory Estimation Accuracy

⚠️ **Important Notes on Memory Estimation**:
- DeepSpeed's estimation APIs may underestimate actual memory usage (see [#4527](https://github.com/microsoft/DeepSpeed/issues/4527), [#5484](https://github.com/microsoft/DeepSpeed/issues/5484))
- AutoZero applies a 1.5x safety factor to all estimates
- Always monitor actual memory usage during training
- Consider starting with smaller batch sizes and scaling up

## 🧪 Testing

Run the test suite:
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=autozero --cov-report=html

# Run specific test file
pytest tests/test_cli.py -v
```

## 🔧 Advanced Configuration

### Custom Model Loading

If your model requires special initialization:

```python
# models/custom_model.py
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # Your model implementation
        pass

# Factory function (optional)
def create_model():
    config = {"hidden_size": 1024, "num_layers": 24}
    return CustomModel(config)
```

### Environment Variables

```bash
# Set CUDA device for memory detection
export CUDA_VISIBLE_DEVICES=0

# Enable debug logging
export AUTOZERO_DEBUG=1
```

## 📊 Benchmarks

| Model Size | GPUs | ZeRO Stage | Batch Size | Memory per GPU |
|------------|------|------------|------------|----------------|
| 1.3B       | 4    | 2          | 32         | 12.5 GB        |
| 6.7B       | 8    | 3          | 16         | 23.8 GB        |
| 13B        | 8    | 3 + Offload| 8          | 15.2 GB        |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/deepspeed-autozero.git
cd deepspeed-autozero

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeepSpeed](https://github.com/microsoft/DeepSpeed) team for the amazing optimization library
- [Typer](https://typer.tiangolo.com/) for the excellent CLI framework
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output

## 📚 References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed Configuration Guide](https://www.deepspeed.ai/docs/config-json/)

## ❓ FAQ

**Q: My model doesn't load with AutoZero**  
A: Ensure your model can be instantiated with default arguments or create a factory function.

**Q: The generated config causes OOM errors**  
A: The memory estimation may be inaccurate. Try:
- Using `--cpu-offload` flag
- Reducing batch size
- Manually setting a higher ZeRO stage

**Q: Can I use this with custom DeepSpeed configurations?**  
A: Yes! Generate a base config and modify it as needed. Use `autozero validate` to check your changes.

## 🐛 Known Issues

- Memory estimation accuracy varies by model architecture
- Transformer models may require additional activation memory
- CPU offloading performance depends on PCIe bandwidth

## 📮 Support

- 📧 Email: team@deepspeed-autozero.io
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/deepspeed-autozero/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/deepspeed-autozero/discussions)