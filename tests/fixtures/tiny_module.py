"""
Tiny test models for unit testing.
"""

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    """A minimal model for testing."""
    
    def __init__(self, hidden_size=128, num_layers=2, vocab_size=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.ReLU()
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output(x)


class TransformerTinyModel(nn.Module):
    """A tiny transformer model for testing."""
    
    def __init__(self, d_model=256, nhead=8, num_layers=2, vocab_size=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, src):
        src = self.embedding(src) * (self.d_model ** 0.5)
        seq_len = src.size(1)
        src = src + self.pos_encoder[:, :seq_len, :]
        output = self.transformer(src)
        return self.output(output)


class InvalidModel:
    """Not a nn.Module - for testing error handling."""
    pass


def model_factory():
    """Factory function that returns a model."""
    return TinyModel(hidden_size=64, num_layers=1)