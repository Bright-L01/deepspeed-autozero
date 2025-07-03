"""
Simple test model for DeepSpeed AutoZero testing.
"""

import torch
import torch.nn as nn


class SimpleTestModel(nn.Module):
    """A simple model for testing purposes."""
    
    def __init__(self, hidden_size=512, num_layers=6, vocab_size=10000):
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