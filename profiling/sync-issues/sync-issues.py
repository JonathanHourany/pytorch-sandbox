"""
Forcing the Host and Device to synchronize when they otherwise wouldn't have to can be
a real source of inefficiency as processes on both devices must halt and wait for data
to be put on the bus and transmitted. Often, the cause of these syncs can be very
subtle and unintuitive. The easiest place to spot this is calling `.item()` on a tensor
that is in the GPU, but what is often missed is that using a Python object to slice an
object also forces a Host->Device transmission.

@author: Jonathan Hourany
"""

import torch
import torch.nn as nn
import torch.optim as optim
from rich.traceback import install
from torch.profiler import ProfilerActivity, profile, schedule

install(show_locals=True)


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int = 1000, d_model: int = 512, optimized=True):
        """
        Creates a Positional Embedding layer with dimensions (max_seq_length, embed_dim).
        When called, adds the sin(x) for even indices and cos(x) for odd indices, where
        x = pos * exp(-log(10000) * 2i/d_model)

        If `optimized` is False, a Python list will be used to slice the position
        embeddings. Though we wouldn't do this normally, it can be used to show case
        how using Python objects on torch tensors that are in the GPU can cause slow
        downs.
        """
        self.pos_encoder = torch.zeros((max_seq_len, d_model))

        idx = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            -torch.log(torch.tensor(10000.0))
            * torch.arange(0, d_model, 2, dtype=torch.float)
            / d_model
        )
        self.pos_encoder[:, 0::2] = torch.sin(idx * div_term)
        self.pos_encoder[:, 1::2] = torch.cos(idx * div_term)
        self.pos_encoder = self.pos_encoder.unsqueeze(0)

        self.register_buffer("self.pos_encoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


a = PositionEmbedding(2, 6)
