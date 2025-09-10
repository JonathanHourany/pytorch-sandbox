import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.profiler import ProfilerActivity, profile, schedule


class GPT2Dataset(Dataset):
    pass


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        embedding_dim=128,
        nhead=8,
        dim_feedforward=2048,
        num_layers=4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.net = nn.Sequential(
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            ),
            nn.Linear(embedding_dim, vocab_size),
        )

    def forwqrd(self, sequence):
        embeds = self.embedding(sequence)

        return self.transformer(embeds)
