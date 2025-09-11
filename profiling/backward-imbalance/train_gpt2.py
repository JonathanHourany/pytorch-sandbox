import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GPT2Dataset(Dataset):
    def __init__(
        self, corpus: Iterable[Iterable[str]], tokenizer, sequence_len: int = 64
    ):
        self.tokenizer = tokenizer
        self.data = []

        for text in corpus:
            input_ids = tokenizer.encode(text)
            for i in range(len(input_ids) - sequence_len):
                x = input_ids[i : i + sequence_len]
                y = input_ids[i + 1 : i + sequence_len + 1]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]

        return (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_layers=4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.net = nn.Sequential(
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            ),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, sequence):
        embeds = self.embedding(sequence)

        return self.net(embeds)


def main(dataset_size=10_000, batch_size=32, num_epochs=3, num_workers=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    corpus = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=f"train[:{dataset_size}]"
    )["text"]
    dataset = GPT2Dataset(corpus, tokenizer=tokenizer, sequence_len=64)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    print("Data loaded!")

    model = GPT2Model(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_layers=4,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())

    prof_schedule = schedule(wait=0, warmup=0, active=5, repeat=2)

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for epoch in range(num_epochs):
            epoch_loss = 0
            for step, (X, y) in enumerate(dataloader):
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                flatten_logits = logits.view(-1, tokenizer.vocab_size)

                loss = criterion(flatten_logits, y.view(-1))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss

                prof.step()

            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
            print(f"Avg Epoch Loss: {avg_epoch_loss.item():.4f}")


if __name__ == "__main__":
    main(dataset_size=100, batch_size=32, num_epochs=3, num_workers=8)
