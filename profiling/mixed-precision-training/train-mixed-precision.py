"""
Demonstrates the difference in training between using full precision and mixed
precision in PyTorch. Inspecting profiling output shows the GPU total time is cut in
half when trained using mixed precision

@author: Jonathan Hourany
"""

import logging
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import typer
from datasets import load_dataset
from rich import print
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

## Note: The proper way to handle forked multi-threaded tokenization would be to move
# the encoding stage in the Dataset class from the __init__ into the __getitem__ which
# has the added benefit of saving memory. For small datasets and quick demos, disabling
# Huggingface token parallelism is the easiest way to avoid seeing Huggingface's
# warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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
        num_layers=10,
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


def write_key_averages(file_path: Path, cpu_profile, gpu_profile=None):
    with open(file_path / "key_averages.txt", "w") as fp:
        fp.write("===== CPU Profile =====\n")
        fp.write(cpu_profile)

        if gpu_profile:
            fp.write("\n\n===== GPU Profile =====\n")
            fp.write(gpu_profile)


def train_mixed_precision(
    model, X, y, vocab_size, criterion, optimizer, scaler, device
):
    with torch.autocast(device_type=str(device)):
        logits = model(X)
        flatten_logits = logits.view(-1, vocab_size)
        loss = criterion(flatten_logits, y.view(-1))

    scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return loss


def train_full_precision(model, X, y, vocab_size, criterion, optimizer, device):
    logits = model(X)
    flattened_logits = logits.view(-1, vocab_size)

    loss = criterion(flattened_logits, y.view(-1))
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss


def main(
    dataset_size: int = 10_000,
    batch_size: int = 32,
    num_epochs: int = 3,
    num_loader_workers: int = 8,
    mixed_precision_mode: bool = False,
):
    print_cuda_prof = False

    ## Set up file names and paths for exporting profiling runs
    precision_mode = "mixed-precision" if mixed_precision_mode else "full-precision"
    run_details = (
        f"{precision_mode}-num-workers-{num_loader_workers}-batch-size-{batch_size}"
    )
    file_path = Path(f"profiler-output/{run_details}")
    file_path.mkdir(parents=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_cuda_prof = True
    else:
        logging.warning(
            "Properly profiling this code requires a GPU. These results may not demo what is expected"
        )
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("[bold green]Loading data...", end="")
    corpus = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=f"train[:{dataset_size}]"
    )["text"]
    dataset = GPT2Dataset(corpus, tokenizer=tokenizer, sequence_len=64)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_loader_workers,
    )

    print("[bold green]:heavy_check_mark:[/bold green]")
    print("[bold green]Initializing model...", end="")

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
    scaler = torch.amp.grad_scaler.GradScaler()

    print("[bold green]:heavy_check_mark:[/bold green]")
    print("[bold green]Starting training...[/bold green]")

    prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for epoch in range(num_epochs):
            epoch_loss = 0
            for step, (X, y) in enumerate(dataloader):
                X = X.to(device)
                y = y.to(device)

                if not mixed_precision_mode:
                    loss = train_full_precision(
                        model=model,
                        X=X,
                        y=y,
                        vocab_size=tokenizer.vocab_size,
                        criterion=criterion,
                        optimizer=optimizer,
                        device=device,
                    )
                else:
                    loss = train_mixed_precision(
                        model=model,
                        X=X,
                        y=y,
                        vocab_size=tokenizer.vocab_size,
                        criterion=criterion,
                        optimizer=optimizer,
                        scaler=scaler,
                        device=device,
                    )

                epoch_loss += loss
                prof.step()

            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch: {epoch} | Avg Epoch Loss: {avg_epoch_loss.item():.4f}")

        print("=" * 20)
        cpu_profile = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        print(cpu_profile)

        if print_cuda_prof:
            gpu_profile = prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            )
            print(gpu_profile)
        else:
            gpu_profile = None

        write_key_averages(
            file_path=file_path, cpu_profile=cpu_profile, gpu_profile=gpu_profile
        )
        prof.export_chrome_trace(str(file_path / f"trace-{run_details}.json"))
        prof.export_memory_timeline(str(file_path / f"memory-{run_details}.html"))
        prof.export_stacks(str(file_path / f"stacks-{run_details}.json"))


if __name__ == "__main__":
    typer.run(main)
