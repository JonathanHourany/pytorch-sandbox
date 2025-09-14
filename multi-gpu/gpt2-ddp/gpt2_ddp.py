import os
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from rich.traceback import install
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

## Note: The proper way to handle forked multi-threaded tokenization would be to move
# the encoding stage in the Dataset class from the __init__ into the __getitem__ which
# has the added benifit of saving memory. For small datasets and quick demos, disabling
# Huggingface token paralelism is the easiest way to avoid seeing Huggingface's
# warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

install(show_locals=True)


class GPT2Dataset(Dataset):
    def __init__(self, corpus: Iterable[Iterable[str]], tokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.data = []

        for text in corpus:
            input_ids = tokenizer.encode(text)
            for i in range(len(input_ids) - seq_len):
                x = input_ids[i : i + seq_len]
                y = input_ids[i + 1 : i + seq_len + 1]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]

        return (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_attn_heads: int = 2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_attn_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        self.lm_head = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        transf = self.transformer(embed)

        return self.lm_head(transf)


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1235"

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def trace_handler(batch_size):
    """Saves traces in both Chrome and Tensorboard formats"""

    def _trace_handler(prof):
        prof.export_chrome_trace(f"mini-x-trace-batch-{batch_size}")
        prof.export_events(f"./tensorboard/trace_rank{dist.get_rank()}")

    return _trace_handler


def write_key_averages(file_path: Path, cpu_profile, gpu_profile=None):
    with open(file_path / "key_averages.txt", "w") as fp:
        fp.write("===== CPU Profile =====\n")
        fp.write(cpu_profile)

        if gpu_profile:
            fp.write("\n\n===== GPU Profile =====\n")
            fp.write(gpu_profile)


def train(
    rank: int,
    world_size: int,
    num_epochs: int,
    dataset_size: float = 500,
    batch_size: int = 64,
    dataload_workers: int = 8,
):
    setup(rank=rank, world_size=world_size)

    run_details = f"world-size-{world_size}-num-workers-{dataload_workers}-batch-size-{batch_size}"
    file_path = Path(f"profiler-output/{run_details}/rank{rank}")
    file_path.mkdir(parents=True)

    device = f"cuda:{rank}"
    num_layers = 2

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=f"train[:{dataset_size}]"
    )["text"]
    train_dataset = GPT2Dataset(dataset, tokenizer=tokenizer, seq_len=64)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=dataload_workers,
    )

    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_layers=num_layers,
        num_attn_heads=2,
    )

    ddp_model = DDP(
        model.to(device),
        device_ids=[rank],
        output_device=rank,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)

    ddp_model.train()

    prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

    print(
        f"Dataset Size {dataset_size}, Batch Size {batch_size}, Dataload Workers {dataload_workers}"
    )
    print("Setup complete, starting training loop...")

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        schedule=prof_schedule,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        # on_trace_ready=trace_handler,
    ) as prof:
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            for input_ids, pred_ids in train_dataloader:

                input_ids = input_ids.to(device)
                pred_ids = pred_ids.to(device)

                with record_function(f"RANK {rank} Forward"):
                    logits = ddp_model(input_ids)

                flatten_logits = logits.view(-1, tokenizer.vocab_size)

                with record_function(f"RANK {rank} Criterion"):
                    loss = criterion(flatten_logits, pred_ids.view(-1))

                with record_function(f"RANK {rank} Backward"):
                    loss.backward()

                with record_function(f"RANK {rank} Optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss

                prof.step()

            if rank == 0:
                print(f"Avg Epoch loss: {epoch_loss.item() / len(train_dataloader)}")

        print("=" * 20)
        cpu_profile = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        print(cpu_profile)

        gpu_profile = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print(gpu_profile)

        write_key_averages(
            file_path=file_path, cpu_profile=cpu_profile, gpu_profile=gpu_profile
        )
        prof.export_chrome_trace(str(file_path / f"trace-{run_details}.json"))
        prof.export_memory_timeline(str(file_path / f"memory-{run_details}.html"))
        prof.export_stacks(str(file_path / f"stacks-{run_details}.json"))

    cleanup()


if __name__ == "__main__":
    world_size = 1
    num_epochs = 3

    mp.spawn(
        fn=train,
        args=(world_size, num_epochs),
        nprocs=world_size,
    )
    # train(0, 1, 1)
