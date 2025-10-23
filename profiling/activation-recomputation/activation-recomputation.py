"""
Demonstrates the memory savings of Activation Recomputation (a.k.a Gradient
Checkpointing). Depending on model architecture, GPU memory may be occupied
predominantly by model activations. Parameters such as the length of sequences or
sizes of batches can cause the size of the activations to explode, forcing
limitations on how we train our models.

We can decide to never store the activations after a layer, but since the activations
are needed for the backward pass, this would mean we'd have to run the same data
through the model multiple times as we iterate through the backward pass, scaling the
number of evaluations by n^2.

Better yet, we can checkpoint the activations at certain points. When it's time for the
backward pass, we can recompute the layers needed from a checkpoint instead of starting
from the beginning of a model, reducing the memory needed to train at the cost of some
redundant computation

See also: https://github.com/cybertronai/gradient-checkpointing
"""

import typer
from rich import print
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.checkpoint import checkpoint
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class ToyEncoder(nn.Module):
    def __init__(
        self, d_model=1024, nhead=16, dim_feedforward=4096, num_layers=3, dropout=0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            (
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            )
        )
        self.fc_out = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x):
        for module in self.layers:
            x = module(x)

        return self.fc_out(x).mean(dim=1)


class CheckpointedEncoder(ToyEncoder):
    def forward(self, x):
        for module in self.layers:
            x = checkpoint(module, x, use_reentrant=False)

        return self.fc_out(x).mean(dim=1)


def write_key_averages(file_path: Path, cpu_profile, gpu_profile=None):
    with open(file_path / "key_averages.txt", "w") as fp:
        fp.write("===== CPU Profile =====\n")
        fp.write(cpu_profile)

        if gpu_profile:
            fp.write("\n\n===== GPU Profile =====\n")
            fp.write(gpu_profile)


def main(
    with_checkpoints: bool = True,
    num_layers: int = 3,
    batch_size: int = 8,
    num_workers: int = 4,
):
    torch.cuda.memory._record_memory_history(max_entries=100000)

    print_cuda_prof = False

    ## Set up file names and paths for exporting profiling runs
    checkpoint_mode = "with-checkpoints" if with_checkpoints else "no-checkpoints"
    run_details = f"{checkpoint_mode}-num-workers-{num_workers}-batch-size-{batch_size}"
    file_path = Path(f"profiler-output/{run_details}")
    file_path.mkdir(parents=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_cuda_prof = True
    else:
        logging.warning(
            "Properly profiling this code requires a GPU. "
            "These results may not demo what is expected"
        )
        device = torch.device("cpu")

    num_samples = 5_000
    seq_len = 512
    d_model = 256

    X = torch.randn(size=(num_samples, seq_len, d_model))
    y = (X.mean(dim=(1,2)) > 0).float()
    dataset = TensorDataset(X, y)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset=dataset, lengths=(train_size, test_size)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=num_workers
    )

    if with_checkpoints:
        model = CheckpointedEncoder(d_model=d_model, num_layers=num_layers)
    else:
        model = ToyEncoder(d_model=d_model, num_layers=num_layers)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    model.train()

    prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=0)
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        schedule=prof_schedule,
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        for epoch in range(3):
            for x_train, y_train in train_dataloader:
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                logits = model(x_train)

                loss = criterion(logits.squeeze(), y_train)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                prof.step()

            print(f"Epoch {epoch} | Loss: {loss.item()}")

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

        torch.cuda.memory._dump_snapshot(str(file_path / f"memory-profile-{run_details}.pkl"))
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    typer.run(main)
