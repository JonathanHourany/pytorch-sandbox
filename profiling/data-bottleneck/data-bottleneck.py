import logging
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def write_key_averages(file_path: Path, cpu_profile, gpu_profile=None):
    with open(file_path / "key_averages.txt", "w") as fp:
        fp.write("===== CPU Profile =====\n")
        fp.write(cpu_profile)

        if gpu_profile:
            fp.write("\n\n===== GPU Profile =====\n")
            fp.write(gpu_profile)


def main(num_loader_workers=0):
    print_cuda_prof = False
    gpu_profile = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_cuda_prof = True
    else:
        logging.warning(
            "Properly profiling this code requires a GPU. These results may not demo what is expected"
        )
        device = torch.device("cpu")

    X = torch.randn(50_000, 512)
    y = torch.randint(low=0, high=2, size=(50_000,), dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset=dataset, batch_size=64, num_workers=num_loader_workers)

    prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

    model = ToyModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        schedule=prof_schedule,
        profile_memory=True,
        with_flops=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)

            loss = criterion(preds.squeeze(), y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            prof.step()

        logging.info(f"Loss: {loss.item()}")
        print("=" * 20)
        cpu_profile = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        print(cpu_profile)

        if print_cuda_prof:
            gpu_profile = prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            )
            print(gpu_profile)

        file_path = Path(f"profiler-output/num-workers-{num_loader_workers}")
        file_path.mkdir(parents=True)
        write_key_averages(
            file_path=file_path, cpu_profile=cpu_profile, gpu_profile=gpu_profile
        )

        prof.export_chrome_trace(str(file_path / "trace.json"))
        prof.export_memory_timeline(str(file_path / "memory.html"))
        prof.export_stacks(str(file_path / "stacks.json"))


if __name__ == "__main__":
    main(num_loader_workers=4)
