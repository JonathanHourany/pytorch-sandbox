import torch
import torch.nn as nn
import torch.optim as optim
import typer
from rich.traceback import install
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, TensorDataset

install(show_locals=True)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, num_heads: int, dropout=0.1):
        super().__init__()

        self.sub_layer1 = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            ),
        )
        self.sub_layer2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(in_features=d_model, out_features=d_model)
        )

    def forward(self, x):
        mha_residules = self.sub_layer1(x) + x
        ffw_residules = self.sub_layer2(mha_residules) + x

        return ffw_residules


class ToyTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dim_feedforward: int,
        num_heads: int,
        dropout=0.1,
        num_blocks=2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            *(
                TransformerBlock(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            )
        )
        self.llm_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.tensor):
        return self.llm_head(self.net(x))


def main(
    d_model: int = 256,
    dim_feedforward: int = 512,
    num_heads: int = 4,
    num_blocks: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    true_weights = torch.rand((d_model, 1))
    X = torch.randn((10_000, 128, d_model))
    y = torch.rand(10_000, d_model) @ true_weights + torch.randn(10_000, d_model) * 0.1

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    model = ToyTransformer(
        d_model=d_model,
        num_classes=128,
        dim_feedforward=dim_feedforward,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

    model.to(device)
    model.train()

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True,
    ) as prof:
        for _ in range(2):
            epoch_loss = 0.0

            for train_X, train_y in dataloader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)

                loss = criterion(logits, y)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss

            print(f"Avg Epoch Loss: {epoch_loss / len(dataloader.dataset)}")


if __name__ == "__main__":
    typer.run(main)
