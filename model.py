import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    def __init__(self, data_dim: int, width: int = 256):
        super().__init__()
        self.data_dim = data_dim
        self.width = width
        self.layers = nn.ModuleList(
            [
                nn.Linear(data_dim + 1, width),
                nn.Linear(width, width),
                nn.Linear(width, width),
                nn.Linear(width, width),
            ]
        )
        self.out_adapter = nn.Linear(width, data_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_col = t.view(-1, 1).to(dtype=z.dtype)
        h = torch.cat([z, t_col], dim=1)
        h = F.relu(self.layers[0](h), inplace=False)
        h = F.relu(self.layers[1](h), inplace=False)
        h = F.relu(self.layers[2](h), inplace=False)
        h = F.relu(self.layers[3](h), inplace=False)
        return self.out_adapter(h)
