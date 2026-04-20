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
        # Concatenate time as a scalar conditioning feature for each sample.
        t_col = t.view(-1, 1).to(dtype=z.dtype)
        h = torch.cat([z, t_col], dim=1)
        for layer in self.layers:
            h = F.relu(layer(h), inplace=False)
        return self.out_adapter(h)
