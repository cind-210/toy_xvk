from dataclasses import dataclass
from typing import Dict, List, Union

import torch

from model import ToyMLP


@dataclass
class DiffusionCfg:
    t_eps: float
    noise_scale: Union[float, torch.Tensor]
    sample_steps: int
    sample_method: str


def sample_t(batch_size: int, cfg: DiffusionCfg, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)


def pred_to_v(pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor, pred_type: str, t_eps: float) -> torch.Tensor:
    one_minus_t = (1.0 - t).clamp_min(t_eps)
    if pred_type == "x":
        return (pred - z) / one_minus_t
    if pred_type == "e":
        return (z - pred) / t.clamp_min(t_eps)
    if pred_type == "v":
        return pred
    raise ValueError(f"Unknown pred_type={pred_type!r}")


def train_one_pred_type(
    x_data: torch.Tensor,
    pred_type: str,
    cfg: DiffusionCfg,
    epochs: int,
    batch_size: int,
    lr: float,
    width: int,
    device: torch.device,
) -> Dict[str, object]:
    model = ToyMLP(data_dim=x_data.shape[1], width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses: List[float] = []

    n = x_data.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x = x_data[idx]
        t = sample_t(batch_size, cfg, device).view(-1, 1)
        e = torch.randn_like(x) * cfg.noise_scale

        # Interpolate from pure noise to data and always optimize in v-space,
        # regardless of whether the network predicts x, e, or v directly.
        z = t * x + (1.0 - t) * e
        v_target = (x - z) / (1.0 - t).clamp_min(cfg.t_eps)
        pred = model(z, t.flatten())
        v_pred = pred_to_v(pred, z, t, pred_type, cfg.t_eps)
        loss = ((v_target - v_pred) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"[{pred_type}] epoch {ep+1:5d}/{epochs}  loss={losses[-1]:.6f}")

    return {"model": model, "losses": losses}


@torch.no_grad()
def sample_from_model(
    model: torch.nn.Module,
    pred_type: str,
    num_samples: int,
    data_dim: int,
    cfg: DiffusionCfg,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    z = cfg.noise_scale * torch.randn(num_samples, data_dim, device=device)
    t_seq = torch.linspace(0.0, 1.0, cfg.sample_steps + 1, device=device)

    def v_of(z_now: torch.Tensor, t_now: torch.Tensor) -> torch.Tensor:
        pred = model(z_now, t_now.expand(z_now.shape[0]))
        return pred_to_v(pred, z_now, t_now.view(1, 1), pred_type, cfg.t_eps)

    for i in range(cfg.sample_steps - 1):
        t = t_seq[i]
        t_next = t_seq[i + 1]
        dt = t_next - t
        if cfg.sample_method == "euler":
            v = v_of(z, t)
            z = z + dt * v
        elif cfg.sample_method == "heun":
            # Heun uses an Euler proposal plus a corrected slope estimate.
            v_t = v_of(z, t)
            z_euler = z + dt * v_t
            v_t_next = v_of(z_euler, t_next)
            z = z + dt * 0.5 * (v_t + v_t_next)
        else:
            raise ValueError(f"Unsupported sample_method={cfg.sample_method!r}")

    t = t_seq[-2]
    t_next = t_seq[-1]
    z = z + (t_next - t) * v_of(z, t)
    return z
