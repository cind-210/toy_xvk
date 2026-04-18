import math

import torch


def make_spiral_points(
    num_points: int,
    turns: float = 2.0,
    noise_std: float = 0.02,
    curve_res: int = 20000,
) -> torch.Tensor:
    t_dense = torch.linspace(0.0, turns * 2.0 * math.pi, curve_res)
    r_dense = t_dense / (turns * 2.0 * math.pi)
    x_dense = r_dense * torch.cos(t_dense)
    y_dense = r_dense * torch.sin(t_dense)
    curve = torch.stack([x_dense, y_dense], dim=1).float()

    seg = curve[1:] - curve[:-1]
    seg_len = torch.norm(seg, dim=1)
    cum = torch.cat([torch.zeros(1), torch.cumsum(seg_len, dim=0)], dim=0)
    total = cum[-1]
    target = torch.linspace(0.0, float(total), num_points)

    idx = torch.searchsorted(cum, target, right=True).clamp(min=1, max=curve_res - 1)
    left = idx - 1
    right = idx
    l0 = cum[left]
    l1 = cum[right]
    w = ((target - l0) / (l1 - l0).clamp_min(1e-8)).unsqueeze(1)
    pts = curve[left] * (1.0 - w) + curve[right] * w

    if noise_std > 0:
        pts = pts + noise_std * torch.randn_like(pts)
    pts = torch.clamp(pts, -1.0, 1.0)
    return pts


def make_line_points(num_points: int, noise_std: float = 0.02) -> torch.Tensor:
    x = torch.linspace(-0.5, 0.5, num_points)
    y = torch.full_like(x, 0.5)
    pts = torch.stack([x, y], dim=1).float()
    if noise_std > 0:
        pts = pts + noise_std * torch.randn_like(pts)
    return pts


def make_projection_matrix(high_dim: int, device: torch.device) -> torch.Tensor:
    raw = torch.randn(high_dim, 2, device=device)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    p = q[:, :2].contiguous()
    return p


def make_identity_projection_matrix(device: torch.device) -> torch.Tensor:
    return torch.eye(2, device=device)
