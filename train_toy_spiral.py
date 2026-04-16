import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_spiral_points(
    num_points: int,
    turns: float = 2.0,
    noise_std: float = 0.02,
    curve_res: int = 20000,
) -> torch.Tensor:
    # Approximately uniform in XY via arc-length spacing on the spiral,
    # then add Gaussian noise so points are not exactly on the curve.
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


def save_scatter(points_2d: torch.Tensor, title: str, out_path: str) -> None:
    arr = points_2d.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(arr[:, 0], arr[:, 1], s=8, alpha=0.75)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_projection_matrix(high_dim: int, device: torch.device) -> torch.Tensor:
    # P shape: (high_dim, 2), with P^T P = I_2
    raw = torch.randn(high_dim, 2, device=device)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    p = q[:, :2].contiguous()
    return p


def make_identity_projection_matrix(device: torch.device) -> torch.Tensor:
    # Identity projection for 2D: P = I_2
    return torch.eye(2, device=device)


class ToyMLP(nn.Module):
    # 5 linear layers total:
    # 1) (D+1)->W, 2) W->W, 3) W->W, 4) W->W, 5) out_adapter W->D
    # t is concatenated directly as one extra input channel.
    def __init__(self, data_dim: int, width: int = 256):
        super().__init__()
        self.data_dim = data_dim
        self.width = width
        self.layers = nn.ModuleList([
            nn.Linear(data_dim + 1, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
        ])
        self.out_adapter = nn.Linear(width, data_dim)
        # nn.init.constant_(self.out_adapter.weight, 0.0)
        # nn.init.constant_(self.out_adapter.bias, 0.0)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_col = t.view(-1, 1).to(dtype=z.dtype)
        h = torch.cat([z, t_col], dim=1)
        h = F.relu(self.layers[0](h), inplace=False)
        # No residual connections for now.
        h = F.relu(self.layers[1](h), inplace=False)
        h = F.relu(self.layers[2](h), inplace=False)
        h = F.relu(self.layers[3](h), inplace=False)
        return self.out_adapter(h)


@dataclass
class DiffusionCfg:
    t_eps: float
    noise_scale: float
    sample_steps: int
    sample_method: str


def sample_t(batch_size: int, cfg: DiffusionCfg, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)


def pred_to_v(pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor, pred_type: str, t_eps: float) -> torch.Tensor:
    one_minus_t = (1.0 - t).clamp_min(t_eps)
    if pred_type == "x":
        return (pred - z) / one_minus_t
    if pred_type == "e":
        # z = t*x + (1-t)*e  => v = (x-z)/(1-t) = (z-e)/t
        return (z - pred) / t.clamp_min(t_eps)
    if pred_type == "v":
        return pred
    raise ValueError(f"Unknown pred_type={pred_type!r}")


def apply_v_k_subtract(v_pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor, t_eps: float, k: float) -> torch.Tensor:
    # Same form as JiT option: v <- v - (z/(1-t))*max(0, k*t+1)
    one_minus_t = (1.0 - t).clamp_min(t_eps)
    f_t = torch.clamp(k * t + 1.0, min=0.0)
    return v_pred - (z / one_minus_t) * f_t


def apply_v_h_subtract(v_pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor, t_eps: float, h: float) -> torch.Tensor:
    # Per request: f_t = exp(ln(0.01) * t / h)
    one_minus_t = (1.0 - t).clamp_min(t_eps)
    f_t = torch.exp(math.log(0.01) * t / h)
    return v_pred - (z / one_minus_t) * f_t


def parse_pred_specs(spec: str) -> List[Dict[str, Optional[float]]]:
    out: List[Dict[str, Optional[float]]] = []
    for raw in [x.strip().lower() for x in spec.split(",") if x.strip()]:
        if raw in ("x", "e", "v"):
            out.append({"name": raw, "pred_type": raw, "k": None, "h": None})
            continue
        if raw.startswith("k="):
            k = float(raw.split("=", 1)[1])
            name = f"k={k:g}"
            out.append({"name": name, "pred_type": "v", "k": k, "h": None})
            continue
        if raw.startswith("h="):
            h = float(raw.split("=", 1)[1])
            if h == 0.0:
                raise ValueError("h must be non-zero in h=<float>.")
            name = f"h={h:g}"
            out.append({"name": name, "pred_type": "v", "k": None, "h": h})
            continue
        raise ValueError(f"Unsupported pred spec: {raw!r}. Use x/e/v or k=<float> or h=<float>.")
    return out


def train_one_pred_type(
    x_data: torch.Tensor,
    pred_type: str,
    vpred_k: Optional[float],
    vpred_h: Optional[float],
    cfg: DiffusionCfg,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, object]:
    model = ToyMLP(data_dim=x_data.shape[1], width=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses: List[float] = []

    n = x_data.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x = x_data[idx]
        t = sample_t(batch_size, cfg, device).view(-1, 1)
        e = torch.randn_like(x) * cfg.noise_scale

        z = t * x + (1.0 - t) * e
        v_target = (x - z) / (1.0 - t).clamp_min(cfg.t_eps)
        pred = model(z, t.flatten())
        v_pred = pred_to_v(pred, z, t, pred_type, cfg.t_eps)
        if vpred_k is not None:
            v_pred = apply_v_k_subtract(v_pred, z, t, cfg.t_eps, vpred_k)
        if vpred_h is not None:
            v_pred = apply_v_h_subtract(v_pred, z, t, cfg.t_eps, vpred_h)
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
    model: nn.Module,
    pred_type: str,
    vpred_k: Optional[float],
    vpred_h: Optional[float],
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
        v_pred = pred_to_v(pred, z_now, t_now.view(1, 1), pred_type, cfg.t_eps)
        if vpred_k is not None:
            v_pred = apply_v_k_subtract(v_pred, z_now, t_now.view(1, 1), cfg.t_eps, vpred_k)
        if vpred_h is not None:
            v_pred = apply_v_h_subtract(v_pred, z_now, t_now.view(1, 1), cfg.t_eps, vpred_h)
        return v_pred

    for i in range(cfg.sample_steps - 1):
        t = t_seq[i]
        t_next = t_seq[i + 1]
        dt = t_next - t
        if cfg.sample_method == "euler":
            v = v_of(z, t)
            z = z + dt * v
        elif cfg.sample_method == "heun":
            v_t = v_of(z, t)
            z_euler = z + dt * v_t
            v_t_next = v_of(z_euler, t_next)
            z = z + dt * 0.5 * (v_t + v_t_next)
        else:
            raise ValueError(f"Unsupported sample_method={cfg.sample_method!r}")

    # last step euler, same style as JiT
    t = t_seq[-2]
    t_next = t_seq[-1]
    z = z + (t_next - t) * v_of(z, t)
    return z


def plot_all_losses(loss_dict: Dict[str, List[float]], out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for k, v in loss_dict.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (v-loss)")
    plt.title("Training Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_compare(real_xy: torch.Tensor, gen_xy_dict: Dict[str, torch.Tensor], out_path: str) -> None:
    keys = list(gen_xy_dict.keys())
    cols = len(keys) + 1
    plt.figure(figsize=(5 * cols, 5))

    ax = plt.subplot(1, cols, 1)
    r = real_xy.detach().cpu().numpy()
    ax.scatter(r[:, 0], r[:, 1], s=8, alpha=0.75)
    ax.set_title("Real Spiral (2D)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    for i, k in enumerate(keys, start=2):
        ax = plt.subplot(1, cols, i)
        g = gen_xy_dict[k].detach().cpu().numpy()
        ax.scatter(g[:, 0], g[:, 1], s=8, alpha=0.75)
        ax.set_title(f"Generated ({k}-pred)")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser("Toy x/e/v-pred on 2D spiral with random orthonormal projection")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of spiral points")
    parser.add_argument("--turns", type=float, default=2.0, help="Spiral turns")
    parser.add_argument("--high_dim", type=int, default=64, help="Projected dimension D (P is D x 2)")
    parser.add_argument(
        "--projection_mode",
        type=str,
        default="random_orthonormal",
        choices=["random_orthonormal", "identity"],
        help="Projection matrix mode: random_orthonormal (default) or identity (P=I_2).",
    )
    parser.add_argument("--noise_std", type=float, default=0.02, help="Std of Gaussian noise added to 2D points")
    parser.add_argument("--curve_res", type=int, default=20000, help="Dense curve resolution for arc-length sampling")
    parser.add_argument(
        "--pred_types",
        type=str,
        default="x,e,v",
        help="Comma-separated specs: x,e,v and/or k=<float> and/or h=<float>. Example: x,e,v,k=0,h=0.3",
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--sample_method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="toy/out")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    pred_specs = parse_pred_specs(args.pred_types)

    # 1) sample 2D spiral and visualize
    x2 = make_spiral_points(args.num_points, args.turns, args.noise_std, args.curve_res)
    save_scatter(
        x2,
        f"2D Spiral noisy ({args.num_points} pts, turns={args.turns}, noise={args.noise_std})",
        os.path.join(args.out_dir, "spiral_2d.png"),
    )
    print(f"Saved: {os.path.join(args.out_dir, 'spiral_2d.png')}")

    # 2) projection matrix P (random orthonormal or identity)
    if args.projection_mode == "identity":
        p = make_identity_projection_matrix(device=device)
    else:
        p = make_projection_matrix(args.high_dim, device=device)
    ptp = (p.transpose(0, 1) @ p).detach().cpu().numpy()
    ptp_err = float(np.abs(ptp - np.eye(2)).max())
    print(
        f"Projection matrix P built. mode={args.projection_mode}, "
        f"shape={tuple(p.shape)}, max|P^T P - I|={ptp_err:.6e}"
    )

    x2_dev = x2.to(device)
    x_hd = x2_dev @ p.transpose(0, 1)  # N x Dp, this is Px

    # Show P^T(Px) reconstruction on plane.
    x2_recon = x_hd @ p
    save_scatter(x2_recon, "P^T(Px) on plane", os.path.join(args.out_dir, "spiral_recovered_from_highdim.png"))
    print(f"Saved: {os.path.join(args.out_dir, 'spiral_recovered_from_highdim.png')}")

    cfg = DiffusionCfg(
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        sample_steps=args.sample_steps,
        sample_method=args.sample_method,
    )

    # 3) train x/e/v-pred models (all with v-loss)
    trained: Dict[str, Dict[str, object]] = {}
    for spec in pred_specs:
        name = str(spec["name"])
        pred_type = str(spec["pred_type"])
        vpred_k = spec["k"]
        vpred_h = spec["h"]
        print(f"\n=== Training {name} (base={pred_type}, v-loss) ===")
        trained[name] = train_one_pred_type(
            x_data=x_hd,
            pred_type=pred_type,
            vpred_k=vpred_k,
            vpred_h=vpred_h,
            cfg=cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

    # 4) sample in high-dim and map back by P^T y for visualization
    gen_xy: Dict[str, torch.Tensor] = {}
    all_losses: Dict[str, List[float]] = {}
    for spec in pred_specs:
        name = str(spec["name"])
        pred_type = str(spec["pred_type"])
        vpred_k = spec["k"]
        vpred_h = spec["h"]
        model = trained[name]["model"]  # type: ignore[index]
        losses = trained[name]["losses"]  # type: ignore[index]
        all_losses[name] = losses  # type: ignore[assignment]

        y_hd = sample_from_model(
            model=model,  # type: ignore[arg-type]
            pred_type=pred_type,
            vpred_k=vpred_k,
            vpred_h=vpred_h,
            num_samples=args.num_points,
            data_dim=x_hd.shape[1],
            cfg=cfg,
            device=device,
        )
        y_xy = y_hd @ p
        gen_xy[name] = y_xy

        save_scatter(
            y_xy,
            title=f"Generated mapped by P^T ({name})",
            out_path=os.path.join(args.out_dir, f"generated_{name}_plane.png"),
        )
        print(f"Saved: {os.path.join(args.out_dir, f'generated_{name}_plane.png')}")

    plot_all_losses(all_losses, os.path.join(args.out_dir, "loss_curves.png"))
    print(f"Saved: {os.path.join(args.out_dir, 'loss_curves.png')}")

    plot_compare(x2_dev, gen_xy, os.path.join(args.out_dir, "compare_real_vs_generated.png"))
    print(f"Saved: {os.path.join(args.out_dir, 'compare_real_vs_generated.png')}")

    # Save metadata and projection matrix
    torch.save(
        {
            "P": p.detach().cpu(),
            "ptp_max_err": ptp_err,
            "args": vars(args),
            "final_losses": {k: float(v[-1]) for k, v in all_losses.items()},
        },
        os.path.join(args.out_dir, "run_meta.pt"),
    )
    print(f"Saved: {os.path.join(args.out_dir, 'run_meta.pt')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
