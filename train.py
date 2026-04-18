import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from data import (
    make_identity_projection_matrix,
    make_line_points,
    make_projection_matrix,
    make_spiral_points,
)
from diffusion import DiffusionCfg, sample_from_model, train_one_pred_type
from plot import plot_all_losses, plot_compare_grid, save_scatter
from utils import (
    build_default_out_dir,
    ensure_dir,
    ensure_unique_default_out_dir,
    parse_high_dims,
    parse_noise_scale_modes,
    parse_pred_specs,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser("Toy x/e/v-pred on 2D spiral with random orthonormal projection")
    parser.add_argument("--shape", type=str, default="spiral", choices=["spiral", "line"], help="2D source shape.")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of spiral points")
    parser.add_argument("--turns", type=float, default=2.0, help="Spiral turns")
    parser.add_argument("--high_dim", type=str, default="64", help="Projected dimension list. Example: 2,4,8")
    parser.add_argument(
        "--projection_mode",
        type=str,
        default="random_orthonormal",
        choices=["random_orthonormal", "identity"],
        help="Projection matrix mode: random_orthonormal (default) or identity (P=I_2).",
    )
    parser.add_argument("--noise_std", type=float, default=0.02, help="Std of Gaussian noise added to 2D points")
    parser.add_argument("--curve_res", type=int, default=20000, help="Dense curve resolution for arc-length sampling")
    parser.add_argument("--pred_types", type=str, default="x,e,v", help="Comma-separated specs: x,e,v")
    parser.add_argument(
        "--pred_type",
        type=str,
        default="",
        help="Alias of --pred_types (single string). If set, it overrides --pred_types.",
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=256, help="MLP hidden width.")
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument(
        "--noise_scale",
        type=str,
        default="auto",
        help="Noise-scale mode(s): float, comma list, auto, e, var, ep. Examples: auto | 1.0 | 0.2,0.3 | e | var | ep",
    )
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--sample_method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="", help="Output dir. Empty means auto naming.")
    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    if args.pred_type.strip() != "":
        args.pred_types = args.pred_type
    pred_specs = parse_pred_specs(args.pred_types)
    noise_modes = parse_noise_scale_modes(args.noise_scale)
    high_dims = parse_high_dims(args.high_dim)
    needs_ep = any(str(m["mode"]) == "ep" for m in noise_modes)
    tnn_estimator = None
    if needs_ep:
        try:
            from skdim.id import TwoNN
        except Exception as exc:
            raise RuntimeError(
                "--noise_scale ep requires scikit-dimension. Please install package 'scikit-dimension'."
            ) from exc
        tnn_estimator = TwoNN()

    combo_dims = int(len(high_dims) > 1) + int(len(pred_specs) > 1) + int(len(noise_modes) > 1)
    if combo_dims > 3:
        raise ValueError("Combination dimensionality > 3 is not supported for current visualization.")

    if args.out_dir.strip() == "":
        args.out_dir = build_default_out_dir(args.pred_types, high_dims, args.shape, noise_modes, args.width)
        args.out_dir = ensure_unique_default_out_dir(args.out_dir)
    ensure_dir(args.out_dir)

    if args.shape == "line":
        x2 = make_line_points(args.num_points, args.noise_std)
        title = f"2D Line noisy ({args.num_points} pts, y=0.5, x in [-0.5,0.5], noise={args.noise_std})"
    else:
        x2 = make_spiral_points(args.num_points, args.turns, args.noise_std, args.curve_res)
        title = f"2D Spiral noisy ({args.num_points} pts, turns={args.turns}, noise={args.noise_std})"
    save_scatter(x2, title, os.path.join(args.out_dir, "spiral_2d.png"))
    print(f"Saved: {os.path.join(args.out_dir, 'spiral_2d.png')}")

    x2_dev = x2.to(device)

    col_templates: List[Dict[str, object]] = []
    col_idx = 0
    for nm in noise_modes:
        ns_mode = str(nm["mode"])
        ns_val = nm["value"]
        for spec in pred_specs:
            pred_type = str(spec["pred_type"])
            col_idx += 1
            col_templates.append(
                {
                    "col_id": f"c{col_idx}",
                    "pred_type": pred_type,
                    "noise_mode": ns_mode,
                    "noise_value": ns_val,
                }
            )

    grid_xy: Dict[Tuple[int, str], torch.Tensor] = {}
    all_losses: Dict[str, List[float]] = {}
    run_meta_rows: List[Dict[str, object]] = []
    ptp_err_by_hd: Dict[int, float] = {}
    sources: List[Dict[str, object]] = []

    col_titles: Dict[str, str] = {}
    col_ns_labels: Dict[str, str] = {}
    for t in col_templates:
        cid = str(t["col_id"])
        pred_type = str(t["pred_type"])
        noise_mode = str(t["noise_mode"])
        noise_value = t["noise_value"]
        base = f"{pred_type}-pred"
        col_titles[cid] = base
        if noise_mode == "auto":
            if pred_type == "v":
                col_ns_labels[cid] = "ns=e(auto)"
            else:
                col_ns_labels[cid] = "ns=1.0"
        elif noise_mode == "fixed" and noise_value is not None:
            col_ns_labels[cid] = f"ns={float(noise_value):g}"
        elif noise_mode in ("e", "var", "ep"):
            col_ns_labels[cid] = f"ns={noise_mode}"
        else:
            col_ns_labels[cid] = ""

    # Prepare shared data source (P and projected data) once per high_dim.
    for hd in high_dims:
        if args.projection_mode == "identity":
            if hd != 2:
                raise ValueError("--projection_mode identity only supports high_dim=2.")
            p = make_identity_projection_matrix(device=device)
        else:
            p = make_projection_matrix(hd, device=device)
        ptp = (p.transpose(0, 1) @ p).detach().cpu().numpy()
        ptp_err = float(np.abs(ptp - np.eye(2)).max())
        ptp_err_by_hd[int(hd)] = ptp_err
        print(
            f"[high_dim={hd}] Projection matrix P built. mode={args.projection_mode}, "
            f"shape={tuple(p.shape)}, max|P^T P - I|={ptp_err:.6e}"
        )

        x_hd = x2_dev @ p.transpose(0, 1)
        x_for_stats = x_hd.detach().cpu().numpy().astype(np.float64, copy=False)
        e_energy = float(np.mean(np.linalg.norm(x_for_stats, axis=1) ** 2))
        var_total = float(np.var(x_for_stats, axis=0, ddof=0).sum())
        r_dim = float(tnn_estimator.fit(x_for_stats).dimension_) if tnn_estimator is not None else float("nan")
        d_dim = float(hd)
        e_noise_scale = float(math.sqrt(e_energy / float(hd)))
        var_noise_scale = float(math.sqrt(var_total / float(hd)))
        if tnn_estimator is not None:
            print(f"[high_dim={hd}] TwoNN intrinsic dim r={r_dim:.6g}")

        ep_exp = 1.0 + (r_dim - d_dim) / 2.0
        e_safe = max(e_energy, 1e-12)
        ep_num = math.pow(e_safe, ep_exp)
        ep_denom = (math.exp(-d_dim) + 1.0) * d_dim
        ep_noise_scale = float(math.sqrt(ep_num / ep_denom))

        sources.append(
            {
                "high_dim": int(hd),
                "P": p,
                "x_hd": x_hd,
                "E": e_energy,
                "var": var_total,
                "r_dim": r_dim,
                "e_ns": e_noise_scale,
                "var_ns": var_noise_scale,
                "ep_ns": ep_noise_scale,
            }
        )

    if len(sources) > 0:
        first = sources[0]
        x2_recon = first["x_hd"] @ first["P"]  # type: ignore[operator]
        save_scatter(x2_recon, "P^T(Px) on plane", os.path.join(args.out_dir, "spiral_recovered_from_highdim.png"))
        print(f"Saved: {os.path.join(args.out_dir, 'spiral_recovered_from_highdim.png')}")

    # Reuse prepared source for each configuration.
    for src in sources:
        hd = int(src["high_dim"])
        p = src["P"]  # type: ignore[assignment]
        x_hd = src["x_hd"]  # type: ignore[assignment]
        e_energy = float(src["E"])
        var_total = float(src["var"])
        r_dim = float(src["r_dim"])
        e_noise_scale = float(src["e_ns"])
        var_noise_scale = float(src["var_ns"])
        ep_noise_scale = float(src["ep_ns"])

        for t in col_templates:
            col_id = str(t["col_id"])
            pred_type = str(t["pred_type"])
            noise_mode = str(t["noise_mode"])
            noise_value = t["noise_value"]  # type: ignore[assignment]

            if noise_mode == "auto":
                if pred_type == "v":
                    run_noise_scale = e_noise_scale
                    noise_label = f"ns={run_noise_scale:.4g},E={e_energy:.4g}"
                else:
                    run_noise_scale = 1.0
                    noise_label = "ns=1"
            elif noise_mode == "e":
                run_noise_scale = e_noise_scale
                noise_label = f"ns={run_noise_scale:.4g},E={e_energy:.4g}"
            elif noise_mode == "var":
                run_noise_scale = var_noise_scale
                noise_label = f"ns={run_noise_scale:.4g},var={var_total:.4g}"
            elif noise_mode == "ep":
                run_noise_scale = ep_noise_scale
                noise_label = f"ns={run_noise_scale:.4g},ep(r={r_dim:.0f},d={hd})"
            else:
                assert noise_value is not None
                run_noise_scale = float(noise_value)
                noise_label = f"ns={run_noise_scale:g}"

            run_name = f"{pred_type},{noise_label},hd={hd}"
            cfg_run = DiffusionCfg(
                t_eps=args.t_eps,
                noise_scale=run_noise_scale,
                sample_steps=args.sample_steps,
                sample_method=args.sample_method,
            )

            print(f"\n=== Training {run_name} (base={pred_type}, v-loss) ===")
            trained = train_one_pred_type(
                x_data=x_hd,
                pred_type=pred_type,
                cfg=cfg_run,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                width=args.width,
                device=device,
            )

            y_hd = sample_from_model(
                model=trained["model"],  # type: ignore[arg-type]
                pred_type=pred_type,
                num_samples=args.num_points,
                data_dim=x_hd.shape[1],
                cfg=cfg_run,
                device=device,
            )
            y_xy = y_hd @ p
            grid_xy[(int(hd), col_id)] = y_xy
            all_losses[run_name] = trained["losses"]  # type: ignore[assignment]

            run_meta_rows.append(
                {
                    "high_dim": int(hd),
                    "col_id": col_id,
                    "run_name": run_name,
                    "pred_type": pred_type,
                    "noise_mode": noise_mode,
                    "noise_scale": float(run_noise_scale),
                    "E": float(e_energy),
                    "var": float(var_total),
                    "r_dim": float(r_dim),
                    "final_loss": float(trained["losses"][-1]),  # type: ignore[index]
                }
            )

    plot_all_losses(all_losses, args.out_dir, args.epochs)
    if args.epochs > 200:
        print(f"Saved: {os.path.join(args.out_dir, 'loss_curves_first200.png')}")
        print(f"Saved: {os.path.join(args.out_dir, 'loss_curves_last10.png')}")
    else:
        print(f"Saved: {os.path.join(args.out_dir, 'loss_curves.png')}")

    plot_compare_grid(
        real_xy=x2_dev,
        grid_xy=grid_xy,
        row_high_dims=high_dims,
        col_keys=[str(t["col_id"]) for t in col_templates],
        col_titles=col_titles,
        col_ns_labels=col_ns_labels,
        out_path=os.path.join(args.out_dir, "compare_real_vs_generated.png"),
    )
    print(f"Saved: {os.path.join(args.out_dir, 'compare_real_vs_generated.png')}")

    torch.save(
        {
            "args": vars(args),
            "high_dims": [int(x) for x in high_dims],
            "ptp_max_err_by_high_dim": ptp_err_by_hd,
            "columns": col_titles,
            "column_ns_labels": col_ns_labels,
            "runs": run_meta_rows,
            "final_losses": {k: float(v[-1]) for k, v in all_losses.items()},
        },
        os.path.join(args.out_dir, "run_meta.pt"),
    )
    print(f"Saved: {os.path.join(args.out_dir, 'run_meta.pt')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
