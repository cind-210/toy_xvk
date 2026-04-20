import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

POINT_ALPHA = 0.5
TOP_ANNOT_FONTSIZE = 30


def save_scatter(points_2d: torch.Tensor, title: str, out_path: str) -> None:
    arr = points_2d.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(arr[:, 0], arr[:, 1], s=8, alpha=POINT_ALPHA)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_loss_window(
    loss_dict: Dict[str, List[float]],
    out_path: str,
    start_idx: int,
    end_idx: int,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    xs = np.arange(start_idx + 1, end_idx + 1)
    for k, v in loss_dict.items():
        ys = np.asarray(v[start_idx:end_idx], dtype=np.float32)
        plt.plot(xs, ys, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (v-loss)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_all_losses(loss_dict: Dict[str, List[float]], out_dir: str, epochs: int) -> None:
    if epochs > 200:
        _plot_loss_window(
            loss_dict=loss_dict,
            out_path=os.path.join(out_dir, "loss_curves_first200.png"),
            start_idx=0,
            end_idx=min(200, epochs),
            title="Training Curves (First 200 Epochs)",
        )
        _plot_loss_window(
            loss_dict=loss_dict,
            out_path=os.path.join(out_dir, "loss_curves_last10.png"),
            start_idx=max(0, epochs - 10),
            end_idx=epochs,
            title="Training Curves (Last 10 Epochs)",
        )
    else:
        _plot_loss_window(
            loss_dict=loss_dict,
            out_path=os.path.join(out_dir, "loss_curves.png"),
            start_idx=0,
            end_idx=epochs,
            title="Training Curves",
        )


def plot_compare_grid(
    real_xy: torch.Tensor,
    grid_xy: Dict[Tuple[int, str], torch.Tensor],
    row_high_dims: List[int],
    col_keys: List[str],
    col_titles: Dict[str, str],
    col_ns_labels: Dict[str, str],
    out_path: str,
) -> None:
    rows = len(row_high_dims)
    cols = len(col_keys) + 1
    fig = plt.figure(figsize=(4.8 * cols, 4.0 * rows))
    real_np = real_xy.detach().cpu().numpy()
    first_col_axes: List[object] = []
    top_row_axes: List[object] = []

    for r_idx, hd in enumerate(row_high_dims):
        ax = plt.subplot(rows, cols, r_idx * cols + 1)
        first_col_axes.append(ax)
        ax.scatter(real_np[:, 0], real_np[:, 1], s=8, alpha=POINT_ALPHA)
        if r_idx == 0:
            top_row_axes.append(ax)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        for c_idx, ck in enumerate(col_keys, start=1):
            ax = plt.subplot(rows, cols, r_idx * cols + c_idx + 1)
            g = grid_xy[(hd, ck)].detach().cpu().numpy()
            ax.scatter(g[:, 0], g[:, 1], s=8, alpha=POINT_ALPHA)
            if r_idx == 0:
                top_row_axes.append(ax)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0.16, 0.0, 1.0, 0.88])

    # Add titles after tight_layout so they align with the final subplot positions.
    for ax, title in zip(top_row_axes, ["real"] + [col_titles.get(ck, ck) for ck in col_keys]):
        pos = ax.get_position()
        x_center = 0.5 * (pos.x0 + pos.x1)
        y_title = min(0.992, pos.y1 + 0.045)
        plt.figtext(x_center, y_title, title, ha="center", va="bottom", fontsize=TOP_ANNOT_FONTSIZE)

    for ax, ck in zip(top_row_axes[1:], col_keys):
        ns_label = col_ns_labels.get(ck, "")
        if ns_label == "":
            continue
        pos = ax.get_position()
        x_center = 0.5 * (pos.x0 + pos.x1)
        y_ns = min(0.972, pos.y1 + 0.018)
        plt.figtext(
            x_center,
            y_ns,
            ns_label,
            ha="center",
            va="bottom",
            fontsize=max(8, int(TOP_ANNOT_FONTSIZE * 0.5)),
        )

    if len(first_col_axes) > 0:
        left_x0 = first_col_axes[0].get_position().x0
        x_col = max(0.015, left_x0 - 0.11)
        y_top = min(0.992, first_col_axes[0].get_position().y1 + 0.045)
        plt.figtext(x_col, y_top, "high_dim", ha="center", va="top", fontsize=TOP_ANNOT_FONTSIZE)
        for ax, hd in zip(first_col_axes, row_high_dims):
            pos = ax.get_position()
            y_center = 0.5 * (pos.y0 + pos.y1)
            plt.figtext(x_col, y_center, str(hd), ha="center", va="center", fontsize=TOP_ANNOT_FONTSIZE)

    plt.savefig(out_path, dpi=180)
    plt.close()
