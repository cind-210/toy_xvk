# Toy: x/e/v-pred on Spiral (v-loss)

This toy project compares `x-pred`, `e-pred`, and `v-pred` under the same `v-loss` training objective.

## What it does

1. Samples a 2D spiral with 2 turns (`--num_points`, default 1024), coordinates in `[-1, 1]`.
   Sampling is approximately uniform in XY (arc-length based), and Gaussian noise is added (`--noise_std`), so points are not strictly on the spiral.
2. Saves the sampled plane plot (`spiral_2d.png`).
3. Builds a random fixed projection matrix `P` with orthonormal columns (`P^T P = I`).
4. Embeds points into high-dimensional space as `Px`.
5. Trains a 5-layer ReLU MLP (width 256, plus input/output linear adapters) for each pred type (`x/e/v`), always with `v-loss`.
6. Samples in high-dim, maps generated points back to 2D via `P^T y`, and saves plots.

## Run

```bash
python toy/train_toy_spiral.py \
  --num_points 1024 \
  --noise_std 0.02 \
  --high_dim 64 \
  --pred_types x,e,v \
  --epochs 4000 \
  --batch_size 256 \
  --sample_steps 100 \
  --sample_method heun \
  --out_dir toy/out
```

## Outputs

- `spiral_2d.png`
- `spiral_recovered_from_highdim.png`
- `generated_x_plane.png`
- `generated_e_plane.png`
- `generated_v_plane.png`
- `loss_curves.png`
- `compare_real_vs_generated.png`
- `run_meta.pt`
