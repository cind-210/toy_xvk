# Toy: x/e/v-pred on 2D Data

This toy project compares `x-pred`, `e-pred`, and `v-pred` under the same `v-loss`.

## What it does

1. Generates 2D data (`spiral` or `line`) with Gaussian noise.
2. Projects 2D points to high-dim space with a fixed matrix `P` (`X = P x`).
3. Trains a ReLU MLP denoiser with `v-loss`.
4. Samples in high-dim and maps back to 2D (`P^T y`) for visualization.
5. Supports grid comparison:
   - rows: `high_dim`
   - columns: pred/noise configurations

## Key options

- `--pred_types`: comma list from `x,e,v`
- `--high_dim`: comma list, e.g. `2,4,8`
- `--noise_scale`: `auto`, numeric, numeric list, `e`, `var`, or `ep`
  - `auto`: `v` uses `e`-estimated noise scale, `x/e` use `1.0`
  - `e`: noise_scale = sqrt(E / high_dim), where E = mean(||X||^2)
  - `var`: noise_scale = sqrt(var_total / high_dim), where var_total = sum of per-dim variances
  - `ep`: noise_scale = sqrt(E^(1 + (r - d)/2) / ((exp(-d) + 1) * d))
    - `r`: `TwoNN` 在当前高维数据 `X` 上估计的维度
    - `d`: high_dim
- `--width`: MLP width (default `256`)
- `--shape`: `spiral` or `line`

## Visualization notes

- Left-most column is `real`.
- Left annotation column shows `high_dim` values.
- In `auto` mode:
  - `x-pred` column title adds `(sigma=1.0)`
  - `v-pred` column title adds `(SRN=1.0)`
- If `e-pred` uses manual numeric `noise_scale`, title shows `(noise_scale=...)`.

## Combination rule

The script checks combination-space dimensionality over:
- `high_dim`
- `pred_types`
- `noise_scale`

If dimensionality is greater than 3, it raises an error.

## Default output directory

If `--out_dir` is empty, default is:

`./out/{pred_types}_{high_dim}{optional_noise_suffix}{optional_line_suffix}{optional_width_suffix}`

## Example

```bash
python toy/train.py \
  --shape spiral \
  --num_points 1024 \
  --high_dim 2,4,8 \
  --pred_types x,v \
  --noise_scale auto \
  --width 256 \
  --epochs 4000 \
  --batch_size 256 \
  --sample_steps 100 \
  --sample_method heun
```

## Outputs

- `spiral_2d.png`
- `spiral_recovered_from_highdim.png`
- `compare_real_vs_generated.png`
- `loss_curves.png` (when `epochs <= 200`)
- `loss_curves_first200.png` and `loss_curves_last10.png` (when `epochs > 200`)
- `run_meta.pt`

