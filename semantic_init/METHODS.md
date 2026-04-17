# `--method` (semantic warm-start for inversion)

When you run `semantic_inversion.py` or `run_semantic.sh`, **`--method`** (or shell env **`SEMANTIC_METHOD`**) is the single switch that chooses **how the anchor library is turned into an initial image** for gradient inversion. Each value is implemented in a dedicated **`baseline_*`** function so you can reproduce and compare runs.

## Anchor directory layout (multiple images per class)

Under `--anchors-dir` (default `anchors/` next to this script):

1. **Flat files** (as before): `animal.jpg`, `building.jpg`, ... — each file is one anchor; display id = filename stem.
2. **Per-class subfolders**: `animal/a.jpg`, `animal/b.jpg`, ... — every **direct** image file inside a subdirectory becomes its own anchor with display id **`<folder>/<stem>`** (e.g. `animal/a`). You can mix flat files and subfolders.

All anchors are resized to the target ViT input resolution; KNN / best / mixup operate on this **expanded list** of images.

## Available `method` values

| `method` | One-line meaning | Main implementation |
|----------|------------------|----------------------|
| **`gray`** | **Minimal baseline:** no anchors, no semantic warm-start — multi-restart **gray (+noise) init** only (`invert_gray`). Same objective as other methods, but this is the reference curve people usually compare against. | `baseline_gray_inversion_for_layer` |
| **`best`** | At a chosen ViT layer, score every anchor by feature similarity and use **one best anchor per layer** as the warm-start. | `baseline_best_semantic_init_for_layer` (`AnchorLibrary.find_best`; **cosine** on mean-pooled tokens) |
| **`mixup`** | Resize **all** anchors to the target size, **uniform pixel average**, optional **Gaussian blur**, one global init reused **for every inversion layer**. | `baseline_mixup_prepare_global` + `baseline_mixup_semantic_init_for_layer` |
| **`knn-mixup`** | Rank anchors by similarity at a chosen layer, take **top-k**, **uniform pixel average** of those k images, optional blur; **each inversion layer** re-selects neighbors. | `baseline_knn_mixup_semantic_init_for_layer` (`AnchorLibrary.knn_mixup_pixel`) |
| **`knn-weighted-mixup`** | Same KNN ranking as `knn-mixup`, but pixels are combined with **softmax(score / tau)** weights over the top-k (tau from **`--knn-weight-tau`**; tau ≤ 0 ⇒ **uniform** weights). Optional blur after mixing. | `baseline_knn_weighted_mixup_semantic_init_for_layer` (`AnchorLibrary.knn_weighted_mixup_pixel`) |

Central dispatcher: **`run_semantic_layer_init_from_method`** (all anchor-based methods). **`gray`** is handled separately in `main` because it does not use anchors.

For new experiments, add a new **`baseline_*`** and a new branch in the dispatcher; **avoid editing old `baseline_*` bodies** so historical runs stay comparable.

**`--compare`:** when `method=gray`, the extra gray run would be redundant, so **`--compare` is ignored** (a note is printed). For other methods, `--compare` still runs the gray baseline alongside semantic warm-start for PSNR comparison.

## Parameters tied to each `method`

- **`gray`**: only inversion hyperparameters matter; **`--anchors-dir` is not read** (no `AnchorLibrary`).
- **Shared (non-gray)**: `--anchors-dir`, `--match` (`all` / `cls` / `patch`), `--match-layer` (`same` or fixed index for **best** / **knn-***).
- **`mixup` / `knn-*`**: `--mixup-blur-kernel`, `--mixup-blur-sigma` (sigma `0` = no blur after averaging).
- **`knn-mixup` / `knn-weighted-mixup`**: `--knn-k`, `--knn-sim` (`cosine_mean`, `neg_l2_mean`).
- **`knn-weighted-mixup` only**: `--knn-weight-tau` (default `0.1`) — softmax temperature; smaller ⇒ sharper weighting on the nearest neighbor.

Inversion itself: `--layers`, `--steps`, `--lr`, loss weights, `--restarts`. Gray baseline: **`--compare`** / `SEMANTIC_COMPARE=1`.

## CLI and script examples

```bash
python semantic_inversion.py --image path/to.jpg --method gray
python semantic_inversion.py --image path/to.jpg --method best
python semantic_inversion.py --image path/to.jpg --method knn-weighted-mixup --knn-k 5 --knn-weight-tau 0.05
```

```bash
SEMANTIC_METHOD=knn-weighted-mixup SEMANTIC_KNN_WEIGHT_TAU=0.2 bash run_semantic.sh /path/to.jpg
```

## Outputs and logging

- Default output: `results/runs/<YYYYMMDD-HHMMSS>/` (or `--output-dir`).
- `summary.json` includes **`method`**, **`experiment_description`**, and per-row extras such as **`knn_top`**, **`knn_softmax_weights`** (for `knn-weighted-mixup`).
