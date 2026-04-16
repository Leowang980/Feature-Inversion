# `--method` (semantic warm-start for inversion)

When you run `semantic_inversion.py` or `run_semantic.sh`, **`--method`** (or shell env **`SEMANTIC_METHOD`**) is the single switch that chooses **how the anchor library is turned into an initial image** for gradient inversion. Each value is implemented in a dedicated **`baseline_*`** function so you can reproduce and compare runs.

## Available `method` values

| `method` | One-line meaning | Main implementation |
|----------|------------------|----------------------|
| **`best`** | At a chosen ViT layer, score every anchor by feature similarity and use **one best anchor per layer** as the warm-start. | `baseline_best_semantic_init_for_layer` (calls `AnchorLibrary.find_best`; similarity is **cosine** on mean-pooled tokens) |
| **`mixup`** | Resize **all** anchors to the target size, take their **uniform pixel average**, optionally **Gaussian blur**, yielding **one** global init reused **for every inversion layer**. | `baseline_mixup_prepare_global` (once) + `baseline_mixup_semantic_init_for_layer` (per-layer logging / target vs mix feat cosine) |
| **`knn-mixup`** | At a chosen layer, rank anchors by similarity to the target, take **top-k**, **average those k images in pixel space**, optionally blur; **each inversion layer** re-selects the k neighbors from that layer’s target features (k and score layer set by CLI). | `baseline_knn_mixup_semantic_init_for_layer` (uses `AnchorLibrary.knn_mixup_pixel`) |

Central dispatcher: **`run_semantic_layer_init_from_method`**. For new experiments, add a new **`baseline_*`** and a new branch there; **avoid editing old `baseline_*` bodies** so historical runs stay comparable.

## Parameters tied to each `method`

- **Shared**: `--anchors-dir`, `--match` (`all` / `cls` / `patch` — token slice before mean-pool), `--match-layer` (`same` or fixed index: where **best** / **knn-mixup** compute similarity / neighbors).
- **`mixup` / `knn-mixup`**: `--mixup-blur-kernel`, `--mixup-blur-sigma` (sigma `0` = no blur after averaging).
- **`knn-mixup` only**: `--knn-k` (neighbor count), `--knn-sim` (`cosine_mean`: normalized dot on mean pools; `neg_l2_mean`: negative squared L2 on raw mean pools).

Inversion itself (independent of `method`): `--layers`, `--steps`, `--lr`, loss weights, `--restarts`. Gray baseline is optional: **`--compare`** (in the shell: `SEMANTIC_COMPARE=1`).

## CLI and script examples

```bash
# Switch method only
python semantic_inversion.py --image path/to.jpg --method best
python semantic_inversion.py --image path/to.jpg --method mixup
python semantic_inversion.py --image path/to.jpg --method knn-mixup --knn-k 5 --knn-sim neg_l2_mean
```

```bash
SEMANTIC_METHOD=best bash run_semantic.sh /path/to.jpg
```

## Outputs and logging

- By default each run writes under `results/runs/<YYYYMMDD-HHMMSS>/` (or pass `--output-dir` for a fixed folder).
- `summary.json` includes **`method`** and **`experiment_description`** (English prose of this run’s settings).
