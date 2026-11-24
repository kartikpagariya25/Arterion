# PS3 — Coronary Dominance: Status

Trained on this machine (RTX 5050 8 GB), 2026-11-24.

## Data (real structure, inspected)
`data/extracted/anonymous_main_dataset/normal/{Left,Right}_Dominance/<Study>/{RCA,LCA}/*.npz`, key `pixel_array` (T,512,512) uint8.
Only `main_normal` was extracted: **779 studies** (251 left / 528 right), 5,157 videos. (The earlier PRD assumed 1,025 studies incl. `main_others`; those were not used.)

## Method
- `build_cache.py`: 8 evenly spaced frames per video, 192px grayscale cache; stratified study-level split 70/15/15 (545 / 117 / 117 studies, seed 42).
- `DominanceNet`: ImageNet ResNet-18 frame encoder -> per-video feature + view (RCA/LCA) embedding -> MLP -> mean over a study's videos.
- Class-weighted CE (label smoothing 0.05), AdamW, cosine LR, AMP, early stop on val balanced accuracy.

## Result (held-out test, 117 studies)
AUROC 0.9963 · balanced accuracy 0.961 · accuracy 0.966 · confusion [[77,2],[2,36]] (rows = true right/left). See `artifacts/metrics.json`.

## Caveats
- Studies are anonymized; the split is by study, but multiple studies from one patient cannot be ruled out, which would inflate the numbers.
- Single-site data; not evaluated on the dataset's `real_distribution` / `domain_shift` sets.
- Research prototype, not a medical device.
