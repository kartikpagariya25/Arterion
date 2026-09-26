# PS1 — Real-Time Coronary Vessel Analyzer — Implementation Plan

Companion to `PRD.md`. Phase-wise build order. Each phase has a "done when" check — do not move to the next phase until it passes.

**Note on this plan vs. PS5's:** PS5's Phase 1 could hardcode exact record lists because MIT-BIH's structure is well-documented and stable. ARCADE's exact archive layout is confirmed by actually looking at it — Phase 1 below is deliberately an inspection step before any parsing code is written, to avoid guessing wrong paths.

---

## Folder Structure

```
ml-training/vessel_analysis/
  PRD.md
  IMPLEMENTATION.md
  data/
    raw/                       # downloaded ARCADE archives, gitignored
    segmentation/              # extracted, inspected structure — filled in Phase 1
    stenosis/                  # extracted, inspected structure — filled in Phase 1
    processed/                 # preprocessed tensors/masks, gitignored
  src/
    config.py
    download_data.py            # Zenodo API downloader, resumable (same pattern as PS5)
    inspect_dataset.py          # Phase 1 only — prints real structure, does not assume it
    coco_loader.py               # COCO-annotation parsing, written after Phase 1 confirms paths
    dataset.py                   # PyTorch Dataset for segmentation masks
    model.py                     # U-Net (shared architecture, two independent weight sets)
    train.py                     # trains Model A or Model B via --task flag
    evaluate.py
    severity.py                  # skeleton-based width analysis at lesion location
    export_onnx.py
    infer.py
  artifacts/
    vessel_model.pt / .onnx
    stenosis_model.pt / .onnx
    metrics_vessel.json
    metrics_stenosis.json
  notebooks/
```

Nothing under `data/` or `*.pt` is committed — only ONNX exports and metrics JSON.

---

## Phase 0 — Environment Setup

This project uses **one shared venv for all of `ml-training/`** (not a separate venv per PS) — torch/torchvision are large, GPU-specific installs, and there's no reason to redownload them for every problem statement. If `ml-training/venv/` already exists (from PS5), skip straight to installing this PS's extra packages.

```bash
cd ml-training
venv\Scripts\activate          # reuse the existing shared venv

cd vessel_analysis
pip install -r requirements.txt   # torch/torchvision are NOT in here — only PS1-specific packages

# IMPORTANT: install/reinstall torch+torchvision LAST, after requirements.txt.
# Some packages (segmentation-models-pytorch, via timm) list "torch" as a plain
# PyPI dependency with no CUDA pin — if pip resolves that after your GPU build
# is already installed, it can silently replace it with a CPU-only wheel from
# default PyPI. Installing the CUDA build last guarantees it's the one that sticks.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

If the shared venv doesn't exist yet (first PS being set up):
```bash
cd ml-training
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**pycocotools check (do this before anything else depends on it):**
```bash
python -c "import pycocotools; print('ok')"
```
If this fails on Windows, try:
```bash
pip uninstall pycocotools -y
pip install pycocotools-windows
```
and re-verify before proceeding.

**Done when:** CUDA check prints `True` + GPU name, and the pycocotools import prints `ok`.

---

## Phase 1 — Data Acquisition & Structure Inspection

`src/download_data.py` hits the Zenodo API for record `8386059`, lists every file with its real name and size, and downloads each with progress + resume (same resumable pattern that fixed PS5's "stuck download" problem) into `data/raw/`.

```bash
python download_data.py
```

Then, **before writing any parsing code**:
```bash
python inspect_dataset.py
```
This script (extracts the archives if not already extracted, then) prints the actual directory tree and the actual keys inside any COCO-format JSON it finds (`categories`, sample `annotations[0]`, sample `images[0]`). It does not assume field names — it reports what's really there.

**Done when:** the real folder structure and COCO JSON schema are known and written into a short note at the top of `coco_loader.py` before that file's parsing logic is implemented — so a future session (or a Claude picking this back up) doesn't have to re-discover it.

---

## Phase 2 — COCO Annotation Loading & Mask Generation (`src/coco_loader.py`)

Only written after Phase 1's real structure is confirmed. Converts COCO polygon/RLE annotations into binary mask images:

- Segmentation subtask → collapse all vessel-segment category IDs into a single foreground class (binary mask)
- Stenosis subtask → binary mask of stenotic regions from their annotated polygons/boxes

Both saved as preprocessed `(image, mask)` pairs (resized to 512×512, normalized) under `data/processed/{segmentation,stenosis}/`.

**Done when:** a handful of generated masks are visually spot-checked (saved as overlay PNGs) and clearly align with the vessel/lesion locations in the source image.

---

## Phase 3 — Dataset Split

- 80/10/10 train/val/test split, stratified by whether an image contains at least one annotated lesion (for the stenosis subtask, so the small test set isn't accidentally all-negative or all-positive)
- Same split logic reused for both subtasks independently (they're separate image sets)
- Split indices saved to `data/processed/{segmentation,stenosis}_split.json` so re-runs are reproducible

**Done when:** split file exists, class balance per split printed and sane (no split with zero positive lesion images).

---

## Phase 4 — Model Architecture (`src/model.py`)

**U-Net with a ResNet34 encoder** (ImageNet-pretrained), used identically for both Model A (vessel) and Model B (stenosis) — same architecture, independently trained weights, since they're different binary segmentation problems on different image sets.

```
Encoder: ResNet34 (pretrained), stages give skip connections at 4 resolutions
Decoder: 4 up-sampling blocks (transpose conv + concat skip + double conv)
Head: Conv 1x1 -> 1 channel -> sigmoid (binary mask)
Input: 512x512x3 (angiogram frames are grayscale but stored as 3-channel for the pretrained encoder)
Output: 512x512x1 probability mask
```

**Done when:** a forward pass on a random batch produces output shape `(B, 1, 512, 512)` with no shape errors, and a VRAM check at the intended batch size does not OOM on an 8GB card (verify this explicitly — this is the PS1-specific version of the PS5 CUDA-build lesson: check hardware fit before committing to a full run).

---

## Phase 5 — Training (`src/train.py`)

| Setting | Value |
|---|---|
| Seed | 42 |
| Loss | 0.5 × Dice loss + 0.5 × focal-Tversky loss (alpha=0.7, beta=0.3, gamma=1.33 — tuned toward recall, since missing a stenosis is worse than a false positive in this demo's framing) |
| Optimizer | AdamW, lr = 1e-4, weight_decay = 1e-4 |
| Schedule | CosineAnnealingLR |
| Batch size | Start at 8 for 512×512 on 8GB VRAM; reduce to 4 with gradient accumulation ×2 if Phase 4's VRAM check shows pressure |
| Mixed precision | `torch.cuda.amp` enabled — meaningfully reduces memory pressure at this resolution |
| Augmentations | horizontal flip, small rotation (±10°), brightness/contrast jitter, elastic deformation (mild) — standard for angiography, avoids anything that would distort vessel topology (no aggressive shearing) |
| Early stopping | Monitor validation Dice, patience 7 |

`python train.py --task vessel` and `python train.py --task stenosis` train the two models independently.

**Done when:** both runs complete, validation Dice trending upward and logged per epoch, best weights saved to `artifacts/{vessel,stenosis}_model.pt`.

---

## Phase 6 — Evaluation (`src/evaluate.py`)

Computes Dice, IoU, precision, recall on the held-out test split for both models, writes `metrics_vessel.json` and `metrics_stenosis.json` with the same honesty discipline as PS5 (report real numbers, note known weak points rather than hide them).

**Done when:** both metrics files exist; vessel Dice checked against ≥0.75, stenosis Dice against ≥0.60.

---

## Phase 7 — Severity Estimation (`src/severity.py`)

Given a vessel mask and a stenosis mask/bbox:
1. Skeletonize the vessel mask (`skimage.morphology.skeletonize`)
2. Walk the skeleton to get a local vessel-width profile (distance-transform value at each skeleton point)
3. At the stenosis location, find the minimum local width
4. Compare it to the mean width in a window just proximal and distal to the lesion (reference "normal" width)
5. `stenosis_percent = (1 - min_width / reference_width) * 100`, clamped to [0, 100]

**Done when:** tested on a handful of known-stenotic training images and produces plausible percentages (not validated against a ground-truth percentage, since ARCADE doesn't provide one — this is explicitly disclosed, per the PRD).

---

## Phase 8 — Export & Verification (`src/export_onnx.py`)

Same discipline as PS5: export both models to ONNX, verify PyTorch vs ONNX Runtime outputs match within 1e-4 on real held-out images.

**Done when:** both exports print `PASS`.

---

## Phase 9 — Integration (`src/infer.py` → main repo)

`analyze(image: np.ndarray) -> dict` — runs both ONNX models, generates the overlay image, runs severity estimation, returns the exact contract from `PRD.md` Section 6.

Replaces `ml-inference/app/vessel_analysis/model.py`, loading `vessel_model.onnx` + `stenosis_model.onnx` from `ml-inference/app/vessel_analysis/weights/`.

**Done when:** `POST /vessel-analysis/predict` in the main repo, called with a real ARCADE test image, returns a populated, non-placeholder response with a plausible overlay and severity estimate.

---

## Testing Checklist

- [ ] `test_coco_loader.py` — mask generation matches known annotation on a hand-checked sample
- [ ] `test_severity.py` — width-ratio formula on a synthetic mask with a known constriction
- [ ] `test_infer.py` — end-to-end on one real test image, output matches contract schema
- [ ] ONNX vs PyTorch parity check passes for both models

## Build Order Summary

```
Phase 0  Environment setup           → CUDA + pycocotools confirmed working
Phase 1  Data acquisition            → real ARCADE structure inspected and documented
Phase 2  COCO parsing + masks        → visually verified
Phase 3  Dataset split               → reproducible, balance checked
Phase 4  Model architecture          → forward pass + VRAM fit verified
Phase 5  Training (both models)      → vessel_model.pt, stenosis_model.pt saved
Phase 6  Evaluation                  → honest metrics for both
Phase 7  Severity estimation         → plausible on known samples
Phase 8  Export                      → both ONNX exports PASS
Phase 9  Integration                 → wired into main repo's ml-inference service
```
