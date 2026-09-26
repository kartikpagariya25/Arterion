# PS2 — Heart Pumping Efficiency Estimator — Implementation Plan

Companion to `PRD.md`. Phase-wise build order.

**Shared dataset note:** The raw EchoNet-Dynamic download lives in `ml-training/echonet_data/` — **shared** with PS4 (`lv_outline/`), since both PS use the exact same videos and CSVs. Each PS's `src/` reads from that shared location but writes its own preprocessed tensors to its own `data/processed/`.

---

## Folder Structure

```
ml-training/
  echonet_data/
    raw/                        # the 7GB download, gitignored
    extracted/                  # Videos/, FileList.csv, VolumeTracings.csv — gitignored
  ef_estimation/
    PRD.md
    IMPLEMENTATION.md
    data/
      processed/                 # cached clip tensors, gitignored
    src/
      config.py
      inspect_dataset.py          # Phase 1 — confirm real structure before parsing
      dataset.py                  # video clip sampling + PyTorch Dataset
      model.py                    # R(2+1)D-18 regression head
      train.py
      evaluate.py
      export_onnx.py
      infer.py
    artifacts/
      ef_model.pt / .onnx
      metrics.json
    notebooks/
```

---

## Phase 0 — Environment Setup

Reuses the shared `ml-training/venv/`. Only PS2-specific packages need adding:

```bash
cd ml-training
venv\Scripts\activate
cd ef_estimation
pip install -r requirements.txt
# torch/torchvision already installed correctly (cu128) from PS1 — do NOT reinstall
# unless `python -c "import torch; print(torch.cuda.is_available())"` prints False
```

`requirements.txt` here deliberately does not list torch/torchvision, for the exact reason documented in PS1's `IMPLEMENTATION.md`: any package with a plain `torch` dependency can silently pull a CPU-only wheel if installed after the CUDA build. If you ever need to reinstall torch for any reason, do `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` **last**, after everything else.

**Done when:** `python -c "import torch; print(torch.cuda.is_available())"` still prints `True`.

---

## Phase 1 — Data Extraction & Structure Inspection

```bash
python src\inspect_dataset.py
```

Extracts the 7GB archive into `ml-training/echonet_data/extracted/` (shared with PS4, only done once), then prints:
- The real column names in `FileList.csv` and `VolumeTracings.csv`
- Row counts, the actual Split values used (confirm they really are TRAIN/VAL/TEST as documented)
- One sample video's actual resolution, FPS, and frame count via OpenCV, to confirm it matches what `FileList.csv` claims

**Done when:** real structure is confirmed and matches (or the deviations are noted) before `dataset.py` is written.

---

## Phase 2 — Dataset & Clip Sampling (`src/dataset.py`)

Following the original EchoNet training recipe:
- Sample a fixed-length clip (default 32 frames) from each video at a fixed stride, starting from a random offset (train) or a fixed centered offset (val/test) — this is standard video-model practice, not something to improvise per-run
- Resize frames to 112×112 (a smaller resolution than PS1's 512×512 — video models are far more VRAM-hungry per frame, so resolution is deliberately reduced to fit the 3D convolutions in 8GB)
- Normalize with Kinetics-400 mean/std (since the encoder is Kinetics-pretrained)
- Label: the video's EF value from `FileList.csv`, normalized to [0, 1] for training stability, de-normalized at inference

**Done when:** a batch from the DataLoader has shape `(B, 3, 32, 112, 112)` and the label range matches expected EF percentages (0-100).

---

## Phase 3 — Model Architecture (`src/model.py`)

`torchvision.models.video.r2plus1d_18` (Kinetics-400 pretrained), final FC layer replaced with a single-output regression head.

**Done when:** forward pass on a random batch produces `(B, 1)` output, and a VRAM check at the intended batch size (start at 4, given 3D conv memory cost) does not OOM on an 8GB card — verify this explicitly before Phase 4, exactly as PS1's Phase 4 required.

---

## Phase 4 — Training (`src/train.py`)

| Setting | Value |
|---|---|
| Loss | MSE (this is a regression task, unlike PS1/PS5's classification/segmentation losses) |
| Optimizer | AdamW, lr = 1e-4, weight_decay = 1e-4 |
| Schedule | CosineAnnealingLR |
| Batch size | 4 initially (video models are VRAM-heavy — verified in Phase 3, adjusted if needed) |
| Mixed precision | AMP enabled |
| Augmentations | random crop (small), horizontal flip — kept minimal, since aggressive spatial augmentation can distort the volume cues the model needs for EF |
| Early stopping | Monitor validation MAE, patience 7 |

**Done when:** validation MAE trending downward, logged per epoch, best weights saved.

---

## Phase 5 — Evaluation (`src/evaluate.py`)

MAE, RMSE, R² on the official test split, plus a scatter plot (predicted vs. true EF) saved for the pitch deck. Writes `metrics.json`.

**Done when:** metrics file exists, MAE checked against ≤5.0 target — reported honestly regardless.

---

## Phase 6 — Export & Verification (`src/export_onnx.py`)

Same discipline as PS1/PS5: export, verify PyTorch vs. ONNX Runtime match within a small tolerance on real held-out clips, must print `PASS`.

---

## Phase 7 — Integration (`src/infer.py` → main repo)

`analyze(video_bytes: bytes) -> dict` — decodes the video, samples a clip the same way as training, runs ONNX inference, derives the clinical band, returns the exact contract from `PRD.md` Section 7.

Replaces `ml-inference/app/ef_estimation/model.py`.

**Done when:** `POST /ef-estimation/predict` with a real test-set video URL returns a plausible, non-placeholder EF and classification.

## Build Order Summary

```
Phase 0  Environment              → shared venv confirmed working
Phase 1  Extraction + inspection  → real EchoNet structure confirmed
Phase 2  Dataset + clip sampling  → batch shapes verified
Phase 3  Model architecture       → forward pass + VRAM fit verified
Phase 4  Training                 → ef_model.pt saved
Phase 5  Evaluation               → honest MAE/R² reported
Phase 6  Export                   → PASS
Phase 7  Integration              → wired into main repo
```
