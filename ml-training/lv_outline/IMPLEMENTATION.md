# PS4 — Left Ventricle Outline Tool — Implementation Plan

Companion to `PRD.md`. Phase-wise build order.

**Shared dataset note:** reads from `ml-training/echonet_data/extracted/` — the same extraction PS2 performs. If PS2's Phase 1 has already run, do not re-extract; just confirm the path exists.

---

## Folder Structure

```
ml-training/lv_outline/
  PRD.md
  IMPLEMENTATION.md
  data/
    processed/                  # images/ + masks/ built from VolumeTracings.csv, gitignored
  src/
    config.py
    build_masks.py               # VolumeTracings.csv -> binary mask images (Phase 2)
    dataset.py                    # reuses the same pattern as PS1's SegmentationDataset
    model.py                      # imports VesselUNet from PS1 rather than duplicating it
    train.py
    evaluate.py
    export_onnx.py
    infer.py
  artifacts/
    lv_model.pt / .onnx
    metrics.json
  notebooks/
```

**On reusing PS1's model code:** `model.py` here imports `VesselUNet` from `ml-training/vessel_analysis/src/model.py` rather than copy-pasting the class — one architecture, two independently trained weight sets, consistent with how PS1 itself already treats its own vessel vs. stenosis models.

---

## Phase 0 — Environment Setup

Same shared `ml-training/venv/`. This PS needs no new packages beyond what PS1 already installed (U-Net + ResNet34 encoder, OpenCV, scikit-image are already present).

**Done when:** `python -c "from segmentation_models_pytorch import Unet"` succeeds without reinstalling anything.

---

## Phase 1 — Structure Confirmation

If PS2's `inspect_dataset.py` has already run, this phase is just: confirm `VolumeTracings.csv`'s real column names and coordinate format (points are stored as a variable number of rows per `(FileName, Frame)` pair — confirmed against the actual file, not assumed, before `build_masks.py` is written).

**Done when:** the real `VolumeTracings.csv` schema is written into a short note at the top of `build_masks.py`.

---

## Phase 2 — Mask Generation (`src/build_masks.py`)

For each `(FileName, Frame)` pair in `VolumeTracings.csv` that corresponds to a labeled ED or ES frame:
1. Extract that exact frame from the corresponding video (OpenCV, by frame index)
2. Reconstruct the LV boundary polygon from the point coordinates
3. Fill the polygon to produce a binary mask (`cv2.fillPoly`)
4. Save `(frame_image, mask)` pairs to `data/processed/{train,val,test}/` — split assignment taken from `FileList.csv`'s official `Split` column (same official split as PS2, for consistency)

A visual spot-check overlay is generated for a handful of frames and inspected before proceeding — same discipline as PS1's Phase 2.

**Done when:** spot-check overlays show the mask correctly outlining the left ventricle chamber in the source frame.

---

## Phase 3 — Dataset & Model

`dataset.py` mirrors PS1's `SegmentationDataset` exactly (same augmentation recipe — it's already a well-tested pattern in this codebase). `model.py` imports `VesselUNet` unmodified.

**Done when:** a forward pass produces the expected `(B, 1, 512, 512)` shape (or whatever resolution the real frames turn out to be, confirmed in Phase 1).

---

## Phase 4 — Training (`src/train.py`)

Same loss (Dice + focal-Tversky), optimizer, and schedule as PS1's `train.py` — reused, not reinvented, since the task family (binary segmentation) is identical. The LV's simpler, blob-like shape means this should converge faster and higher than PS1's vessel model.

**Done when:** validation Dice trending toward the ≥0.90 target, best weights saved.

---

## Phase 5 — Evaluation & Export

Same pattern as PS1: `evaluate.py` computes Dice/IoU/precision/recall on the official test split's labeled frames, `export_onnx.py` exports and verifies parity.

**Done when:** both `metrics.json` and a `PASS` from the ONNX check exist.

---

## Phase 6 — Integration (`src/infer.py` → main repo)

`analyze(image: np.ndarray) -> dict` — runs the ONNX model, returns the mask + area + the contract from `PRD.md` Section 7. Replaces `ml-inference/app/lv_outline/model.py`.

**Done when:** `POST /lv-outline/predict` with a real test-set frame returns a plausible, non-placeholder outline.

## Build Order Summary

```
Phase 0  Environment              → confirmed, no reinstall needed
Phase 1  Structure confirmation   → VolumeTracings.csv schema confirmed
Phase 2  Mask generation          → visually spot-checked
Phase 3  Dataset + model          → forward pass verified
Phase 4  Training                 → lv_model.pt saved
Phase 5  Evaluation + export      → metrics.json + PASS
Phase 6  Integration              → wired into main repo
```
