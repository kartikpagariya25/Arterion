# PS1 — Current Status (as of this handoff)

Read alongside `PRD.md` and `IMPLEMENTATION.md` in this same folder.

## What Is Done

All 9 phases are **code-complete and verified**, but only on tiny synthetic/smoke data (a handful of random images per split) for correctness — nothing has been trained on the real, full ARCADE dataset yet.

| File | Status |
|---|---|
| `src/config.py` | Complete — real confirmed ARCADE paths (official train/val/test splits), category mapping (1-25 vessel segments, 26 stenosis) |
| `src/download_data.py` | Complete — resumable, checksum-verified. **Actually run on real data**: 452MB downloaded, checksum verified |
| `src/inspect_dataset.py` | Complete — real structure inspected, confirmed against actual downloaded archive |
| `src/coco_loader.py` | Complete — **run on real ARCADE data**: seg 1000/200/300, stenosis 1000/200/300 (train/val/test), visually spot-checked, masks align perfectly with vessels |
| `src/dataset.py` | Complete — PyTorch Dataset + albumentations augmentations, verified with synthetic data |
| `src/model.py` | Complete — VesselUNet (ResNet34 encoder), 24.4M params, forward pass verified |
| `src/losses.py` | Complete — Dice + focal-Tversky combined loss |
| `src/train.py` | Complete — AMP, cosine schedule, early stopping. Smoke-tested (no crash, loss dropped) |
| `src/evaluate.py` | Complete — Dice/IoU/precision/recall metrics. Smoke-tested |
| `src/severity.py` | Complete — skeleton-based width analysis. **Tested against a synthetic constriction with a known answer: predicted 70.0%, expected 70% — exact match** |
| `src/export_onnx.py` | Complete — export + parity check. Verified PASS on both vessel and stenosis smoke models |
| `src/infer.py` | Complete — full pipeline glue for local testing, verified end-to-end |
| **Main repo integration** | Complete — `ml-inference/app/vessel_analysis/model.py` + `severity.py` wired to load ONNX weights automatically, graceful fallback verified, full pipeline verified with a mocked image download |
| `ml-training/vessel_analysis/requirements.txt`, `ml-inference/requirements.txt` | Updated |

## What Is NOT Done

- No training on the real, full ARCADE dataset (1000 train images per task) — only tiny synthetic smoke tests
- No real `metrics_vessel.json` / `metrics_stenosis.json` exists yet — the Dice targets (≥0.75 vessel, ≥0.60 stenosis) are unverified against real numbers
- No `vessel_model.onnx` / `stenosis_model.onnx` currently sit in `ml-inference/app/vessel_analysis/weights/` — service shows the fallback message until you put them there

## Important Design Note

Unlike PS5 (which had one venv per PS initially), **`ml-training/venv/` is now shared across all problem statements** — torch/torchvision are installed once; each PS's `requirements.txt` only adds its own extra packages on top.

The output contract differs slightly from the original PRD draft: instead of `overlay_image_url`, the wired-up `model.py` returns `overlay_image_base64` (a base64-encoded PNG) — there's no image-hosting/upload step in this ML-only phase of the repo, so returning the image inline is the pragmatic choice for now. When the platform layer (Supabase storage) comes back into the repo at hackathon time, this can be swapped for an actual uploaded URL with a small change to `model.py`.

---

## Manual Steps — What You Need To Do On Your Machine

### 1. Environment (already done, confirm it's still active)
```bash
cd ml-training
venv\Scripts\activate
cd vessel_analysis
```

### 2. Full training (real data, both models)
```bash
cd src
python train.py --task vessel
python train.py --task stenosis
```
Watch `val_dice` — should trend upward. This will take meaningfully longer per epoch than PS5 (image segmentation vs. tiny 1D signals) — expect minutes per epoch, not seconds, on the RTX 5050.

### 3. Evaluate both
```bash
python evaluate.py --task vessel
python evaluate.py --task stenosis
```
Check `artifacts/metrics_vessel.json` and `metrics_stenosis.json` against the PRD's Dice targets.

### 4. Export both to ONNX
```bash
python export_onnx.py --task vessel
python export_onnx.py --task stenosis
```
Both must print `PASS`.

### 5. Deploy into the main app
```bash
copy ..\artifacts\vessel_model.onnx ..\..\..\ml-inference\app\vessel_analysis\weights\
copy ..\artifacts\stenosis_model.onnx ..\..\..\ml-inference\app\vessel_analysis\weights\
```

### 6. Verify live
Same pattern as PS5 — start the ml-inference service, use the `/vessel-analysis/predict` endpoint via Swagger UI with a real ARCADE test image URL (you can serve one locally with `python -m http.server`, same trick as PS5's CSV test).

---

## What To Hand The Next Claude Session

Paste `PRD.md`, `IMPLEMENTATION.md`, and this `STATUS.md`, plus which of the 6 manual steps above are done.
