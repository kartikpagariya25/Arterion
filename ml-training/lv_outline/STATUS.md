# PS4 — Current Status (as of this handoff)

Read alongside `PRD.md` and `IMPLEMENTATION.md`.

## What Is Done

| File | Status |
|---|---|
| `src/config.py` | Complete — shared EchoNet paths, IMAGE_SIZE=112 |
| `src/build_masks.py` | Complete — **run on your real data**: 14,920 train / 2,576 val / 2,552 test masks generated from the real `VolumeTracings.csv`. **Visually spot-checked by you** — 5 real overlay screenshots confirmed the mask correctly traces the left ventricle shape |
| `src/dataset.py` | Complete — reuses PS1's augmentation recipe, verified |
| `src/model.py` | Complete — imports `VesselUNet` from PS1 as `LVUNet` (collision-free cross-import, fixed after an initial `sys.path` bug caught during smoke testing) |
| `src/train.py` | Complete — reuses PS1's loss functions via the same cross-import pattern. Smoke-tested successfully after the import fix |
| `src/evaluate.py` | Complete — Dice/IoU/precision/recall. Smoke-tested |
| `src/export_onnx.py` | Complete — PASS on smoke model |
| `src/infer.py` | Complete |
| **Main repo integration** | Complete — `ml-inference/app/lv_outline/model.py` wired, graceful fallback verified, full pipeline verified with a mocked image download |

## What Is NOT Done

- No training on the real, full mask dataset (14,920 train images) — only tiny synthetic smoke tests
- No real `metrics.json` — Dice ≥0.90 target unverified against real numbers
- No `lv_model.onnx` in `ml-inference/app/lv_outline/weights/` yet

## Note: A Real Bug Was Caught and Fixed During This Build

Initial `model.py`/`train.py` used `sys.path.insert(0, ...)` to import PS1's `VesselUNet` and loss functions. This caused a **module name collision** — both PS1 and PS4 have their own `dataset.py`/`config.py`, and inserting PS1's directory at the front of `sys.path` made Python import PS1's `dataset.py` instead of PS4's own, crashing with a confusing `ImportError`. Fixed by switching to `importlib.util.spec_from_file_location` for the specific PS1 modules needed (`model.py`, `losses.py`), which loads them without polluting the shared module namespace. Verified working after the fix.

---

## Manual Steps

### 1. Train
```bash
cd ml-training/lv_outline/src
python train.py
```
This is 2D per-frame segmentation like PS1 — similar per-epoch time, should be noticeably faster than PS2's video model.

### 2. Evaluate
```bash
python evaluate.py
```
Check `artifacts/metrics.json` against the ≥0.90 Dice target.

### 3. Export
```bash
python export_onnx.py
```

### 4. Deploy
```bash
copy ..\artifacts\lv_model.onnx ..\..\..\ml-inference\app\lv_outline\weights\
```

### 5. Verify live
Same Swagger UI pattern — `POST /lv-outline/predict` with a real test-set frame image URL.

---

## What To Hand The Next Claude Session

Paste `PRD.md`, `IMPLEMENTATION.md`, this `STATUS.md`, and which manual steps are done.
