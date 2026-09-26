# PS2 — Current Status (as of this handoff)

Read alongside `PRD.md` and `IMPLEMENTATION.md`.

## What Is Done

All phases are **code-complete and verified**, but only with synthetic random-noise videos (I don't have access to your gated EchoNet-Dynamic download — Stanford's Research Use Agreement is tied to your account, so I could not download and verify against real data myself, unlike PS1/PS5). Real structure was confirmed from your `inspect_dataset.py` output, so the code is written against confirmed real schema — but has not been trained on real cardiac videos yet.

| File | Status |
|---|---|
| `src/config.py` | Complete — shared EchoNet paths, CLIP_LENGTH=32, FRAME_SIZE=112 (matches native video resolution — no resize needed) |
| `src/inspect_dataset.py` | Complete — **run on your real data**, confirmed FileList.csv/VolumeTracings.csv schema, official split (7465/1288/1277) |
| `src/dataset.py` | Complete — clip sampling, Kinetics normalization. Smoke-tested with synthetic `.avi` files matching the real schema |
| `src/model.py` | Complete — R(2+1)D-18 (Kinetics-pretrained), 31.3M params, forward pass verified |
| `src/train.py` | Complete — MSE loss, AMP, early stopping on MAE. Smoke-tested (no crash) |
| `src/evaluate.py` | Complete — MAE/RMSE/R² + scatter plot. Smoke-tested |
| `src/export_onnx.py` | Complete — PASS on smoke model |
| `src/infer.py` | Complete — full pipeline glue |
| **Main repo integration** | Complete — `ml-inference/app/ef_estimation/model.py` wired, graceful fallback verified, full pipeline verified with a mocked video download |

## What Is NOT Done

- No training on real EchoNet videos — only synthetic random-noise smoke tests (structurally correct, numerically meaningless)
- No real `metrics.json` — MAE ≤5.0 / R² ≥0.75 targets unverified against real numbers
- No `ef_model.onnx` in `ml-inference/app/ef_estimation/weights/` yet

---

## Manual Steps

### 1. Confirm environment (shared venv, torch already correct from PS1 fix)
```bash
cd ml-training/ef_estimation
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Train
```bash
cd src
python train.py
```
**This will be the slowest of all 5 PS to train** — 3D video convolutions over 7,465 training videos. Expect noticeably longer per-epoch time than PS1. Watch `val_mae` — should trend downward from a large initial value toward single digits.

### 3. Evaluate
```bash
python evaluate.py
```
Check `artifacts/metrics.json` (MAE, RMSE, R²) and `artifacts/scatter_plot.png` for the pitch deck.

### 4. Export
```bash
python export_onnx.py
```
Must print `PASS`.

### 5. Deploy
```bash
copy ..\artifacts\ef_model.onnx ..\..\..\ml-inference\app\ef_estimation\weights\
```

### 6. Verify live
Same Swagger UI pattern as PS1/PS5 — `POST /ef-estimation/predict` with a real test-video URL (serve one locally with `python -m http.server` from the `Videos/` folder, same trick as before).

---

## What To Hand The Next Claude Session

Paste `PRD.md`, `IMPLEMENTATION.md`, this `STATUS.md`, and which manual steps are done.
