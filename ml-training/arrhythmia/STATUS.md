# PS5 — Current Status (as of this handoff)

Read alongside `PRD.md` and `IMPLEMENTATION.md` in this same folder. Those two are the fixed spec; this file is the live progress tracker — update it as work continues.

## What Is Done

All 9 phases from `IMPLEMENTATION.md` are **code-complete and verified**, but only on a tiny 2-3 record subset (for correctness, not for real performance). Nothing has been trained on the full dataset yet.

| File | Status |
|---|---|
| `src/config.py` | Complete — AAMI mapping, DS1/DS2 record lists, all hyperparameters |
| `src/preprocess.py` | Complete — verified against real MIT-BIH record 100, 6 passing unit tests |
| `src/dataset_builder.py` | Complete — builds ds1/ds2/intra `.npz`, verified on sample records |
| `src/dataset.py` | Complete — PyTorch `Dataset` with all 5 augmentations |
| `src/model.py` | Complete — CardioNet, 498,453 params, forward pass verified |
| `src/train.py` | Complete — focal loss, AdamW, cosine schedule, early stopping. Smoke-tested for 2 epochs on 2 records (no crash, loss dropped, F1 improved) |
| `src/evaluate.py` | Complete — writes `metrics.json` + confusion matrix. Smoke-tested |
| `src/export_onnx.py` | Complete — export + parity check. Verified PASS (diff = 0.000000) |
| `src/rhythm_rules.py` | Complete — VT/AFib/bigeminy/trigeminy/brady/tachy. 7/7 unit tests pass (`test_rhythm_rules.py`) |
| `src/infer.py` | Complete — full pipeline glue for reference/local testing |
| `src/test_preprocess.py`, `src/test_rhythm_rules.py` | Complete — 13/13 pytest pass |
| **Main repo integration** | Complete — `ml-inference/app/arrhythmia/model.py`, `preprocess.py`, `rhythm_rules.py` wired to load `weights/model.onnx` + `weights/feature_stats.json` automatically. Falls back gracefully to a placeholder response when weights aren't present yet. Verified end-to-end with a mocked CSV download |
| `ml-training/requirements.txt`, `ml-inference/requirements.txt` | Updated with exact pinned versions |

## What Is NOT Done

- No training has happened on the **real, full** DS1 (22 records, tens of thousands of beats) — only 2-3 records were used to smoke-test the code
- No real `metrics.json` / confusion matrix exists yet — the ≥0.85 V-sensitivity target is unverified against real numbers
- No `model.onnx` currently sits in `ml-inference/app/arrhythmia/weights/` — the service will show the "weights not found" fallback until you put one there

---

## Manual Steps — What You Need To Do On Your Machine

Everything below runs on your RTX 5050 machine, not in a Claude session — training is not something an AI chat session can meaningfully do for you.

### 1. Environment setup (one-time)
```bash
cd ml-training
python -m venv venv
venv\Scripts\activate

# torch is installed separately because it needs a build matching your exact GPU/CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```
Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print `True` and your GPU's name.

### 2. Download the full dataset
```bash
cd src
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='../data/mitdb')"
```
This pulls all 48 records (~100MB). Do this once, keep it — do not re-download every session.

### 3. Build the processed dataset
```bash
python dataset_builder.py
```
Watch the printed class distribution — heavy N-class imbalance is expected and correct, not a bug.

### 4. Run the tests (should already pass, this just confirms your environment matches)
```bash
python -m pytest test_preprocess.py test_rhythm_rules.py -v
```

### 5. Train the deployed model (inter-patient, DS1)
```bash
python train.py
```
On your hardware this should take a few minutes for the full 30 epochs (early stopping may cut it shorter). Watch `val_macro_f1` — it should keep improving, not plateau immediately.

### 6. Train the intra-patient comparison model (for the pitch's honesty slide)
```bash
python train.py --intra
```

### 7. Evaluate both
```bash
python evaluate.py
```
This is the moment of truth — check:
- `artifacts/metrics.json` → `inter_patient.per_class.V.sensitivity` should be ≥ 0.85
- `artifacts/confusion_matrix.png` → save this for the pitch deck
- Compare `inter_patient.macro_f1` vs `intra_patient.macro_f1` — intra will look artificially better, that gap **is** the story for judges

### 8. Export to ONNX
```bash
python export_onnx.py
```
Must print `PASS`. If it doesn't, stop and debug before moving on — do not deploy an unverified export.

### 9. Deploy into the main app
```bash
copy artifacts\model.onnx ..\ml-inference\app\arrhythmia\weights\
copy artifacts\feature_stats.json ..\ml-inference\app\arrhythmia\weights\
```
(Use `cp` instead of `copy` if you're on WSL/Mac/Linux.)

### 10. Verify live
Start the ml-inference service:
```bash
cd ../../ml-inference
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Then hit `POST /arrhythmia/predict` with `{"file_url": "<a CSV export of one DS2 record>"}` through the backend, and confirm you get a real `rhythm_type` instead of the "weights not found" fallback message.

---

## What To Hand The Next Claude Session

Paste all three files — `PRD.md`, `IMPLEMENTATION.md`, and this `STATUS.md` — plus tell it which of the 10 manual steps above you've completed. That's the complete context needed to continue.
