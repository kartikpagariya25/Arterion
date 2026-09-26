# PS5 — Heart Rhythm Abnormality Detector — Implementation Plan

Companion to `PRD.md`. This is the exact, phase-wise build order. Each phase has a "done when" check — do not move to the next phase until the current one passes it.

---

## Folder Structure

```
ml-training/arrhythmia/
  PRD.md
  IMPLEMENTATION.md
  data/
    mitdb/                    # raw PhysioNet download (gitignored)
    processed/                # ds1.npz, ds2.npz, intra.npz (gitignored)
  src/
    config.py                 # paths, record lists, hyperparameters, seeds
    preprocess.py              # resample, filter, R-peak segmentation, RR features
    dataset.py                 # PyTorch Dataset + DataLoader wrapping the .npz files
    model.py                   # CardioNet architecture
    train.py                   # training loop (inter-patient + --intra flag)
    evaluate.py                 # writes metrics.json + confusion_matrix.png
    export_onnx.py              # export + numerical verification
    rhythm_rules.py             # beat sequence -> rhythm-level alerts
    infer.py                    # single entrypoint: raw signal -> full output contract
  artifacts/
    model.pt
    model.onnx
    feature_stats.json
    metrics.json
    confusion_matrix.png
  notebooks/
    exploration.ipynb          # scratch work, not part of the pipeline
```

Nothing under `data/` or `*.pt` is committed to git — only `artifacts/model.onnx`, `feature_stats.json`, and `metrics.json` are (they're small and the whole point is reproducible inference without retraining).

---

## Phase 0 — Environment Setup

```bash
cd ml-training/arrhythmia
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build for RTX 5050
pip install wfdb neurokit2 scipy numpy scikit-learn onnx onnxruntime matplotlib
```

**Done when:** `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

---

## Phase 1 — Data Acquisition

`src/config.py` defines `DATA_DIR = "data/mitdb"`. One-time download script:

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/mitdb')
```

**Done when:** `data/mitdb/` contains 48 records (`.dat`, `.hea`, `.atr` files each).

---

## Phase 2 — Signal Preprocessing (`src/preprocess.py`)

Implemented once, imported by both training (`train.py`) and inference (`infer.py`) so they can never drift apart.

| Step | Spec |
|---|---|
| Lead selection | `MLII` channel (present in all but 2 records; those use `V1` as fallback documented in config) |
| Resample | Any non-360Hz input → 360 Hz via `scipy.signal.resample_poly` |
| Bandpass filter | Butterworth order 4, 0.5–40 Hz, zero-phase (`sosfiltfilt`) |
| R-peak detection | Training: ground-truth annotation sample positions. Inference: `neurokit2.ecg_peaks(method='neurokit')`, discard peaks < 200ms apart |
| Beat window | 256 samples: 90 before R-peak, 166 after (R-peak at index 90); skip beats where the window falls outside signal bounds |
| Beat normalization | Per-beat z-score: `(x - mean) / (std + 1e-8)` |
| RR features (4 values, seconds) | `rr_prev`, `rr_next`, `rr_local` (mean of previous ≤10 RR intervals), `rr_ratio = rr_prev / rr_local` |
| RR feature scaling | Standardized using mean/std computed on DS1 training beats only, saved to `artifacts/feature_stats.json` |

**Done when:** unit test confirms a known record produces the correct beat count and window shape `(256,)`.

---

## Phase 3 — Dataset Construction & Inter-Patient Split (`src/config.py` + a data-prep script)

**AAMI EC57 mapping** (as in PRD Section 5) applied while building beat labels.

**Excluded (paced) records:** 102, 104, 107, 217 — never used.

| Set | Records |
|---|---|
| DS1 (train + val) | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 |
| DS2 (test — never seen in training) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 |

- Validation = stratified random 10% of DS1 beats (seed 42) — used only for early stopping.
- Output: `data/processed/{ds1,ds2}.npz` with arrays `beats (n,256)`, `rr (n,4)`, `labels (n,)`, `record (n,)`.
- Separately, for comparison only: an `intra.npz` built from an 80/20 random split of DS1+DS2 combined, to demonstrate in the pitch how much a naive split would have inflated the numbers.

**Done when:** `.npz` files exist, class distribution printed (expect heavy N-class imbalance — this is real and expected, not a bug).

---

## Phase 4 — Model Architecture (`src/model.py`)

**CardioNet** — dual-input 1D ResNet (waveform) + small MLP (RR features), fused before the classification head.

```
Input A: beat waveform (B, 1, 256)
Input B: RR features    (B, 4)

Waveform branch (1D ResNet):
  Stem:    Conv1d(1, 32, k=7) -> BN -> ReLU
  Block 1: ResBlock(32  -> 32,  stride 1)
  Block 2: ResBlock(32  -> 64,  stride 2)
  Block 3: ResBlock(64  -> 128, stride 2)
  Block 4: ResBlock(128 -> 128, stride 2)   <- Grad-CAM target layer
  AdaptiveAvgPool1d(1) -> (B, 128)

RR branch:
  Linear(4, 16) -> ReLU -> (B, 16)

Head:
  concat -> (B, 144) -> Linear(144, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 5)
```

`forward(beat: Tensor, rr: Tensor) -> Tensor` (raw logits over `["N","S","V","F","Q"]`).

**Done when:** a forward pass on a random batch produces the expected output shape `(B, 5)` with no shape errors.

---

## Phase 5 — Training (`src/train.py`)

| Setting | Value |
|---|---|
| Seed | 42 (Python, NumPy, PyTorch) |
| Loss | Focal loss, gamma = 2.0, per-class alpha = inverse-sqrt of class frequency in DS1 training beats, normalized to sum to 5 |
| Optimizer | AdamW, lr = 1e-3, weight_decay = 1e-4 |
| Schedule | CosineAnnealingLR, max 30 epochs |
| Batch size | 256 |
| Early stopping | Monitor validation macro-F1, patience 5, restore best weights |
| Augmentations (train only, p=0.5 each) | amplitude scale ×0.9–1.1; Gaussian noise (std 0.01–0.05 on z-scored beat); time shift ±5 samples; baseline wander (sine 0.1–0.5Hz, amp ≤0.1); moving-average smoothing k=3 (p=0.3); resample jitter ±5% |

On an RTX 5050 with this dataset size and model size, expect full 30-epoch training in a few minutes.

`python train.py` trains the deployed (inter-patient, DS1-trained) model. `python train.py --intra` trains the comparison model on the intra-patient split for the pitch's "honesty" slide.

**Done when:** `artifacts/model.pt` exists, validation macro-F1 logged per epoch and improving.

---

## Phase 6 — Evaluation (`src/evaluate.py`)

Evaluates the DS1-trained model on DS2 (and separately, the intra-patient model on its own held-out split). Writes `artifacts/metrics.json`:

```json
{
  "classes": ["N", "S", "V", "F", "Q"],
  "inter_patient": {
    "per_class": { "N": {"sensitivity": 0.0, "ppv": 0.0, "f1": 0.0, "support": 0} },
    "macro_f1": 0.0,
    "accuracy": 0.0,
    "confusion_matrix": [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
  },
  "intra_patient": { "...same shape..." },
  "dataset": { "name": "MIT-BIH Arrhythmia Database", "excluded_records": [102, 104, 107, 217] },
  "model": { "name": "CardioNet (1D ResNet + RR features)", "params": 0 },
  "generated_at": "ISO-8601"
}
```

Also saves `artifacts/confusion_matrix.png` for the pitch deck.

**Done when:** `metrics.json` written; V-class sensitivity on DS2 checked against the ≥0.85 target (report the real number regardless — never tune against DS2).

---

## Phase 7 — Export & Verification (`src/export_onnx.py`)

- Export `model.pt` → `artifacts/model.onnx` (opset 17, input names `beat` and `rr`, output name `logits`, dynamic batch axis).
- Verify: run 100 random DS2 beats through both PyTorch and ONNX Runtime; logits must match within 1e-4.

**Done when:** script prints `PASS`.

---

## Phase 8 — Rhythm Rule Engine (`src/rhythm_rules.py`)

Runs after beat classification, converts the beat sequence into rhythm-level alerts (this is what actually makes the demo output read like a clinical finding instead of a list of per-beat labels).

| Alert | Rule |
|---|---|
| VT (Ventricular Tachycardia) | ≥3 consecutive V beats, every RR inside the run < 0.60s |
| AFIB | Irregular RR + absent P-wave proxy (approximated via RR variability threshold, tuned on DS1 records with AFib rhythm annotations) |
| BIGEMINY | N,V pattern repeated ≥3 cycles |
| TRIGEMINY | N,N,V pattern repeated ≥3 cycles |
| TACHY | Rolling 10s mean HR > 100bpm (warning) / >150bpm (critical) |
| BRADY | Rolling 10s mean HR < 50bpm |

**Done when:** unit tests on synthetic beat sequences (as in PRD risk section) produce the expected alert for each rule.

---

## Phase 9 — Integration into Main Repo (`src/infer.py`)

Single function: `analyze(signal: np.ndarray, sampling_rate: int) -> dict` that runs Phase 2 preprocessing → Phase 4 model (ONNX Runtime) → Phase 8 rhythm rules → returns the exact output contract from `PRD.md` Section 7.

Replace the placeholder in the main CardioSense repo:
`ml-inference/app/arrhythmia/model.py` → imports `analyze()` from this training repo's exported package, loads `model.onnx` + `feature_stats.json` from `ml-inference/app/arrhythmia/weights/`.

**Done when:** calling `POST /arrhythmia/predict` in the main repo with a real MIT-BIH test-record CSV returns a populated, non-placeholder response matching the contract.

---

## Testing Checklist (minimum, before calling this phase done)

- [ ] `test_preprocess.py` — beat window shape, RR feature formula, resampling correctness
- [ ] `test_rhythm_rules.py` — synthetic sequences for VT / bigeminy / trigeminy / brady / tachy
- [ ] `test_infer.py` — end-to-end on one DS2 record, output matches contract schema
- [ ] ONNX vs PyTorch parity check passes

## Build Order Summary

```
Phase 0  Environment setup           → CUDA confirmed
Phase 1  Data acquisition            → mitdb/ downloaded
Phase 2  Preprocessing               → unit-tested
Phase 3  Dataset + split             → ds1.npz, ds2.npz built
Phase 4  Model architecture          → forward pass verified
Phase 5  Training                    → model.pt saved
Phase 6  Evaluation                  → metrics.json honest numbers
Phase 7  Export                      → model.onnx, PASS verification
Phase 8  Rhythm rules                → alert engine tested
Phase 9  Integration                 → wired into main repo's ml-inference service
```
