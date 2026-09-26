# PS3 — Coronary Dominance Classifier — Implementation Plan

Companion to `PRD.md`. Phase-wise build order.

**Critical difference from every other PS in this project:** the dataset is too large (~55GB for the scoped-down portion) for me to download and inspect ahead of time the way ARCADE, MIT-BIH, or even the EchoNet CSVs were. **Phase 1 here is not a formality — it is the first time in this project that neither of us has seen the real internal structure before writing code.** Go slowly through Phase 1, and paste real outputs at each step before moving on.

---

## Folder Structure

```
ml-training/dominance/
  PRD.md
  IMPLEMENTATION.md
  data/
    raw/                        # main_normal.7z, main_others.7z, gitignored
    extracted/                  # gitignored
    processed/                  # gitignored
  src/
    config.py
    download_data.py             # resumable, SHA256-verified (real hashes below)
    inspect_dataset.py            # Phase 1 — mandatory, no shortcuts
    dataset.py
    model.py
    train.py
    evaluate.py
    export_onnx.py
    infer.py
  artifacts/
    dominance_model.pt / .onnx
    metrics.json
  notebooks/
```

---

## Phase 0 — Environment & Disk Check

```bash
cd ml-training/dominance
pip install -r requirements.txt
```

**Before downloading anything, check free disk space** — you need roughly 55GB for the compressed archives plus headroom for extraction (extracted `.npz` files are typically similar size to source, sometimes larger uncompressed; plan for ~120GB free to be safe, and delete the `.7z` files after successful extraction if space is tight):

```powershell
Get-PSDrive F
```

**Done when:** free space confirmed sufficient before Phase 1 begins.

---

## Phase 1 — Download & Structure Inspection (the load-bearing phase)

```bash
python src\download_data.py
```

Downloads only the two required archives (`main_normal.7z`, `main_others.7z`) from Hugging Face, resumable, with SHA256 verification against the real hashes obtained from the HF API:

| File | Size | SHA256 |
|---|---|---|
| `main_normal.7z` | 40.7GB | `9cf9ff25f1393080eae9f23345f281c44368fb4a073bb1246faf2bac4675c285` |
| `main_others.7z` | 14.2GB | `fcf46fb5feee41bab9467ca13c367581375062f87ba38277c5cd806d6c0f26fa` |

This will take a long time depending on connection speed — it's fine to let it run in the background.

Once downloaded:

```bash
python src\inspect_dataset.py
```

This extracts both archives (7z format — uses `py7zr`) and then, **without assuming anything about internal layout**, walks the extracted tree and prints:
- Directory structure (first few levels)
- For any `.npz` file found: its array keys and shapes (`np.load(...).files`, then each array's `.shape` and `.dtype`)
- For any `.csv`/`.json` label file found: its schema, same as every other PS's Phase 1

**Done when:** the real structure is known and pasted into this project's chat — do not write `dataset.py` before this is confirmed. If the structure differs meaningfully from what's assumed in Phase 2 below, that phase gets rewritten to match reality, not the other way around.

---

## Phase 2 — Dataset Construction (written only after Phase 1 confirms real structure)

Expected (to be confirmed, not assumed): per-study `.npz` arrays for RCA/LCA view frames, plus a label file mapping study ID → dominance (and the 5 quality tags). Once confirmed:
- Sample a fixed number of frames per study from the RCA view (per Section 5's approach)
- Resize/normalize consistent with a standard ImageNet-pretrained 2D CNN encoder
- Split: use the dataset's own train/val assignment if the real files provide one; otherwise construct a stratified split (by dominance label) with a fixed seed, same as PS1's approach when ARCADE's official split needed confirming

**Done when:** a DataLoader batch has the expected shape and the label distribution matches the known 319/706 split (or close to it, accounting for excluded bad-quality studies if that's how it's handled).

---

## Phase 3 — Model Architecture (`src/model.py`)

2D CNN (ResNet34 or EfficientNet-B0, ImageNet-pretrained — reusing the same encoder family already proven in this codebase for PS1/PS4, rather than introducing a third architecture family) with a binary classification head.

**Done when:** forward pass produces `(B, 2)` (or `(B, 1)` for binary logit — decided once Phase 2's label encoding is finalized) with no shape errors.

---

## Phase 4 — Training (`src/train.py`)

| Setting | Value |
|---|---|
| Loss | Weighted binary cross-entropy or focal loss, class weights from the real 319/706 (or whatever Phase 1/2 reveals) split — same imbalance-handling discipline as PS5 |
| Optimizer | AdamW, lr = 1e-4 |
| Schedule | CosineAnnealingLR |
| Early stopping | Monitor validation macro-F1, patience 7 |

**Done when:** validation macro-F1 trending upward, best weights saved.

---

## Phase 5 — Evaluation & Export

Accuracy, macro-F1, per-class recall on held-out data — `evaluate.py`. `export_onnx.py` with the same PyTorch-vs-ONNX parity check used everywhere else in this project.

---

## Phase 6 — Integration (`src/infer.py` → main repo)

Replaces `ml-inference/app/dominance/model.py`.

## Build Order Summary

```
Phase 0  Environment + disk check   → confirmed space available
Phase 1  Download + inspection      → REAL structure confirmed (load-bearing)
Phase 2  Dataset construction       → written to match Phase 1's real findings
Phase 3  Model architecture         → forward pass verified
Phase 4  Training                   → dominance_model.pt saved
Phase 5  Evaluation + export        → metrics.json + PASS
Phase 6  Integration                → wired into main repo
```
