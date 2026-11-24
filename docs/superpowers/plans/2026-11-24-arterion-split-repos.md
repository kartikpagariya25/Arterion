# Arterion Split Repos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Arterion monorepo's five cardiac models into five self-contained public GitHub repos, each with its own distinct website, real trained weights, and honest metrics.

**Architecture:** Per repo, copy and trim the existing code (approach A): `frontend/` (static, own design) + `backend/` (FastAPI, one model, same `POST /uploads` → `POST /<task>/predict {file_url}` contract) + `training/` + `scripts/fetch_weights`. Weights ship as GitHub Release assets. Four GPU training jobs run sequentially in the background while repos are built.

**Tech Stack:** Python 3.11, PyTorch 2.11+cu128 (shared venv `F:\cardiosense\ml-training\venv`), ONNX Runtime, FastAPI, pytest, vanilla HTML/CSS/JS (+GSAP/Chart where a design needs it), nginx, Docker Compose, `gh` CLI.

**Spec:** `docs/superpowers/specs/2026-11-24-arterion-split-repos-design.md`

## Global Constraints

- Attribution: commits/pushes/repos/releases use the existing git identity (Kartik Pagariya); **no `Co-Authored-By` or "Generated with Claude" text anywhere**.
- New repos live in `F:\cardiosense-repos\<name>`, separate from the monorepo; each is its own `git init`.
- Repo names (public, under `kartikpagariya25`): `arterion-vessel-analysis`, `arterion-ef-estimation`, `arterion-lv-outline`, `arterion-coronary-dominance`, `arterion-arrhythmia`. Confirm with the owner immediately before each `gh repo create`.
- No file >100 MB in git; `*.onnx`, `*.pt`, `*.pth`, datasets, `*.db`, `.env`, `venv/` are gitignored. Weights go to a GitHub Release.
- Run `gh` with `GITHUB_TOKEN` unset in agent shells (`env -u GITHUB_TOKEN gh ...`).
- GPU: one RTX 5050 8 GB; run one training job at a time; keep batch size within 8 GB.
- Published metrics must come from the full held-out test split; targets (stenosis Dice ≥0.60, EF R² ≥0.75) are goals — if missed, README states the achieved numbers.
- Each site shows "Research prototype — not a medical device."
- Frontends: verify at desktop and 375 px width; 16 px side gutters; no horizontal scroll.
- Licence: MIT for code; note dataset licences in README. EchoNet terms may restrict publishing derived weights — decide before releasing EF/LV weights (Task 14).

## Review Focus

- Wrong media type sent to an endpoint (video to an image model, image to the ECG endpoint) → HTTP 4xx with a readable message, never a 500 or blank UI.
- Empty or oversized upload (cap 200 MB) → 400 / 413 with message.
- Weights not fetched → API returns the existing "weights not found" fallback and the UI shows a visible notice, not an empty result.
- ECG CSV with extra columns, header row, or <10 s of signal → clear error or best-effort result with a warning.
- Echo video with fewer than 32 frames or non-112×112 size (EF) → still predicts (pad/loop/resize) or returns a clear message.
- Dominance study with only one view (RCA or LCA) → still predicts; response states which views were used.
- Frontend pointed at a backend on another host (`API_BASE` configurable via `config.js`, CORS allowed).

---

## Phase A — Setup

### Task 0: Workspace and monorepo housekeeping

**Files:**
- Create: `F:\cardiosense-repos\` (directory)
- Modify: monorepo git remote

**Interfaces:**
- Produces: `F:\cardiosense-repos\` as the parent of all five repos.

- [ ] **Step 1:** `mkdir F:\cardiosense-repos`.
- [ ] **Step 2:** In `F:\cardiosense`: `git remote set-url origin https://github.com/kartikpagariya25/Arterion.git`; verify with `git remote -v`.
- [ ] **Step 3:** `env -u GITHUB_TOKEN gh auth status` shows logged in as `kartikpagariya25`; `env -u GITHUB_TOKEN gh repo view kartikpagariya25/Arterion` succeeds.

---

## Phase B — Training (sequential on GPU; start Task 1 first, run in background)

All commands use `F:\cardiosense\ml-training\venv\Scripts\python.exe`. Each task ends with `artifacts/metrics.json` from the full test split and an ONNX file that passes `export_onnx.py` parity.

### Task 1: Train coronary dominance (new)

**Files:**
- Modify: `ml-training/dominance/src/inspect_dataset.py` (confirm real `.npz` keys/shapes under `data/extracted/anonymous_main_dataset`, record in `STATUS.md`)
- Create: `ml-training/dominance/src/dataset.py`, `model.py`, `train.py`, `evaluate.py`, `export_onnx.py`, `infer.py`, `STATUS.md`
- Test: `ml-training/dominance/src/test_dataset.py`

**Interfaces:**
- Produces: `artifacts/dominance_model.onnx` (input `frames` float32 `[1,N,3,224,224]`, output `logits` `[1,2]`; class 0 = right, 1 = left), `artifacts/metrics.json` with keys `auroc`, `balanced_accuracy`, `confusion_matrix`, `test_size`; `infer.predict_study(npz_path) -> {"dominance_type": "right"|"left", "confidence": float, "views_used": list[str]}`.

- [ ] **Step 1:** Run `inspect_dataset.py`; paste real structure into `STATUS.md`. Do not write `dataset.py` before this.
- [ ] **Step 2:** Write failing test `test_study_split_has_no_overlap`: study-level train/val/test index sets are disjoint and class ratio within 5 points of 706:319 in each split.
- [ ] **Step 3:** Run it, expect FAIL; implement `dataset.py` with `build_splits(seed=42) -> dict[str, list[str]]` and a `StudyDataset` that samples N=16 frames per study (uniform over frames, both views when present).
- [ ] **Step 4:** Pass the test. Add test `test_single_view_study_loads` (a study with one view yields a tensor `[16,3,224,224]`).
- [ ] **Step 5:** Implement `model.py` (`DominanceNet`: ImageNet-pretrained ResNet-34 frame encoder, mean-pool over frames, linear head) and `train.py` (class-weighted CE, AdamW, cosine, AMP, early stop on val balanced accuracy). Smoke-train 1 epoch on 32 studies; loss must decrease.
- [ ] **Step 6:** Full training in background (`python train.py > train.log`); then `evaluate.py` → `metrics.json`; `export_onnx.py` prints PASS.
- [ ] **Step 7:** Commit in monorepo: `git commit -m "Train coronary dominance classifier"` (artifacts stay ignored).

### Task 2: Retrain and fully evaluate arrhythmia

**Files:**
- Modify: `ml-training/arrhythmia/STATUS.md`; reuse `src/dataset_builder.py`, `train.py`, `evaluate.py`, `export_onnx.py`

**Interfaces:**
- Produces: `artifacts/model.onnx`, `feature_stats.json`, `metrics.json` where `inter_patient.per_class.*.support` sums to the full DS2 beat count (>40,000), plus `confusion_matrix.png`.

- [ ] **Step 1:** Rebuild `data/processed/ds1.npz`, `ds2.npz` from all 44 usable MIT-BIH records (`python dataset_builder.py`); print class counts; DS2 total must be >40,000 beats.
- [ ] **Step 2:** Run `python -m pytest test_preprocess.py test_rhythm_rules.py` → 13 pass.
- [ ] **Step 3:** `python train.py` (background), `python evaluate.py`, `python export_onnx.py` → PASS.
- [ ] **Step 4:** Record V sensitivity and S sensitivity in `STATUS.md` whatever they are.

### Task 3: Push stenosis past Dice 0.60

**Files:**
- Modify: `ml-training/vessel_analysis/src/train.py` / `config.py` (epochs, LR, loss weights, input size) — keep vessel model untouched.

**Interfaces:**
- Produces: `artifacts/stenosis_model.onnx`, `metrics_stenosis.json` with `dice` on the 300-image test split.

- [ ] **Step 1:** Baseline stays 0.5748. Run up to 3 tuned configurations (longer schedule, Tversky weighting, higher resolution if memory allows); keep the best by **validation** Dice, never by test.
- [ ] **Step 2:** Evaluate the chosen run once on test; export ONNX → PASS; record result (met or missed) in `STATUS.md`.

### Task 4: Push EF past R² 0.75

**Files:**
- Modify: `ml-training/ef_estimation/src/train.py` / `config.py` (epochs, LR, clip sampling, augmentation, TTA in `evaluate.py`).

**Interfaces:**
- Produces: `artifacts/ef_model.onnx`, `metrics.json` with `mae`, `rmse`, `r2` on 1,277 test videos.

- [ ] **Step 1:** Baseline MAE 5.128 / R² 0.6764. Run up to 3 tuned configs, selecting by validation MAE.
- [ ] **Step 2:** Evaluate once on test; export ONNX → PASS; record result in `STATUS.md`.

---

## Phase C — Repos (can proceed while Phase B trains; do in this order)

### Task 5: `arterion-vessel-analysis` backend + scaffold (defines the repo template)

**Files (in `F:\cardiosense-repos\arterion-vessel-analysis`):**
- Create: `backend/app/main.py`, `backend/app/uploads.py`, `backend/app/vessel_analysis/{model,severity,report_generator,video_utils,router}.py` (copied from monorepo, auth/db removed), `backend/requirements.txt`, `backend/Dockerfile`, `backend/tests/test_api.py`, `training/` (copy of `ml-training/vessel_analysis` minus data/artifacts/pyc), `scripts/fetch_weights.ps1`, `scripts/fetch_weights.sh`, `docker-compose.yml`, `.gitignore`, `.env.example`, `LICENSE`, `README.md`

**Interfaces:**
- Produces (template for Tasks 7, 9, 11, 13): `GET /health -> {"status":"ok"}`; `POST /uploads` (multipart `file`) `-> {"file_url": str}`, 400 on empty, 413 over 200 MB; `POST /<task>/predict {"file_url": str} -> dict` (shape per model wrapper); `predict()` returns the fallback dict when weights are absent; unknown/bad media → 422 `{"detail": str}`. Backend reads weights from `backend/app/<task>/weights/`; `fetch_weights` downloads `https://github.com/kartikpagariya25/<repo>/releases/download/v1.0.0/<file>` into that folder.

- [ ] **Step 1:** `git init`; write failing tests in `backend/tests/test_api.py`: `test_health`, `test_upload_roundtrip`, `test_upload_empty_rejected` (400), `test_predict_without_weights_returns_fallback` (summary contains "weights not found"), `test_predict_rejects_non_image` (422).
- [ ] **Step 2:** Run `pytest backend/tests -v` → FAIL (app missing).
- [ ] **Step 3:** Copy and trim modules; drop auth/scans routers and SQLite; make Gemini report optional via `GEMINI_API_KEY` env with a rule-based fallback report; add upload size cap and error mapping.
- [ ] **Step 4:** `pytest backend/tests -v` → PASS.
- [ ] **Step 5:** Copy trained ONNX files locally into the weights folder (ignored by git) and run one real ARCADE test image through `predict`: expect non-empty `lesions` or `vessel_mask_present: true`.
- [ ] **Step 6:** README (real metrics from `metrics_vessel.json`/`metrics_stenosis.json`, setup, fetch weights, docker). Commit: `git commit -m "Initial backend, training code and docs"`.

### Task 6: `arterion-vessel-analysis` frontend — dark angiography-lab viewer

**Files:** Create `frontend/index.html`, `frontend/css/app.css`, `frontend/js/{config,app}.js`, `frontend/Dockerfile`, `frontend/nginx.conf`; Modify `docker-compose.yml` (add `frontend` on 8080 → backend 8000).

**Interfaces:** Consumes Task 5 endpoints via `window.API_BASE` from `config.js`. Produces: single-page tool — drop zone (image or video), before/after overlay slider, lesion list with severity badges and stenosis %, report panel.

- [ ] **Step 1:** Build page; handle states: idle, uploading, analysing, result, error, weights-missing notice.
- [ ] **Step 2:** `docker compose up` → open in the browser pane; upload `F:\cardiosense\test_angiogram.mp4` frame and an ARCADE image; confirm overlay slider and lesions render, no console errors.
- [ ] **Step 3:** Check 375 px width and wrong-file error (upload a `.txt`). Screenshot to `docs/screenshot.png`; add to README.
- [ ] **Step 4:** Commit: `git commit -m "Add angiography-lab frontend"`.

### Task 7: `arterion-ef-estimation` backend + scaffold

Same file layout and interface template as Task 5 (`/ef-estimation/predict` → `{ejection_fraction, classification, summary, confidence}`), source `ml-inference/app/ef_estimation` and `ml-training/ef_estimation`.

- [ ] **Step 1:** Failing tests as in Task 5, plus `test_predict_rejects_non_video` (422) and `test_short_video_still_predicts` (a 10-frame synthetic `.avi` returns a numeric EF when weights present, fallback otherwise).
- [ ] **Step 2:** Implement/trim; make `_sample_clip` loop-pad videos shorter than 32 frames and resize frames not 112×112.
- [ ] **Step 3:** Pass tests; run one real EchoNet test video through the stack with final weights; EF within ±15 of the label in `FileList.csv`.
- [ ] **Step 4:** README with real metrics from Task 4; commit.

### Task 8: `arterion-ef-estimation` frontend — clinical echo console

Same Interfaces/steps as Task 6: video drop zone with in-page player, animated EF gauge with normal/mildly/moderately/severely-reduced bands, summary card.

- [ ] Build; test with an EchoNet video and a too-short video; 375 px check; screenshot; commit.

### Task 9: `arterion-lv-outline` backend + scaffold

Same template; `/lv-outline/predict` → `{mask_base64? , overlay_image_base64, lv_area_pixels, summary, confidence}` (source `ml-inference/app/lv_outline`, `ml-training/lv_outline`; training code imports PS1's `VesselUNet` via file path — copy `model.py` and `losses.py` into this repo's `training/src/` and drop the cross-import).

- [ ] **Step 1:** Failing tests as Task 5 (non-image → 422).
- [ ] **Step 2:** Implement; ensure overlay is returned (add if the current wrapper returns only area).
- [ ] **Step 3:** Run a real echo frame; area > 0; README with metrics (Dice 0.9249); commit.

### Task 10: `arterion-lv-outline` frontend — blueprint/editorial

Frame drop zone (image or pick frame from a video), contour traced over the heart with animated stroke, area readout, mask on/off toggle.

- [ ] Build; test image + video-frame pick; 375 px; screenshot; commit.

### Task 11: `arterion-arrhythmia` backend + scaffold

Same template; `/arrhythmia/predict` → `{rhythm_type, flagged_beats, beat_summary, rhythm_alerts, summary, confidence}` (source `ml-inference/app/arrhythmia`, `ml-training/arrhythmia`). Also add `GET /arrhythmia/sample` serving a bundled sample ECG CSV (from `ml-training/arrhythmia/data/test_signals`).

- [ ] **Step 1:** Failing tests as Task 5, plus `test_csv_with_header_row_handled`, `test_signal_too_short_returns_warning`, `test_non_csv_rejected` (422).
- [ ] **Step 2:** Implement with tolerant CSV parsing; Pass tests; run the bundled sample with final weights → non-"unknown" rhythm.
- [ ] **Step 3:** README with Task 2's real metrics; commit.

### Task 12: `arterion-arrhythmia` frontend — ECG monitor

CSV drop zone + "try sample" button; canvas waveform with colour-coded beat markers (N/S/V/F/Q), rhythm banner, per-class beat counts, alerts list.

- [ ] Build; test sample and a malformed CSV; 375 px; screenshot; commit.

### Task 13: `arterion-coronary-dominance` backend + frontend — light anatomical atlas

**Interfaces:** `/dominance/predict {file_url} -> {dominance_type: "right"|"left"|"unknown", confidence, views_used, summary}`; accepts an angiogram video/`.npz`/image sequence upload (same extraction as `infer.predict_study`). Frontend: light atlas look, study drop zone, two-heart schematic highlighting the dominant artery, confidence bar, honesty note showing test AUROC.

- [ ] **Step 1:** Backend failing tests as Task 5 plus `test_single_view_reports_views_used`.
- [ ] **Step 2:** Implement using Task 1's `infer.py`/ONNX; pass tests; run one test-split study → returns a class.
- [ ] **Step 3:** Frontend build, 375 px, screenshot, README with Task 1's real metrics (even if modest); commit backend and frontend separately.

---

## Phase D — Publish and verify

### Task 14: Per-repo release and push (outward-facing — confirm each name first)

- [ ] **Step 1:** Decide EF/LV weight publishing (EchoNet terms): if unclear, publish those two repos with `fetch_weights` documenting "train it yourself" and no Release asset; tell the owner.
- [ ] **Step 2:** For each repo, after owner confirms the name: `env -u GITHUB_TOKEN gh repo create kartikpagariya25/<name> --public --source . --description "<one line>" --push`.
- [ ] **Step 3:** `env -u GITHUB_TOKEN gh release create v1.0.0 <weights...> --title "v1.0.0 weights" --notes "<metrics>"` (no attribution text).
- [ ] **Step 4:** From a fresh clone in a temp dir: run `fetch_weights`, `docker compose up`, open the site, run one sample. All five.

### Task 15: Final verification and link-back

- [ ] **Step 1:** Run each repo's `pytest backend/tests` → all PASS.
- [ ] **Step 2:** Update monorepo `README.md` with links to the five repos and real current metrics; commit and push (owner's identity).
- [ ] **Step 3:** Report to the owner: repo URLs, achieved vs target metrics, anything not done.
