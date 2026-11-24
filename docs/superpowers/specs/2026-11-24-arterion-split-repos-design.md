# Arterion split: five self-contained repos — design

Date: 2026-11-24 · Owner: Kartik Pagariya (kartikpagariya25)

## Intent and why
The Arterion monorepo (GitHub `kartikpagariya25/Arterion`, formerly CardioSense) bundles five cardiac ML models behind one site. The owner wants each model to be its own showcase: a separate GitHub repo with its own purpose-built website, runnable on its own, so each project can be shown, starred and reviewed independently. Models that are not yet good enough should be trained properly on the owner's PC.

## Goals
1. Five new public repos under `kartikpagariya25`, each fully self-contained (clone, fetch weights, `docker compose up`).
2. Five **fully distinct** websites, each built around one tool: upload input → model runs → results view.
3. Real trained weights for all five; real metrics in each README.
4. Everything committed, pushed and released under the owner's identity (see Constraints).

## Non-goals
- No changes to the Arterion monorepo's behaviour (only its git remote URL may be updated).
- No auth/login/dashboard in the per-project sites (my call: focused single-tool sites).
- No clinical claims; each site carries a "research prototype, not a medical device" notice.
- No co-dominance class, no extra datasets beyond those already in `ml-training`.

## Anti-goals
- Metrics that look good because of a smoke-sized or leaky evaluation. Numbers published must come from the full held-out test split.
- Five sites that are palette-swaps of one template.
- Secrets, databases, datasets or `.pt`/`.onnx` files committed to git.

## Constraints
- **Attribution:** commits, pushes, repos and releases use the owner's git identity; no `Co-Authored-By` or "Generated with Claude" lines anywhere.
- Hardware: one RTX 5050 8 GB, Windows, shared venv at `ml-training/venv` (torch 2.11+cu128). GPU jobs run one at a time.
- GitHub: no file >100 MB in git. Weights (98–125 MB) go to **GitHub Releases** with a `scripts/fetch_weights` script; apps degrade gracefully when weights are missing (as the existing model wrappers already do).
- Repo creation and release publishing are outward-facing: names and visibility are confirmed with the owner immediately before each is created. Auth: `gh` is logged in as `kartikpagariya25` (run with `GITHUB_TOKEN` unset in agent shells).

## Approaches considered
- **A. Copy and adapt per repo (chosen).** Trim backend/frontend/training for one model each. Cost: duplicated U-Net/loss code across vessel, LV and EF repos.
- B. Shared `arterion-core` package. Rejected: breaks "clone one repo and run it".
- C. Generate repos from the monorepo by script. Rejected: conflicts with five distinct designs.

## Repos

| Repo | Model / task | Input → output | Design concept |
|---|---|---|---|
| `arterion-vessel-analysis` | VesselUNet ×2 (vessel + stenosis), skeleton-width severity, LLM report | angiogram image or video → overlay, lesions, severity, report | Dark angiography-lab viewer, overlay slider |
| `arterion-ef-estimation` | R(2+1)D-18 regressor | echo video (A4C) → EF % | Clinical echo console, live EF gauge |
| `arterion-lv-outline` | LVUNet (ResNet34) | echo frame → LV mask/contour, area | Blueprint/editorial, contour traced over heart |
| `arterion-arrhythmia` | CardioNet 1D ResNet + RR features + rhythm rules | ECG CSV → beat classes (AAMI N/S/V/F/Q), rhythm | ECG monitor, waveform with beat markers |
| `arterion-coronary-dominance` | New classifier (video/frames → right vs left) | angiogram study → dominance + confidence | Light anatomical-atlas style |

Each repo layout:
```
frontend/    static site (own design), nginx.conf, Dockerfile
backend/     FastAPI: /<task>/predict, /uploads, /health; only this model
training/    existing src/, PRD.md, STATUS.md, requirements.txt (data/ and artifacts/ ignored)
scripts/     fetch_weights.(ps1|sh)
docker-compose.yml, README.md (real metrics, screenshots), LICENSE, .gitignore
```
API contract stays the same as today (`POST /uploads` → `file_url`, then `POST /<task>/predict {file_url}`) so existing wrappers port over. The vessel report's Gemini dependency stays optional, key via env var, with a built-in non-LLM fallback report.

## Training plan (sequential on GPU, background)
1. **Dominance (new):** data already extracted at `ml-training/dominance/data/extracted` (main dataset, 1,025 studies, 319 left / 706 right). Inspect real `.npz` structure first (never assumed), build study-level train/val/test split (no patient leakage), class-weighted training, report AUROC/balanced accuracy/confusion matrix, export ONNX with parity check. Result is published as-is, even if modest.
2. **Arrhythmia:** retrain on full DS1, evaluate on full DS2 (inter-patient); current `metrics.json` is a ~2.3k-beat smoke run. Report per-class sensitivity, especially V and S.
3. **Stenosis:** current Dice 0.575; target ≥0.60 via longer schedule/tuning on ARCADE.
4. **EF:** current MAE 5.13, R² 0.676; target R² ≥0.75 via longer training/tuning on EchoNet.
Targets are goals, not promises; if missed, the README states the achieved numbers.

## Build order
1. Scaffold five repo directories under `F:\cardiosense-repos\<name>` (outside the monorepo), each a fresh git repo.
2. Backends and weight-fetch scripts (no GPU needed) while training runs.
3. Five frontends, one at a time, each verified in the browser at desktop and phone width.
4. Swap in final weights, run each stack end-to-end with real inputs, capture screenshots.
5. Confirm names → `gh repo create`, push, publish Release with weights.

## Verification
Per repo: unit/smoke tests for preprocessing and API; `docker compose up` works from a clean clone with weights fetched; real sample input produces a sensible result in the UI; README metrics match `metrics.json`.

## Left to the builder
Exact frontend component choices and palettes within each concept; dominance architecture (start with a pretrained 2D CNN on sampled frames, aggregate per study; try video model only if time allows); licensing text (MIT for code; dataset licences noted: ARCADE, EchoNet research-use terms, MIT-BIH ODC-By, CoronaryDominance CC0 — **EchoNet terms may restrict redistributing derived weights; check before publishing the EF/LV releases**).

## My calls (please check)
- Repos live in `F:\cardiosense-repos\` (separate from the monorepo).
- No auth/dashboard in per-project sites.
- MIT licence for code.
- Repo name `arterion-arrhythmia` (not "arrhythmia").
- Update monorepo `origin` to the renamed `Arterion` URL.
