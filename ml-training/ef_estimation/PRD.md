# PS2 — Heart Pumping Efficiency Estimator — PRD

**Project:** CardioSense (Synapse Hackathon, Symbiosis Pune — AI in Healthcare track)
**Problem Statement:** Heart Pumping Efficiency Estimator
**Scope of this document:** ML model only.

---

## 1. Problem

Doctors use echocardiogram videos to judge how well the heart pumps blood — a measure called **ejection fraction (EF)**: the percentage of blood the left ventricle pumps out with each beat. A low EF is an early sign of heart failure. Reading EF off a video currently requires a trained sonographer to manually trace the ventricle at two specific moments in the cardiac cycle and compute a volume ratio. We need a model that watches the video and estimates EF automatically.

## 2. Goal

Given a raw echocardiogram video, predict the ejection fraction as a percentage, and flag whether it falls in a clinically reduced range.

## 3. Non-Goals

- Manual frame-by-frame tracing output (that's PS4's job, sharing the same dataset)
- Multi-view echo (EchoNet-Dynamic is apical-4-chamber view only — the model is scoped to that view, which is disclosed rather than implied to generalize)

## 4. Dataset

**EchoNet-Dynamic** (Stanford AIMI, Ouyang et al., *Nature* 2020) — 10,030 apical-4-chamber echocardiogram videos, each with an expert-measured EF, ESV (end-systolic volume), EDV (end-diastolic volume), and an official train/val/test split.

Expected structure (public documentation — **confirmed against the real download in Phase 1 before any parsing code is written**, same discipline as PS1's ARCADE inspection):
- `FileList.csv` — one row per video: filename, EF, ESV, EDV, FPS, frame count, official Split (TRAIN/VAL/TEST)
- `VolumeTracings.csv` — per-frame LV boundary point coordinates (only on the labeled ED/ES frames — this is what PS4 consumes)
- `Videos/` — `.avi` files

Access requires Stanford's Research Use Agreement (already completed for this project via Redivis).

## 5. Approach

Following the original EchoNet methodology (a validated, published approach — not something to reinvent): a **spatiotemporal 3D CNN** (R(2+1)D-18, Kinetics-400 pretrained) takes a fixed-length sampled clip from the video and regresses EF directly as a single number. This is a regression task, not classification — the clinical band (below) is derived from the continuous prediction, not predicted as a separate class.

## 6. Success Criteria

| Metric | Target |
|---|---|
| MAE on official test split | ≤ 5.0 (the original EchoNet paper reports ~4.1 MAE; ≤5.0 is a realistic hackathon-timeline target, and the real number is reported honestly either way) |
| R² correlation with ground-truth EF | ≥ 0.75 |
| Clinical band derived from predicted EF | Standard ASE (American Society of Echocardiography) categories — see Section 7 |
| Export | ONNX, verified numerically identical to PyTorch on a held-out sample |

## 7. Output Contract

```json
{
  "ejection_fraction": 54.2,
  "classification": "normal | mildly_reduced | moderately_reduced | severely_reduced",
  "summary": "Estimated EF 54.2% — within normal range.",
  "confidence": 0.8
}
```

Clinical bands (standard ASE thresholds, not invented):
- `normal`: EF ≥ 55%
- `mildly_reduced`: 45–54%
- `moderately_reduced`: 30–44%
- `severely_reduced`: < 30%

## 8. Deliverables

1. Reproducible training pipeline (`ml-training/ef_estimation/src/`)
2. Trained weights + ONNX export + `metrics.json`
3. Drop-in replacement for `ml-inference/app/ef_estimation/model.py`
4. This PRD + `IMPLEMENTATION.md`

## 9. Hardware Context

RTX 5050 (8GB VRAM). Video models are the heaviest of all 5 PS — a 3D CNN over even a short clip (e.g. 32 frames at 112×112) uses substantially more VRAM than PS1's 2D U-Net. Batch size, clip length, and frame resolution are chosen in `IMPLEMENTATION.md` specifically to fit 8GB, verified before a full run — same lesson as PS1's Phase 4 VRAM check.

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| 7GB dataset, video I/O bottleneck on a single consumer GPU | Frame sampling and clip caching considered in the training pipeline design, not left to chance |
| VRAM pressure from 3D convolutions | Clip length and batch size verified against 8GB before committing to a full run (Phase 3 "done when" check) |
| EF regression is inherently noisy (inter-observer variability in the ground truth itself, published even in the original paper) | Report MAE honestly; do not chase an unrealistic near-zero error |
| Real dataset structure differs from documented public format | Phase 1 is an inspection step, not an assumption — mirrors PS1's approach exactly |
