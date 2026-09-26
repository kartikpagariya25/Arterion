# PS3 — Coronary Dominance Classifier — PRD

**Project:** CardioSense (Synapse Hackathon, Symbiosis Pune — AI in Healthcare track)
**Problem Statement:** Coronary Dominance Classifier
**Scope of this document:** ML model only.

---

## 1. Problem

Before deciding on treatment, doctors need to know which artery (left or right) supplies a key part of the heart — a trait called coronary dominance that varies patient to patient and affects surgical/stenting planning. Determining it manually from an angiogram is subjective and has documented inter-observer variability in the literature. We need a model that classifies dominance automatically from angiogram video.

## 2. Goal

Given an angiogram study (RCA and/or LCA view video(s)), classify the patient as right-dominant or left-dominant.

## 3. Non-Goals

- Co-dominance as a separate third class — the source dataset itself is framed as a binary right/left classification (confirmed from the published paper; re-confirmed against the real data in Phase 1 before any label-mapping code is written)
- Training on the full 88.6GB release — deliberately scoped down (Section 4)

## 4. Dataset

**CoronaryDominance** (Kruzhilov et al., *Scientific Data* 2025), hosted on Hugging Face (`BearSubj13/CoronaryDominance`, CC0-1.0, fully open — no gating, unlike EchoNet).

**Real, confirmed facts** (from the HF API and the published paper — not assumed):
- Full release across 4 archives totals **~88.6GB**: `main_normal.7z` (40.7GB), `main_others.7z` (14.2GB), `real_distribution.7z` (25.9GB), `domain_shift_dataset.7z` (7.8GB)
- **Scope decision for this project:** download and train on `main_normal.7z` + `main_others.7z` only (~55GB) — together, these are the complete "main" dataset (1,025 studies: 319 left-dominant, 706 right-dominant — a real, disclosed ~1:2.2 class imbalance). `real_distribution` and `domain_shift` are the paper's own generalization test sets, not required for a working classifier and explicitly out of scope for hackathon time/disk constraints — this is a stated scope cut, not a hidden shortcut
- Data was converted by the dataset authors to compressed NumPy (`.npz`) arrays, pixel range 0-255, images/frames resized to 512×512
- Each study includes X-ray angiogram video from RCA and/or LCA views (per the authors' related paper, ~30-60 frames per view)
- Five per-study quality tags exist (normal, bad quality, artifact, high uncertainty, RCA occlusion) — used as-is if present in the real label file, not re-derived

**Exact internal file/folder layout inside the `.7z` archives is not publicly documented in enough detail to code against blindly** — Phase 1 in `IMPLEMENTATION.md` is an inspection step before any parsing code is written, more load-bearing here than in any other PS in this project, since even the planning stage (me, writing this PRD) could not pre-verify it — the archives are far too large to download and inspect ahead of time in the way ARCADE/MIT-BIH were.

## 5. Approach

Following the dataset's own published baseline methodology (2D frame-based classification per view, e.g. ConvNeXt/Swin on individual RCA frames — a validated approach from the dataset creators' own related work): a 2D CNN classifier on sampled frames from the RCA view (the view the source literature identifies as primary for this decision), with LCA used as a fallback/secondary signal per the same literature, if the real data structure makes that practical once inspected.

## 6. Success Criteria

| Metric | Target |
|---|---|
| Accuracy on held-out split | ≥ 0.85 (published baselines on this exact dataset report strong performance; realistic hackathon-timeline target, reported honestly either way) |
| Macro-F1 | Reported honestly — the ~1:2.2 class imbalance (left vs. right) means macro-F1 matters more than raw accuracy, same lesson as PS5's class-imbalance handling |
| Per-class recall | Both classes' recall reported individually — do not let the majority (right-dominant) class mask poor left-dominant recall |
| Export | ONNX, verified numerically identical to PyTorch |

## 7. Output Contract

```json
{
  "dominance": "left | right",
  "confidence": 0.91,
  "summary": "Right-dominant coronary circulation predicted with high confidence."
}
```

## 8. Deliverables

1. Reproducible training pipeline (`ml-training/dominance/src/`)
2. Trained weights + ONNX export + `metrics.json`
3. Drop-in replacement for `ml-inference/app/dominance/model.py`
4. This PRD + `IMPLEMENTATION.md`

## 9. Hardware Context

2D frame classification (like PS1/PS4), not video-level 3D convolution (like PS2) — should train at a similar speed to PS1's per-epoch time despite the much larger download, since training operates on sampled 2D frames, not full videos, once preprocessed.

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| 55GB download — bandwidth/time/disk | Resumable, checksum-verified download (same discipline as PS1); real SHA256 hashes obtained from the HF API for both required archives, so a corrupted download is caught immediately rather than discovered mid-training |
| Unknown internal archive structure | Phase 1 inspection is mandatory before Phase 2 parsing code is written — no guessing |
| Class imbalance (706 right vs. 319 left) | Class-weighted loss, same pattern as PS5's focal loss approach |
| RCA occlusion / bad-quality studies complicating labels | Use the dataset's own quality tags as provided; disclose in the pitch that these are known-hard cases per the source paper itself, not hidden |
| Disk space: ~55GB compressed + extraction overhead | Confirm free disk space before starting the download (Phase 0 check) — do not discover a full disk mid-extraction |
