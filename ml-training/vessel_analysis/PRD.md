# PS1 — Real-Time Coronary Vessel Analyzer — PRD

**Project:** CardioSense (Synapse Hackathon, Symbiosis Pune — AI in Healthcare track)
**Problem Statement:** Real-Time Coronary Vessel Analyzer
**Scope of this document:** ML model only (data → training → evaluation → exported inference artifact). Platform integration already exists in the main CardioSense repo and is out of scope here.

---

## 1. Problem

During a cath lab procedure, doctors read live X-ray angiograms to judge where coronary arteries are narrowed or blocked (stenosis). This is currently done by eye. We need a model that takes an angiogram frame and automatically highlights narrowed/blocked vessel segments, helping the doctor judge severity faster.

## 2. Goal

Build a system that:
- Segments coronary vessels in an X-ray angiogram frame (vessel vs. background)
- Detects and localizes stenotic (narrowed) regions within the vessel tree
- Estimates a numeric stenosis severity (% narrowing) at each detected lesion by comparing the vessel's local width at the lesion to its normal width nearby
- Outputs an overlay image highlighting the lesion with its severity, for doctor review

## 3. Non-Goals

- Full 26-class SYNTAX-style anatomical vessel segment labeling (not needed for the demo's "highlight the blockage" goal — binary vessel-vs-background is sufficient and safer to train well in the time available)
- Real-time video stream processing (ARCADE provides annotated still frames, not continuous video — the demo operates frame-by-frame, which is an honest and explainable scope, not a shortcut to hide)
- 3D vessel reconstruction

## 4. Dataset

**ARCADE** (Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs) — MICCAI 2023 challenge dataset, published on Zenodo (record 8386059), with a companion Nature Scientific Data paper. It ships **two separately annotated subtasks**, both in COCO format:

| Subtask | What it labels | Used for |
|---|---|---|
| Segmentation (aka "syntax") | Per-vessel-segment masks (26 anatomical classes) | We collapse all vessel classes into one binary "vessel" mask — Model A |
| Stenosis | Stenotic-region masks/boxes | Model B |

- Access: fully open on Zenodo, no credentialing required
- 1500 expert-annotated images per subtask, 512×512 resolution
- No dataset files are committed to the repo; only code and final trained artifacts are

**Important — resolved in Phase 1, not assumed here:** the exact internal folder/file layout of the downloaded ARCADE archive (zip names, whether it's split into train/val/test-phase zips, exact COCO JSON filenames) is confirmed by actually downloading and inspecting it, not guessed in advance. Guessing wrong paths here would just cause the same kind of avoidable error we hit with PS5 — `IMPLEMENTATION.md` Phase 1 reflects this explicitly.

## 5. Success Criteria

| Metric | Target |
|---|---|
| Vessel segmentation (Model A) — Dice score on held-out ARCADE test split | ≥ 0.75 (coronary vessels are thin, branching structures — this is a harder segmentation target than typical organ segmentation, and the literature reflects that) |
| Stenosis segmentation (Model B) — Dice score on held-out test split | ≥ 0.60 (stenosis regions are small and subtle; this is the genuinely hard part of this PS, and honest reporting here is a differentiator, same as PS5's S/F-class story) |
| Severity estimate | Reported as a relative-width-reduction percentage, with the estimation method disclosed — not claimed as clinically validated, since ground-truth severity percentages are not part of ARCADE's labels |
| Inference latency | Under 1s per frame on CPU acceptable for a hackathon demo (single-frame analysis, not a real-time video requirement) |
| Export format | ONNX for both models, verified numerically identical to PyTorch on a held-out sample (same discipline as PS5) |

## 6. Output Contract

Matches what `ml-inference/app/vessel_analysis/router.py` expects to return:

```json
{
  "overlay_image_url": "<base64 or hosted URL of the annotated frame>",
  "vessel_mask_present": true,
  "lesions": [
    { "bbox": [x, y, w, h], "stenosis_percent": 62.4, "severity_band": "moderate" }
  ],
  "summary": "One moderate stenosis detected (~62% narrowing) in the mid-vessel segment.",
  "confidence": 0.81
}
```

`severity_band`: mild (<50%), moderate (50–70%), severe (>70%) — standard clinical stenosis-severity buckets, not invented thresholds.

## 7. Deliverables

1. Reproducible training pipeline (`ml-training/vessel_analysis/src/`) for both Model A and Model B
2. Trained weights + ONNX exports + `metrics.json` for both models
3. A severity-estimation post-processing module (vessel-mask skeleton width analysis at the lesion location)
4. A drop-in replacement for `ml-inference/app/vessel_analysis/model.py`
5. This PRD + the accompanying `IMPLEMENTATION.md`

## 8. Hardware Context

RTX 5050 (8GB VRAM), 16GB RAM. Unlike PS5's tiny 1D model, this is 2D image segmentation — a U-Net with a ResNet34 encoder at 512×512 is a few tens of millions of parameters and will use meaningfully more VRAM and time than PS5. Batch size and image resolution are tuned in `IMPLEMENTATION.md` specifically to fit 8GB comfortably, based on the lesson from PS5 that hardware-specific choices (like the Blackwell/torch CUDA build mismatch) need to be verified, not assumed.

## 9. Key Risks

| Risk | Mitigation |
|---|---|
| `pycocotools` install friction on Windows (a recurring pain point for this exact library) | Verified in Phase 0 before any other work — if the plain `pip install pycocotools` wheel doesn't resolve cleanly on the machine's Python version, fall back to `pycocotools-windows`, checked explicitly rather than assumed |
| ARCADE's real folder/file structure differs from any assumption made ahead of time | Phase 1 output is an actual directory listing before any COCO-parsing code is written — no hardcoded paths until confirmed |
| Stenosis class severe imbalance (most of an image is not a lesion) | Dice + focal-Tversky loss combination (standard for small-lesion segmentation), not plain BCE |
| Severity-percent metric has no ground truth in ARCADE to validate against | Disclosed as an estimated/derived metric in the pitch, not presented as a validated clinical measurement — same honesty stance as PS5's inter-patient story |
| GPU memory pressure from 512×512 image segmentation on 8GB VRAM | Batch size and mixed-precision (AMP) tuned and verified to fit before a full training run is attempted, not discovered mid-run via an out-of-memory crash |
