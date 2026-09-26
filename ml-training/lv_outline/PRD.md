# PS4 — Left Ventricle Outline Tool — PRD

**Project:** CardioSense (Synapse Hackathon, Symbiosis Pune — AI in Healthcare track)
**Problem Statement:** Left Ventricle Outline Tool
**Scope of this document:** ML model only.

---

## 1. Problem

On every echocardiogram, doctors manually trace the outline of the left ventricle (the heart's main pumping chamber) to measure its size — a slow, repetitive task done by hand for every study. We need a model that draws this outline automatically.

## 2. Goal

Given an echocardiogram frame, output a pixel-level segmentation mask of the left ventricle's endocardial border.

## 3. Non-Goals

- Full-video temporal tracking/propagation of the contour (each frame is segmented independently — a reasonable, disclosed scope limit, not a hidden shortcut)
- Volume/EF calculation from the outline (that's PS2's job, via a different — and more accurate, video-level — method; this PS's deliverable is the outline itself, which is what the problem statement literally asks for)

## 4. Dataset

**EchoNet-Dynamic** — same videos as PS2 (see that PRD for access details), but this PS uses a different label: `VolumeTracings.csv`, which contains expert-drawn LV boundary point coordinates. Ground truth exists **only on two frames per video** — the end-diastolic (ED) and end-systolic (ES) frames — not every frame. This is a real, disclosed limitation of the source labels, not a shortcut we introduced: the original EchoNet paper itself only has tracings on these two frames per video.

This still gives a substantial labeled set (~2 frames × ~10,000 videos ≈ 20,000 labeled frames) — enough to train a solid per-frame segmentation model that generalizes to any frame at inference time, since LV shape recognition doesn't require temporal context the way EF regression does.

**Exact real column names and coordinate format are confirmed in Phase 1, shared with PS2's inspection step — not assumed here.**

## 5. Approach

Binary segmentation (LV vs. background), same family of architecture as PS1's `VesselUNet` (U-Net + ResNet34 encoder) — reused rather than reinvented, since it's already a proven, tested implementation in this codebase. The LV is a much simpler shape to segment than PS1's thin branching vessels (a filled blob vs. a thin line), so this task is expected to reach a meaningfully higher Dice score than PS1.

## 6. Success Criteria

| Metric | Target |
|---|---|
| Dice score on held-out ED/ES frames (test split) | ≥ 0.90 (published EchoNet segmentation baselines reach ~0.90-0.92 — this is a well-established, achievable target for this specific shape, unlike PS1's genuinely harder thin-vessel case) |
| Export | ONNX, verified numerically identical to PyTorch |

## 7. Output Contract

```json
{
  "outline_mask_base64": "<base64 PNG>",
  "lv_area_pixels": 4820,
  "summary": "Left ventricle outline generated for the analyzed frame.",
  "confidence": 0.9
}
```

(Same `_base64` pragmatic choice as PS1, for the reason documented in PS1's `STATUS.md` — no storage-upload step in this ML-only phase.)

## 8. Deliverables

1. Reproducible training pipeline (`ml-training/lv_outline/src/`)
2. Trained weights + ONNX export + `metrics.json`
3. Drop-in replacement for `ml-inference/app/lv_outline/model.py`
4. This PRD + `IMPLEMENTATION.md`

## 9. Hardware Context

Lighter than PS2 — this is 2D per-frame segmentation (like PS1), not 3D video convolution, so it fits comfortably in 8GB at a similar batch size to PS1.

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| Ground truth only on 2 frames/video — no way to directly validate accuracy on the "other" frames a demo might show | Disclosed plainly in the pitch: accuracy is reported on the frames that actually have expert ground truth, and generalization to other frames is a reasonable, stated assumption — not an inflated claim |
| Coordinate-to-mask conversion errors (polygon fill direction, off-by-one on frame indices) | Visual spot-check overlays generated and inspected before training, exactly as PS1's Phase 2 did with real ARCADE data |
| Reusing PS1's `VesselUNet` class without modification risk | Same architecture, independently trained weights — no shared state between the two tasks, so PS1's results are unaffected |
