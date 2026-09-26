# PS5 — Heart Rhythm Abnormality Detector — PRD

**Project:** CardioSense (Synapse Hackathon, Symbiosis Pune — AI in Healthcare track)
**Problem Statement:** Heart Rhythm Abnormality Detector
**Scope of this document:** ML model only (data → training → evaluation → exported inference artifact). Platform integration (auth, EHR, case pipeline) already exists in the main CardioSense repo and is out of scope here.

---

## 1. Problem

Patients are monitored with ECG machines that record the heart's electrical activity as a waveform. Abnormal rhythms (arrhythmias) currently have to be spotted by a doctor reading the waveform by eye — slow, and error-prone at scale. We need a model that reads a raw ECG signal and classifies each heartbeat, flagging abnormal rhythms automatically.

## 2. Goal

Build a beat-level ECG classifier that:
- Takes a raw ECG signal (or a pre-segmented beat) as input
- Classifies every heartbeat into one of 5 clinically standard classes
- Reports rhythm-level abnormalities (not just isolated beats) derived from the beat sequence
- Is evaluated honestly, on patients never seen during training
- Is exported as a lightweight artifact that plugs into the existing `ml-inference/app/arrhythmia/` service in the main CardioSense repo

## 3. Non-Goals (explicitly out of scope for this model)

- 12-lead ECG (only MIT-BIH's 2-channel data, using lead `MLII`)
- OCR / paper ECG digitization
- Live streaming / WebSocket monitoring
- LLM-generated clinical reports
- Any UI work (handled by the main CardioSense frontend)

These may become future-scope talking points in the pitch, not build items here.

## 4. Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet) — 48 half-hour two-channel ambulatory ECG recordings from 47 patients, sampled at 360 Hz, every beat annotated by a cardiologist. Rhythm-level annotations (AFib, VT, bigeminy, etc.) are also present in the annotation files.

- Access: fully open, no credentialing required — `wfdb.dl_database('mitdb', ...)`
- No dataset files are committed to the repo; only code and the final trained artifacts are.

## 5. Clinical Labeling Standard

Raw MIT-BIH beat symbols are mapped to the **AAMI EC57** standard, the standard actually used in clinical arrhythmia-classification literature — not ad-hoc labels.

| AAMI class | Meaning | MIT-BIH symbols |
|---|---|---|
| N | Normal / bundle branch block beats | N, L, R, e, j |
| S | Supraventricular ectopic beat | A, a, J, S |
| V | Ventricular ectopic beat (dangerous) | V, E |
| F | Fusion of ventricular + normal beat | F |
| Q | Unknown / paced / unclassifiable | /, f, Q |

Fixed class order used everywhere: `["N", "S", "V", "F", "Q"]`.

## 6. Success Criteria

| Metric | Target |
|---|---|
| Evaluation protocol | Inter-patient split (train and test patients never overlap) — this is the credibility differentiator; a random split inflates accuracy and judges will ask about it |
| V-class sensitivity (test set) | ≥ 0.85 — V (ventricular) beats are the most clinically dangerous and the ones a demo should catch reliably |
| Macro-F1 (test set) | Reported honestly, no target inflation — Q and S are known-hard classes in this dataset and that's fine to say out loud |
| Inference latency | Single beat classification in well under 100ms on CPU (this is a tiny model — should be near-instant) |
| Export format | ONNX, verified numerically identical to PyTorch weights on a held-out sample |

## 7. Output Contract (what the model returns per case)

This must match what `ml-inference/app/arrhythmia/router.py` expects to return to the backend:

```json
{
  "rhythm_type": "Normal Sinus Rhythm | Atrial Fibrillation | Ventricular Tachycardia | ...",
  "flagged_beats": [
    { "time_sec": 12.4, "class": "V", "confidence": 0.94 }
  ],
  "beat_summary": { "N": 210, "S": 3, "V": 5, "F": 0, "Q": 1 },
  "summary": "Predominantly normal sinus rhythm with 5 isolated ventricular ectopic beats.",
  "confidence": 0.94
}
```

## 8. Deliverables

1. Reproducible training pipeline (`ml-training/arrhythmia/src/`)
2. Trained model weights + ONNX export + `metrics.json` (honest inter-patient results)
3. A rhythm-rule layer that converts a beat sequence into rhythm-level alerts (VT, AFib, bradycardia, tachycardia, bigeminy, trigeminy)
4. A drop-in replacement for `ml-inference/app/arrhythmia/model.py` in the main repo, wired to the exported ONNX model
5. This PRD + the accompanying `IMPLEMENTATION.md`

## 9. Hardware Context

Training machine: RTX 5050 (8GB VRAM), 16GB RAM. This model is intentionally small (1D ResNet, a few hundred thousand parameters) and the full preprocessed dataset (~650K beats × 256 samples) fits comfortably in RAM — training completes in minutes, not hours, on this hardware. No memory pressure is expected at any phase.

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| Class imbalance (N beats vastly outnumber S/V/F/Q) | Focal loss with inverse-frequency class weighting (Section on training config in IMPLEMENTATION.md) |
| R-peak detection failing on noisy inference-time signals | Use `neurokit2` robust R-peak detection at inference; ground-truth annotation positions used only at training time |
| Overfitting to the 22 training patients | Inter-patient split is mandatory; validation set is a held-out slice of training patients only, never test patients |
| Q class near-absent after excluding paced records | Report as "insufficient samples" if test-set Q count < 20; never drop the class from the model itself |
