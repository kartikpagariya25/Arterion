import json
import numpy as np
import onnxruntime as ort

from config import SAMPLING_RATE, CLASSES
from preprocess import (
    resample_to_target, bandpass_filter, detect_r_peaks,
    extract_beat_window, normalize_beat, compute_rr_features,
)
from rhythm_rules import detect_alerts

RHYTHM_DESCRIPTIONS = {
    "VT": "Ventricular Tachycardia",
    "AFIB": "Atrial Fibrillation",
    "BIGEMINY": "Ventricular Bigeminy",
    "TRIGEMINY": "Ventricular Trigeminy",
    "TACHY": "Sinus Tachycardia",
    "BRADY": "Sinus Bradycardia",
}


class ArrhythmiaAnalyzer:
    def __init__(self, model_path: str, feature_stats_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        with open(feature_stats_path) as f:
            stats = json.load(f)
        self.rr_mean = np.array(stats["rr_mean"], dtype=np.float32)
        self.rr_std = np.array(stats["rr_std"], dtype=np.float32)

    def _classify_beats(self, cleaned_signal: np.ndarray, r_peaks: np.ndarray):
        beats, rrs, valid_peaks = [], [], []
        for i, r in enumerate(r_peaks):
            window = extract_beat_window(cleaned_signal, r)
            if window is None:
                continue
            beats.append(normalize_beat(window))
            rrs.append(compute_rr_features(r_peaks, i))
            valid_peaks.append(r)

        if not beats:
            return []

        beats = np.array(beats, dtype=np.float32)[:, None, :]
        rrs = (np.array(rrs, dtype=np.float32) - self.rr_mean) / (self.rr_std + 1e-8)

        logits = self.session.run(["logits"], {"beat": beats, "rr": rrs})[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

        results = []
        for r, pred, conf in zip(valid_peaks, preds, confidences):
            results.append({
                "time_sec": round(float(r) / SAMPLING_RATE, 3),
                "class": CLASSES[pred],
                "confidence": round(float(conf), 4),
            })
        return results

    def _overall_rhythm(self, alerts, beat_summary):
        critical = [a for a in alerts if a["severity"] == "critical"]
        warning = [a for a in alerts if a["severity"] == "warning"]

        if critical:
            return RHYTHM_DESCRIPTIONS.get(critical[0]["type"], critical[0]["type"])
        if warning:
            return RHYTHM_DESCRIPTIONS.get(warning[0]["type"], warning[0]["type"])
        if beat_summary.get("V", 0) > 0 or beat_summary.get("S", 0) > 0:
            return "Normal Sinus Rhythm with Isolated Ectopic Beats"
        return "Normal Sinus Rhythm"

    def _summary_text(self, rhythm_type, beat_summary, alerts):
        total = sum(beat_summary.values())
        abnormal = total - beat_summary.get("N", 0)

        if not alerts and abnormal == 0:
            return f"Normal sinus rhythm across {total} analyzed beats. No abnormalities detected."

        parts = [f"{rhythm_type} pattern detected."]
        if beat_summary.get("V", 0) > 0:
            parts.append(f"{beat_summary['V']} ventricular ectopic beat(s).")
        if beat_summary.get("S", 0) > 0:
            parts.append(f"{beat_summary['S']} supraventricular ectopic beat(s).")
        if alerts:
            parts.append(f"{len(alerts)} rhythm-level alert(s) flagged for review.")
        return " ".join(parts)

    def analyze(self, signal: np.ndarray, sampling_rate: int) -> dict:
        resampled = resample_to_target(signal, sampling_rate, SAMPLING_RATE)
        cleaned = bandpass_filter(resampled, SAMPLING_RATE)
        r_peaks = detect_r_peaks(cleaned, SAMPLING_RATE)

        beats = self._classify_beats(cleaned, r_peaks)
        if not beats:
            return {
                "rhythm_type": "Undetermined",
                "flagged_beats": [],
                "beat_summary": {},
                "summary": "Unable to detect valid beats in the provided signal.",
                "confidence": 0.0,
            }

        beat_summary = {c: 0 for c in CLASSES}
        for b in beats:
            beat_summary[b["class"]] += 1

        rhythm_input = [{"time": b["time_sec"], "class": b["class"]} for b in beats]
        rhythm_alerts = detect_alerts(rhythm_input)
        rhythm_type = self._overall_rhythm(rhythm_alerts, beat_summary)

        flagged_beats = [b for b in beats if b["class"] != "N"]
        overall_confidence = round(float(np.mean([b["confidence"] for b in beats])), 4)

        return {
            "rhythm_type": rhythm_type,
            "flagged_beats": flagged_beats,
            "beat_summary": beat_summary,
            "rhythm_alerts": rhythm_alerts,
            "summary": self._summary_text(rhythm_type, beat_summary, rhythm_alerts),
            "confidence": overall_confidence,
        }
