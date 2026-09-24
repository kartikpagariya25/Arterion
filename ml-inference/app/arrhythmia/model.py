MODEL_PATH = "weights/arrhythmia.pt"

_model = None


def load_model():
    global _model
    if _model is None:
        _model = "loaded"
    return _model


def predict(file_url: str) -> dict:
    load_model()
    return {
        "rhythm_type": "unknown",
        "flagged_beats": [],
        "summary": "Arrhythmia detection model not yet wired up.",
        "confidence": 0.0,
    }
