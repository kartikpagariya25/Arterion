MODEL_PATH = "weights/dominance.pt"

_model = None


def load_model():
    global _model
    if _model is None:
        _model = "loaded"
    return _model


def predict(file_url: str) -> dict:
    load_model()
    return {
        "dominance_type": "unknown",
        "summary": "Dominance classification model not yet wired up.",
        "confidence": 0.0,
    }
