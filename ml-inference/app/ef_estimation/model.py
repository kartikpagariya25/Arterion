MODEL_PATH = "weights/ef_estimation.pt"

_model = None


def load_model():
    global _model
    if _model is None:
        _model = "loaded"
    return _model


def predict(file_url: str) -> dict:
    load_model()
    ejection_fraction = 0.0
    classification = "normal" if ejection_fraction >= 55 else "reduced"
    return {
        "ejection_fraction": ejection_fraction,
        "classification": classification,
        "summary": "EF estimation model not yet wired up.",
        "confidence": 0.0,
    }
