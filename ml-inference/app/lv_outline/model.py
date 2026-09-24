MODEL_PATH = "weights/lv_outline.pt"

_model = None


def load_model():
    global _model
    if _model is None:
        _model = "loaded"
    return _model


def predict(file_url: str) -> dict:
    load_model()
    return {
        "outline_video_url": file_url,
        "edv": 0.0,
        "esv": 0.0,
        "summary": "LV outline model not yet wired up.",
        "confidence": 0.0,
    }
