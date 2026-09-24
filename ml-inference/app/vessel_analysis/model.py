MODEL_PATH = "weights/vessel_analysis.pt"

_model = None


def load_model():
    global _model
    if _model is None:
        # torch.load(MODEL_PATH) once the ARCADE-trained checkpoint is ready
        _model = "loaded"
    return _model


def predict(file_url: str) -> dict:
    load_model()
    return {
        "overlay_image_url": file_url,
        "stenosis_percent": 0.0,
        "location": "unknown",
        "summary": "Vessel analysis model not yet wired up.",
        "confidence": 0.0,
    }
