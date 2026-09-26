import os
import importlib.util

_PS1_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vessel_analysis", "src", "model.py")
)
_spec = importlib.util.spec_from_file_location("ps1_vessel_model", _PS1_MODEL_PATH)
_ps1_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps1_model)

LVUNet = _ps1_model.VesselUNet

__all__ = ["LVUNet"]
