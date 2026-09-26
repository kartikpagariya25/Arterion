import os
import numpy as np
import torch

from config import ARTIFACTS_DIR, IMAGE_SIZE
from model import LVUNet


def export():
    model_path = os.path.join(ARTIFACTS_DIR, "lv_model.pt")
    onnx_path = os.path.join(ARTIFACTS_DIR, "lv_model.onnx")

    model = LVUNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported to {onnx_path}")
    return model, onnx_path


def verify(model, onnx_path, n_samples: int = 5):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for _ in range(n_samples):
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            torch_out = model(x).numpy()
        onnx_out = session.run(["logits"], {"image": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))

    print(f"max absolute difference over {n_samples} random samples: {max_diff:.6f}")
    print("PASS" if max_diff < 1e-3 else "FAIL")


if __name__ == "__main__":
    model, onnx_path = export()
    verify(model, onnx_path)
