import os
import json
import numpy as np
import torch

from config import PROCESSED_DIR, ARTIFACTS_DIR, CLASSES
from model import CardioNet


def export(model_path, output_path):
    model = CardioNet(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_beat = torch.randn(1, 1, 256)
    dummy_rr = torch.randn(1, 4)

    torch.onnx.export(
        model,
        (dummy_beat, dummy_rr),
        output_path,
        input_names=["beat", "rr"],
        output_names=["logits"],
        dynamic_axes={"beat": {0: "batch"}, "rr": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported to {output_path}")
    return model


def verify(model, onnx_path, npz_path, rr_mean, rr_std, n_samples=100):
    import onnxruntime as ort

    data = np.load(npz_path)
    n = min(n_samples, len(data["labels"]))
    idx = np.random.choice(len(data["labels"]), n, replace=False)

    beats = data["beats"][idx].astype(np.float32)
    rr = ((data["rr"][idx] - rr_mean) / (rr_std + 1e-8)).astype(np.float32)

    beats_t = torch.tensor(beats).unsqueeze(1)
    rr_t = torch.tensor(rr)
    with torch.no_grad():
        torch_logits = model(beats_t, rr_t).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"beat": beats[:, None, :], "rr": rr})[0]

    max_diff = np.abs(torch_logits - onnx_logits).max()
    print(f"max absolute difference over {n} samples: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("PASS")
    else:
        print("FAIL")


def main():
    model_path = os.path.join(ARTIFACTS_DIR, "model.pt")
    onnx_path = os.path.join(ARTIFACTS_DIR, "model.onnx")
    stats_path = os.path.join(ARTIFACTS_DIR, "feature_stats.json")
    ds2_path = os.path.join(PROCESSED_DIR, "ds2.npz")

    with open(stats_path) as f:
        stats = json.load(f)
    rr_mean = np.array(stats["rr_mean"], dtype=np.float32)
    rr_std = np.array(stats["rr_std"], dtype=np.float32)

    model = export(model_path, onnx_path)
    verify(model, onnx_path, ds2_path, rr_mean, rr_std)


if __name__ == "__main__":
    main()
