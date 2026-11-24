import os
import json
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ARTIFACTS_DIR
from dataset import EFDataset, CLIP_LENGTH, _clip_at, _sample_clip, to_tensor, window_starts
from model import EFRegressor


def evaluate(tag: str = None, period: int = 1, windows: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EFRegressor().to(device)
    model_path = os.path.join(ARTIFACTS_DIR, f"{tag or 'ef'}_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_set = EFDataset("test", period)
    all_preds, all_true, center_preds = [], [], []
    with torch.no_grad():
        for i in range(len(test_set)):
            video, ef = test_set.load_video(i)
            starts = window_starts(video.shape[0], CLIP_LENGTH, period, windows)
            clips = torch.from_numpy(np.stack([to_tensor(_clip_at(video, s, CLIP_LENGTH, period)) for s in starts])).float().to(device)
            p = model(clips).cpu().numpy() * 100
            all_preds.append(float(p.mean()))
            center_preds.append(float(model(torch.from_numpy(to_tensor(_sample_clip(video, CLIP_LENGTH, False, period))[None]).float().to(device)).cpu().numpy()[0] * 100))
            all_true.append(ef * 100)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    center_preds = np.array(center_preds)

    mae = float(np.mean(np.abs(all_preds - all_true)))
    rmse = float(np.sqrt(np.mean((all_preds - all_true) ** 2)))
    ss_res = np.sum((all_true - all_preds) ** 2)
    ss_tot = np.sum((all_true - all_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "task": "ef_estimation",
        "test_set_size": len(test_set),
        "protocol": {"frame_period": period, "windows": windows, "clip_length": CLIP_LENGTH},
        "mae_center_window_only": round(float(np.mean(np.abs(center_preds - all_true))), 3),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "model": {"name": "R(2+1)D-18 EF regressor", "params": sum(p.numel() for p in model.parameters())},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"ef_estimation: mae={metrics['mae']}, rmse={metrics['rmse']}, r2={metrics['r2']}")
    print(f"saved {metrics_path}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true, all_preds, alpha=0.4, s=10)
    lims = [0, 100]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("True EF (%)")
    ax.set_ylabel("Predicted EF (%)")
    ax.set_title(f"EF Prediction (MAE={mae:.2f}, R²={r2:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "scatter_plot.png"))
    plt.close(fig)
    print("saved scatter_plot.png")

    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--windows", type=int, default=1)
    a = ap.parse_args()
    evaluate(a.tag, a.period, a.windows)
