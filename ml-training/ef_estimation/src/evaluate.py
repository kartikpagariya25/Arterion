import os
import json
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ARTIFACTS_DIR
from dataset import EFDataset
from model import EFRegressor


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EFRegressor().to(device)
    model_path = os.path.join(ARTIFACTS_DIR, "ef_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_set = EFDataset("test")
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

    all_preds, all_true = [], []
    with torch.no_grad():
        for clips, efs in test_loader:
            clips = clips.to(device)
            preds = model(clips).cpu().numpy() * 100
            all_preds.extend(preds.tolist())
            all_true.extend((efs.numpy() * 100).tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    mae = float(np.mean(np.abs(all_preds - all_true)))
    rmse = float(np.sqrt(np.mean((all_preds - all_true) ** 2)))
    ss_res = np.sum((all_true - all_preds) ** 2)
    ss_tot = np.sum((all_true - all_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "task": "ef_estimation",
        "test_set_size": len(test_set),
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
    evaluate()
