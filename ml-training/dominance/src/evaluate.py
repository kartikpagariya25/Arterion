import datetime
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from config import ARTIFACTS_DIR
from dataset import StudyDataset, collate
from model import DominanceNet
from train import SIZE, run_eval


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DominanceNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "dominance_model.pt"), map_location=device))
    te = StudyDataset("test", False, SIZE)
    p, y, bacc, auc = run_eval(model, DataLoader(te, 6, num_workers=0, collate_fn=collate), device)
    pred = (p > 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics = {
        "task": "coronary_dominance",
        "classes": ["right", "left"],
        "test_size": int(len(y)),
        "auroc": round(float(auc), 4),
        "balanced_accuracy": round(float(bacc), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "sensitivity_left": round(float(cm[1, 1] / max(cm[1].sum(), 1)), 4),
        "specificity_right": round(float(cm[0, 0] / max(cm[0].sum(), 1)), 4),
        "confusion_matrix": cm.tolist(),
        "model": {"name": "ResNet-18 frame encoder + view embedding + study mean", "params": sum(x.numel() for x in model.parameters())},
        "dataset": "CoronaryDominance main_normal (study-level 70/15/15 split, seed 42)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
