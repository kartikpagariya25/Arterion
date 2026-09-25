import os
import json
from datetime import datetime, timezone
import numpy as np
import torch

from config import PROCESSED_DIR, ARTIFACTS_DIR, CLASSES
from dataset import BeatDataset
from model import CardioNet


def load_feature_stats():
    path = os.path.join(ARTIFACTS_DIR, "feature_stats.json")
    with open(path) as f:
        stats = json.load(f)
    return np.array(stats["rr_mean"], dtype=np.float32), np.array(stats["rr_std"], dtype=np.float32)


def per_class_metrics(preds, targets, num_classes):
    metrics = {}
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        confusion[t, p] += 1

    for c in range(num_classes):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp
        fp = confusion[:, c].sum() - tp
        support = confusion[c, :].sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0.0

        metrics[CLASSES[c]] = {
            "sensitivity": round(float(sensitivity), 4),
            "ppv": round(float(ppv), 4),
            "f1": round(float(f1), 4),
            "support": int(support),
        }

    macro_f1 = float(np.mean([metrics[c]["f1"] for c in CLASSES]))
    accuracy = float(np.trace(confusion) / confusion.sum())

    return metrics, macro_f1, accuracy, confusion.tolist()


def run_eval(model_path, npz_path, rr_mean, rr_std, device):
    model = CardioNet(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = BeatDataset(npz_path, rr_mean=rr_mean, rr_std=rr_std, augment=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for beat, rr, label in loader:
            beat, rr = beat.to(device), rr.to(device)
            logits = model(beat, rr)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(label.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    per_class, macro_f1, accuracy, confusion = per_class_metrics(preds, targets, len(CLASSES))
    return {
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": confusion,
    }


def save_confusion_plot(confusion, path):
    import matplotlib.pyplot as plt

    confusion = np.array(confusion)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Inter-patient confusion matrix (DS2)")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, confusion[i, j], ha="center", va="center", color="black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rr_mean, rr_std = load_feature_stats()

    inter_model_path = os.path.join(ARTIFACTS_DIR, "model.pt")
    inter_npz = os.path.join(PROCESSED_DIR, "ds2.npz")
    inter_results = run_eval(inter_model_path, inter_npz, rr_mean, rr_std, device)
    print(f"Inter-patient (DS2) macro-F1: {inter_results['macro_f1']}, accuracy: {inter_results['accuracy']}")
    print(f"V-class sensitivity: {inter_results['per_class']['V']['sensitivity']}")

    intra_model_path = os.path.join(ARTIFACTS_DIR, "model_intra.pt")
    intra_results = None
    if os.path.exists(intra_model_path):
        intra_npz = os.path.join(PROCESSED_DIR, "intra.npz")
        intra_results = run_eval(intra_model_path, intra_npz, rr_mean, rr_std, device)
        print(f"Intra-patient macro-F1: {intra_results['macro_f1']} (for pitch comparison only)")

    total_params = sum(p.numel() for p in CardioNet(num_classes=len(CLASSES)).parameters())

    metrics = {
        "classes": CLASSES,
        "inter_patient": inter_results,
        "intra_patient": intra_results,
        "dataset": {
            "name": "MIT-BIH Arrhythmia Database",
            "excluded_records": [102, 104, 107, 217],
        },
        "model": {
            "name": "CardioNet (1D ResNet + RR features)",
            "params": total_params,
            "input": "256-sample beat @360 Hz + 4 RR features",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {metrics_path}")

    save_confusion_plot(inter_results["confusion_matrix"], os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"))
    print("saved confusion_matrix.png")


if __name__ == "__main__":
    main()
