import os
import json
import argparse
from datetime import datetime, timezone
import torch
from torch.utils.data import DataLoader

from config import ARTIFACTS_DIR
from dataset import SegmentationDataset
from model import VesselUNet

TASK_MAP = {"vessel": "segmentation", "stenosis": "stenosis"}


def compute_metrics(logits, targets, threshold: float = 0.5, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return dice, iou, precision, recall


def evaluate(task_key: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TASK_MAP[task_key]

    model = VesselUNet().to(device)
    model_path = os.path.join(ARTIFACTS_DIR, f"{task_key}_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_set = SegmentationDataset(task, "test")
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    all_dice, all_iou, all_precision, all_recall = [], [], [], []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            dice, iou, precision, recall = compute_metrics(logits, masks)
            all_dice.extend(dice.cpu().tolist())
            all_iou.extend(iou.cpu().tolist())
            all_precision.extend(precision.cpu().tolist())
            all_recall.extend(recall.cpu().tolist())

    def mean(x):
        return round(sum(x) / len(x), 4)

    metrics = {
        "task": task_key,
        "test_set_size": len(test_set),
        "dice": mean(all_dice),
        "iou": mean(all_iou),
        "precision": mean(all_precision),
        "recall": mean(all_recall),
        "model": {"name": "VesselUNet (ResNet34 encoder)", "params": sum(p.numel() for p in model.parameters())},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, f"metrics_{task_key}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{task_key}: dice={metrics['dice']}, iou={metrics['iou']}, precision={metrics['precision']}, recall={metrics['recall']}")
    print(f"saved {metrics_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["vessel", "stenosis"], required=True)
    args = parser.parse_args()
    evaluate(args.task)
