import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_DIR, ARTIFACTS_DIR, CLASSES, SEED, BATCH_SIZE, LR,
    WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOP_PATIENCE, FOCAL_GAMMA, VAL_FRACTION,
)
from dataset import BeatDataset
from model import CardioNet


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


def compute_class_alpha(labels, num_classes):
    counts = np.array([(labels == i).sum() for i in range(num_classes)], dtype=np.float64)
    counts = np.maximum(counts, 1)
    inv_sqrt = 1.0 / np.sqrt(counts)
    alpha = inv_sqrt / inv_sqrt.sum() * num_classes
    return torch.tensor(alpha, dtype=torch.float32)


def macro_f1(preds, targets, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_loader(model, loader, device):
    model.eval()
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
    return macro_f1(preds, targets, len(CLASSES))


def train(intra: bool = False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    npz_name = "intra.npz" if intra else "ds1.npz"
    npz_path = os.path.join(PROCESSED_DIR, npz_name)
    data = np.load(npz_path)
    labels = data["labels"]

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_FRACTION, stratify=labels, random_state=SEED
    )
    train_labels = labels[train_idx]

    rr_mean = data["rr"][train_idx].mean(axis=0)
    rr_std = data["rr"][train_idx].std(axis=0)

    train_dataset = BeatDataset(npz_path, rr_mean=rr_mean, rr_std=rr_std, augment=True)
    val_dataset = BeatDataset(npz_path, rr_mean=rr_mean, rr_std=rr_std, augment=False)

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CardioNet(num_classes=len(CLASSES)).to(device)
    alpha = compute_class_alpha(train_labels, len(CLASSES)).to(device)
    criterion = FocalLoss(alpha, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_f1 = -1.0
    patience_counter = 0
    suffix = "_intra" if intra else ""
    model_path = os.path.join(ARTIFACTS_DIR, f"model{suffix}.pt")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for beat, rr, label in train_loader:
            beat, rr, label = beat.to(device), rr.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(beat, rr)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * beat.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_set)
        val_f1 = evaluate_loader(model, val_loader, device)

        print(f"epoch {epoch:02d}/{MAX_EPOCHS}  loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {epoch} (best val_macro_f1={best_f1:.4f})")
                break

    print(f"training complete. best model saved to {model_path}")

    if not intra:
        stats_path = os.path.join(ARTIFACTS_DIR, "feature_stats.json")
        with open(stats_path, "w") as f:
            json.dump({"rr_mean": rr_mean.tolist(), "rr_std": rr_std.tolist()}, f, indent=2)
        print(f"saved feature stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intra", action="store_true", help="train the intra-patient comparison model")
    args = parser.parse_args()
    train(intra=args.intra)
