import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from config import ARTIFACTS_DIR, SEED
from dataset import StudyDataset, collate
from model import DominanceNet

SIZE = 160


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    probs, labels = [], []
    for frames, views, mask, y in loader:
        with torch.autocast("cuda", enabled=device == "cuda"):
            logits = model(frames.to(device), views.to(device), mask.to(device))
        probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    p, y = np.concatenate(probs), np.concatenate(labels)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return p, y, balanced_accuracy_score(y, p > 0.5), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use N studies")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = StudyDataset("train", True, SIZE)
    va = StudyDataset("val", False, SIZE)
    if a.limit:
        tr.studies, va.studies = tr.studies[: a.limit], va.studies[: a.limit]
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)
    vl = DataLoader(va, a.batch, num_workers=0, collate_fn=collate)

    labels = np.array([s["label"] for s in tr.studies])
    w = torch.tensor(len(labels) / (2 * np.maximum(np.bincount(labels, minlength=2), 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    model = DominanceNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, bad = -1.0, 0
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for frames, views, mask, y in tl:
            frames, views, mask, y = (t.to(device) for t in (frames, views, mask, y))
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = crit(model(frames, views, mask).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        _, _, bacc, auc = run_eval(model, vl, device)
        print(f"epoch {ep} loss {tot / len(tl):.4f} val_bacc {bacc:.4f} val_auroc {auc:.4f} {time.time() - t0:.0f}s", flush=True)
        if bacc > best:
            best, bad = bacc, 0
            torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "dominance_model.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break
    print("best val balanced accuracy", best)


if __name__ == "__main__":
    main()
