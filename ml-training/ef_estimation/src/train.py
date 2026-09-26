import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ARTIFACTS_DIR, SEED
from dataset import EFDataset
from model import EFRegressor


def evaluate_loader(model, loader, device):
    model.eval()
    total_abs_error = 0.0
    n = 0
    with torch.no_grad():
        for clips, efs in loader:
            clips, efs = clips.to(device), efs.to(device)
            preds = model(clips)
            total_abs_error += (preds - efs).abs().sum().item()
            n += efs.size(0)
    return (total_abs_error / n) * 100  # de-normalize back to EF percentage points


def train(epochs: int = 50, batch_size: int = 4, patience: int = 7):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_set = EFDataset("train")
    val_set = EFDataset("val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = EFRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_mae = float("inf")
    patience_counter = 0
    model_path = os.path.join(ARTIFACTS_DIR, "ef_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for clips, efs in train_loader:
            clips, efs = clips.to(device), efs.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(clips)
                loss = criterion(preds, efs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * clips.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_set)
        val_mae = evaluate_loader(model, val_loader, device)

        print(f"epoch {epoch:02d}/{epochs}  loss={avg_loss:.5f}  val_mae={val_mae:.3f}", flush=True)

        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping at epoch {epoch} (best val_mae={best_mae:.3f})")
                break

    print(f"training complete. best model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
