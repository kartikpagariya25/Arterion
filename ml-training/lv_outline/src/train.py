import os
import importlib.util
import torch
from torch.utils.data import DataLoader

_PS1_LOSSES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vessel_analysis", "src", "losses.py")
)
_spec = importlib.util.spec_from_file_location("ps1_losses", _PS1_LOSSES_PATH)
_ps1_losses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps1_losses)
CombinedLoss = _ps1_losses.CombinedLoss
dice_score = _ps1_losses.dice_score

from config import ARTIFACTS_DIR, SEED
from dataset import LVSegmentationDataset
from model import LVUNet


def evaluate_loader(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            scores.append(dice_score(logits, masks))
    return sum(scores) / len(scores)


def train(epochs: int = 50, batch_size: int = 8, patience: int = 7):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_set = LVSegmentationDataset("train")
    val_set = LVSegmentationDataset("val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = LVUNet().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice = -1.0
    patience_counter = 0
    model_path = os.path.join(ARTIFACTS_DIR, "lv_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_set)
        val_dice = evaluate_loader(model, val_loader, device)

        print(f"epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}  val_dice={val_dice:.4f}", flush=True)

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping at epoch {epoch} (best val_dice={best_dice:.4f})")
                break

    print(f"training complete. best model saved to {model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
