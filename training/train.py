"""
SageMaker Training Job entry point.

Hyperparameters (passed as CLI args by SageMaker):
  --epochs          int     default 10
  --batch-size      int     default 32
  --lr              float   default 1e-3
  --max-samples     int     optional, for local smoke tests

SageMaker environment variables used:
  SM_CHANNEL_TRAIN   path to training data
  SM_CHANNEL_VAL     path to validation data (separate channel in pipeline)
  SM_MODEL_DIR       where to write the final model artifact
  SM_CHECKPOINT_DIR  where to write/read checkpoints (spot resumption)
  SM_NUM_GPUS        number of GPUs on the instance
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from dataset import NUM_CLASSES, RVLCDIPDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap dataset size — useful for local testing")
    # SageMaker injects these automatically; kept as args for local overrides.
    # SM_CHANNEL_VAL is set when the pipeline maps train/val as separate channels;
    # falls back to --data-dir so local runs (single root with all splits) still work.
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    parser.add_argument("--val-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_VAL"))
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "model_output"))
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.environ.get("SM_CHECKPOINT_DIR", "checkpoints"))
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace the classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_acc: float,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        },
        path / "checkpoint.pt",
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    ckpt_file = path / "checkpoint.pt"
    if not ckpt_file.exists():
        return 0, 0.0
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]
    print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")
    return start_epoch, best_acc


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(dim=1).eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()

    num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
    device = torch.device("cuda" if num_gpus > 0 and torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | GPUs: {num_gpus}")

    # In the SageMaker pipeline train/val are separate input channels.
    # val_dir falls back to data_dir for local runs where both splits live under one root.
    val_dir = args.val_dir or args.data_dir
    train_dataset = RVLCDIPDataset(args.data_dir, "train", max_samples=args.max_samples)
    val_dataset = RVLCDIPDataset(val_dir, "val", max_samples=args.max_samples)
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir)
    start_epoch, best_acc = load_checkpoint(checkpoint_dir, model, optimizer)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    training_log: list[dict] = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        # Save checkpoint every epoch for spot instance resumption
        save_checkpoint(checkpoint_dir, epoch, model, optimizer, best_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            print(f"  New best model saved (val_acc={best_acc:.4f})")

    # Save final model + metadata
    torch.save(model.state_dict(), model_dir / "final_model.pt")
    with open(model_dir / "training_log.json", "w") as f:
        json.dump({"best_val_acc": round(best_acc, 4), "epochs": training_log}, f, indent=2)

    print(f"\nTraining complete. Best val_acc: {best_acc:.4f}")
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
