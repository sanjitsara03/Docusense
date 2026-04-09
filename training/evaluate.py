"""
Evaluation script — runs after training, produces metrics.

Outputs written to --output-dir:
  metrics.json          accuracy + per-class F1 + confusion matrix
  confusion_matrix.png  confusion matrix visualization

SageMaker Pipelines calls this as a processing step. The evaluate step reads
metrics.json to decide whether to register the model (accuracy > 0.90).
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, NUM_CLASSES, RVLCDIPDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TEST", "data"))
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "eval_output"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    import tarfile
    from train import build_model
    # SageMaker ProcessingJob downloads model artifacts as model.tar.gz — extract if needed
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        tar_path = model_path_obj.parent / "model.tar.gz"
        if tar_path.exists():
            with tarfile.open(tar_path) as tar:
                tar.extractall(model_path_obj.parent)
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def save_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, device)

    test_dataset = RVLCDIPDataset(args.data_dir, "test", max_samples=args.max_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    print(f"Evaluating on {len(test_dataset)} test samples")

    preds, labels = run_evaluation(model, test_loader, device)

    accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    f1_per_class = f1_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)))
    f1_macro = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_per_class": {
            CLASS_NAMES[i]: round(float(f1_per_class[i]), 4)
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "num_test_samples": len(labels),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"F1 macro:  {f1_macro:.4f}")
    print("F1 per class:")
    for name, score in metrics["f1_per_class"].items():
        print(f"  {name:<10} {score:.4f}")
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")

    save_confusion_matrix(cm, output_dir)
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
