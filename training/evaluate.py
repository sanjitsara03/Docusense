"""
Evaluation script — runs after training, produces metrics + Grad-CAM samples.

Outputs written to --output-dir:
  metrics.json          accuracy + per-class F1 + confusion matrix
  confusion_matrix.png  heatmap visualization
  gradcam/              Grad-CAM overlays for one sample per class

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
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

from dataset import CLASS_NAMES, NUM_CLASSES, RVLCDIPDataset, get_transforms


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
    from train import build_model
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks into the last conv block (features[-1]) to capture activations + gradients.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # EfficientNet-B0: features is a Sequential, last block is features[-1]
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns a (224, 224) float32 heatmap in [0, 1].
        image_tensor: (1, 3, 224, 224)
        """
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, class_idx]
        score.backward()

        # Global average pool the gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)


def overlay_heatmap(image_tensor: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    """Blend heatmap onto the original image. Returns (224, 224, 3) uint8."""
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]  # (224, 224, 3) in [0, 1]
    blended = (0.5 * img + 0.5 * colored).clip(0, 1)
    return (blended * 255).astype(np.uint8)


def save_gradcam_samples(
    model: torch.nn.Module,
    dataset: RVLCDIPDataset,
    output_dir: Path,
    device: torch.device,
    n_per_class: int = 2,
) -> None:
    gradcam = GradCAM(model)
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # Collect indices per class
    buckets: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(dataset.samples):
        if len(buckets[label]) < n_per_class:
            buckets[label].append(idx)
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    for class_idx, indices in buckets.items():
        class_name = CLASS_NAMES[class_idx]
        for i, sample_idx in enumerate(indices):
            image_tensor, label = dataset[sample_idx]
            inp = image_tensor.unsqueeze(0).to(device)

            with torch.enable_grad():
                heatmap = gradcam.generate(inp, class_idx)

            overlay = overlay_heatmap(image_tensor, heatmap)
            out_path = gradcam_dir / f"{class_name}_{i}.png"
            Image.fromarray(overlay).save(out_path)

    print(f"Grad-CAM samples saved to {gradcam_dir}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    # Annotate cells
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

    # Grad-CAM — run with gradients enabled even though model is in eval mode
    save_gradcam_samples(model, test_dataset, output_dir, device)


if __name__ == "__main__":
    main()
