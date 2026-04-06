"""
RVL-CDIP dataset loader, filtered to 5 classes.

RVL-CDIP has 16 classes. We use:
  0  → letter
  1  → form
  2  → email
  10 → budget
  11 → invoice

These are remapped to contiguous labels 0-4 for training.

Dataset structure expected:
  <root>/
    images/           # subdirs with .tif files
    labels/
      train.txt       # lines: "images/subdir/file.tif <class_idx>"
      val.txt
      test.txt
"""

import os
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# RVL-CDIP original indices → our label names
SELECTED_CLASSES: dict[int, str] = {
    0: "letter",
    1: "form",
    2: "email",
    10: "budget",
    11: "invoice",
}

# Remapped contiguous indices (used as training targets)
LABEL_MAP: dict[int, int] = {
    orig: new for new, orig in enumerate(sorted(SELECTED_CLASSES.keys()))
}
# {0: 0, 1: 1, 2: 2, 10: 3, 11: 4}

CLASS_NAMES: list[str] = [SELECTED_CLASSES[k] for k in sorted(SELECTED_CLASSES.keys())]
NUM_CLASSES: int = len(CLASS_NAMES)

# ImageNet stats that EfficientNet was pretrained on 
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# Returns a torchvision transform pipeline for the given split.
# Training adds random augmentations (crop, flip, rotation) to reduce overfitting.
# Val/test use deterministic center crop so metrics are reproducible.
def get_transforms(split: Literal["train", "val", "test"]) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

# Reads the labels file for the given split, filters to our 5 classes,
# and builds self.samples — a list of (image_path, remapped_label) pairs.
# No images are loaded here; loading is deferred to __getitem__.
class RVLCDIPDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = get_transforms(split)
        self.samples: list[tuple[Path, int]] = []

        label_file = self.root / "labels" / f"{split}.txt"
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, class_str = line.rsplit(" ", 1)
                orig_class = int(class_str)
                if orig_class not in SELECTED_CLASSES:
                    continue
                img_path = self.root / rel_path
                self.samples.append((img_path, LABEL_MAP[orig_class]))

        if max_samples is not None:
            per_class = max_samples // NUM_CLASSES
            buckets: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
            for path, label in self.samples:
                if len(buckets[label]) < per_class:
                    buckets[label].append((path, label))
            self.samples = [s for bucket in buckets.values() for s in bucket]

    # Returns the total number of samples in this split.
    # Called by DataLoader to determine how many batches to generate.
    def __len__(self) -> int:
        return len(self.samples)

    # Loads and preprocesses a single sample by index.
    # Returns a (tensor, label) tuple
    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        # RVL-CDIP images are grayscale .tif so we convert to RGB for EfficientNet
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
