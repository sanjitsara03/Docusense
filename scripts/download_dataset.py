"""
Download RVL-CDIP from Hugging Face and write it into the local directory
structure expected by dataset.py:

  data/
    images/
      <split>/
        <class_name>/
          <idx>.png
    labels/
      train.txt
      val.txt
      test.txt

Each labels/<split>.txt line: "images/<split>/<class>/<idx>.png <remapped_label>"

Usage:
  python scripts/download_dataset.py --output-dir data/

Requires:
  pip install datasets Pillow
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image

# RVL-CDIP original label index → name (only our 5 classes)
SELECTED: dict[int, str] = {
    0: "letter",
    1: "form",
    2: "email",
    10: "budget",
    11: "invoice",
}

LABEL_REMAP: dict[int, int] = {
    orig: new for new, orig in enumerate(sorted(SELECTED.keys()))
}


def save_split(
    dataset,
    split: str,
    output_dir: Path,
) -> None:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create per-class subdirs
    for name in SELECTED.values():
        (images_dir / name).mkdir(parents=True, exist_ok=True)

    label_lines: list[str] = []
    idx_per_class: dict[int, int] = {k: 0 for k in SELECTED}

    for example in dataset:
        orig_label: int = example["label"]
        if orig_label not in SELECTED:
            continue

        class_name = SELECTED[orig_label]
        remapped = LABEL_REMAP[orig_label]
        i = idx_per_class[orig_label]

        rel_path = f"images/{split}/{class_name}/{i}.png"
        abs_path = output_dir / rel_path

        # Save image as PNG (dataset provides PIL Images)
        img: Image.Image = example["image"]
        img.convert("RGB").save(abs_path)

        label_lines.append(f"{rel_path} {remapped}")
        idx_per_class[orig_label] += 1

    with open(labels_dir / f"{split}.txt", "w") as f:
        f.write("\n".join(label_lines) + "\n")

    counts = {SELECTED[k]: idx_per_class[k] for k in SELECTED}
    print(f"  {split}: {sum(counts.values())} images — {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Root directory to write dataset into")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Which HF splits to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Downloading RVL-CDIP → {output_dir.resolve()}")
    print(f"Keeping classes: {list(SELECTED.values())}\n")

    # HF split names: "train", "validation", "test"
    # dataset.py expects: "train", "val", "test"
    hf_to_local = {"train": "train", "validation": "val", "test": "test"}

    for hf_split in args.splits:
        local_split = hf_to_local[hf_split]
        print(f"Loading {hf_split} split from Hugging Face...")
        ds = load_dataset("aharley/rvl_cdip", split=hf_split, streaming=False)
        print(f"Saving {local_split}...")
        save_split(ds, local_split, output_dir)

    print(f"\nDone. Dataset written to {output_dir.resolve()}")
    print("Run training with: python training/train.py --data-dir data/")


if __name__ == "__main__":
    main()
