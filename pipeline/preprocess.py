"""
SageMaker Processing Job script

Reads raw images + label files from S3 ,
applies resize + normalize, and writes processed splits to
/opt/ml/processing/output/ for the training job.

"""

import argparse
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Resize images to this size before saving")
    return parser.parse_args()


def process_split(
    split: str,
    input_dir: Path,
    output_dir: Path,
    image_size: int,
) -> None:
    label_file = input_dir / "labels" / f"{split}.txt"
    out_split_dir = output_dir / split
    out_labels_dir = out_split_dir / "labels"
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    valid_lines: list[str] = []
    skipped = 0

    with open(label_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        rel_path, label = line.rsplit(" ", 1)
        src = input_dir / rel_path
        dst = out_split_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(src).convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img.save(dst, format="PNG")
            valid_lines.append(f"{rel_path} {label}")
        except Exception as e:
            print(f"  Skipping {src}: {e}")
            skipped += 1

    out_label_file = out_labels_dir / f"{split}.txt"
    with open(out_label_file, "w") as f:
        f.write("\n".join(valid_lines) + "\n")

    print(f"  {split}: {len(valid_lines)} images processed, {skipped} skipped")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Resize: {args.image_size}px\n")

    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        process_split(split, input_dir, output_dir, args.image_size)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
