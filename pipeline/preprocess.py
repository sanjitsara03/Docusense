"""
SageMaker Processing Job script

Reads raw images + label files from S3,
applies resize + normalize, and writes processed splits to
/opt/ml/processing/output/ for the training job.

Uses multiprocessing to parallelize image resizing across all available CPUs.
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Resize images to this size before saving")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: all CPUs)")
    return parser.parse_args()


def process_image(args: tuple[Path, Path, int]) -> str | None:
    """Process a single image. Returns the label line on success, None on failure."""
    src, dst, image_size = args
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img.save(dst, format="PNG")
        return None  # signal success — caller reconstructs the label line
    except Exception as e:
        print(f"  Skipping {src}: {e}")
        return str(src)  # signal failure


def process_split(
    split: str,
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    workers: int,
) -> None:
    label_file = input_dir / "labels" / f"{split}.txt"
    out_split_dir = output_dir / split
    out_labels_dir = out_split_dir / "labels"
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(label_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Build work items: (src_path, dst_path, image_size)
    work: list[tuple[Path, Path, int]] = []
    label_map: dict[str, str] = {}
    for line in lines:
        rel_path, label = line.rsplit(" ", 1)
        src = input_dir / rel_path
        dst = out_split_dir / rel_path
        work.append((src, dst, image_size))
        label_map[rel_path] = label

    failed: set[str] = set()
    with Pool(processes=workers) as pool:
        for src, result in zip(
            [w[0] for w in work],
            pool.imap_unordered(process_image, work, chunksize=64),
        ):
            if result is not None:
                failed.add(result)

    valid_lines = [
        f"{rel_path} {label}"
        for rel_path, label in label_map.items()
        if str(input_dir / rel_path) not in failed
    ]

    out_label_file = out_labels_dir / f"{split}.txt"
    with open(out_label_file, "w") as f:
        f.write("\n".join(valid_lines) + "\n")

    print(f"  {split}: {len(valid_lines)} images processed, {len(failed)} skipped")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Resize:  {args.image_size}px")
    print(f"Workers: {args.workers}\n")

    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        process_split(split, input_dir, output_dir, args.image_size, args.workers)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
