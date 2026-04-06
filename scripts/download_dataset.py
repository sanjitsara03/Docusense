"""
Download RVL-CDIP from Hugging Face and write into the structure expected by dataset.py.

Local mode (default):
  data/
    images/<split>/<class_name>/<idx>.png
    labels/train.txt, val.txt, test.txt

S3 mode (--s3-uri s3://bucket/prefix/):
  s3://bucket/prefix/images/<split>/<class_name>/<idx>.png
  s3://bucket/prefix/labels/train.txt, val.txt, test.txt

Each labels/<split>.txt line: "images/<split>/<class>/<idx>.png <remapped_label>"

Usage (local):
  python scripts/download_dataset.py --output-dir data/

Usage (S3, run from SageMaker notebook):
  python scripts/download_dataset.py --s3-uri s3://your-bucket/docusense/

Requires:
  pip install datasets Pillow boto3
"""

import argparse
import io
from pathlib import Path

import boto3
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


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Returns (bucket, prefix) from s3://bucket/prefix/."""
    s3_uri = s3_uri.rstrip("/")
    without_scheme = s3_uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def save_split_local(dataset, split: str, output_dir: Path) -> None:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

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
        img: Image.Image = example["image"]
        img.convert("RGB").save(output_dir / rel_path)

        label_lines.append(f"{rel_path} {remapped}")
        idx_per_class[orig_label] += 1

    with open(labels_dir / f"{split}.txt", "w") as f:
        f.write("\n".join(label_lines) + "\n")

    counts = {SELECTED[k]: idx_per_class[k] for k in SELECTED}
    print(f"  {split}: {sum(counts.values())} images — {counts}")


def save_split_s3(dataset, split: str, bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")
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
        s3_key = f"{prefix}/{rel_path}" if prefix else rel_path

        buf = io.BytesIO()
        img: Image.Image = example["image"]
        img.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=s3_key, Body=buf.read())

        label_lines.append(f"{rel_path} {remapped}")
        idx_per_class[orig_label] += 1

        if i % 500 == 0:
            print(f"  {split}/{class_name}: {i} images uploaded...")

    label_key = f"{prefix}/labels/{split}.txt" if prefix else f"labels/{split}.txt"
    s3.put_object(
        Bucket=bucket,
        Key=label_key,
        Body=("\n".join(label_lines) + "\n").encode(),
    )

    counts = {SELECTED[k]: idx_per_class[k] for k in SELECTED}
    print(f"  {split}: {sum(counts.values())} images — {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Local directory to write dataset into (ignored if --s3-uri set)")
    parser.add_argument("--s3-uri", type=str, default=None,
                        help="S3 URI to write dataset into, e.g. s3://my-bucket/docusense/")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    use_s3 = args.s3_uri is not None
    if use_s3:
        bucket, prefix = parse_s3_uri(args.s3_uri)
        print(f"Downloading RVL-CDIP → s3://{bucket}/{prefix}/")
    else:
        output_dir = Path(args.output_dir)
        print(f"Downloading RVL-CDIP → {output_dir.resolve()}")

    print(f"Keeping classes: {list(SELECTED.values())}\n")

    for split in args.splits:
        print(f"Loading {split} split from Hugging Face...")
        ds = load_dataset("chainyo/rvl-cdip", split=split, streaming=True)
        print(f"Saving {split}...")
        if use_s3:
            save_split_s3(ds, split, bucket, prefix)
        else:
            save_split_local(ds, split, output_dir)

    if use_s3:
        print(f"\nDone. Dataset written to s3://{bucket}/{prefix}/")
    else:
        print(f"\nDone. Dataset written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
