"""
Generate a labels/<split>.txt from images already in S3.

Walks s3://bucket/prefix/images/<split>/<class>/*.png and writes
the label file without downloading any images.

Usage:
  python scripts/generate_labels.py --s3-uri s3://docusense-data/rvl-cdip/ --split test
"""

import argparse
import boto3

CLASS_TO_LABEL = {
    "letter": 0,
    "form": 1,
    "email": 2,
    "budget": 3,
    "invoice": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-uri", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    s3_uri = args.s3_uri.rstrip("/")
    without_scheme = s3_uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    s3 = boto3.client("s3")
    label_lines: list[str] = []

    for class_name, remapped in CLASS_TO_LABEL.items():
        img_prefix = f"{prefix}/images/{args.split}/{class_name}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=img_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # rel_path is everything after prefix/
                rel_path = key[len(prefix) + 1:]
                label_lines.append(f"{rel_path} {remapped}")

        print(f"  {class_name}: {sum(1 for l in label_lines if f'/{class_name}/' in l)} images")

    label_key = f"{prefix}/labels/{args.split}.txt"
    s3.put_object(
        Bucket=bucket,
        Key=label_key,
        Body=("\n".join(label_lines) + "\n").encode(),
    )
    print(f"\nUploaded s3://{bucket}/{label_key} ({len(label_lines)} lines)")


if __name__ == "__main__":
    main()
