"""
Delete the DocuSense SageMaker endpoint to stop billing.

Run this after demos or when done testing.
Endpoints bill ~$0.115/hr on ml.m5.large even when idle.

Usage:
  python scripts/delete_endpoint.py
"""

import argparse

import boto3

REGION = "us-east-2"
ENDPOINT_NAME = "docusense-endpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--endpoint-name", type=str, default=ENDPOINT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sm_client = boto3.Session(region_name=args.region).client("sagemaker")

    sm_client.delete_endpoint(EndpointName=args.endpoint_name)
    print(f"Deleted endpoint: {args.endpoint_name}")


if __name__ == "__main__":
    main()
