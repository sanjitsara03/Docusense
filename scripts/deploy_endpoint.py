#Deploys the latest approved model from the DocuSenseModels registry to a
#sageMaker endpoint


import argparse

import boto3
import sagemaker
from sagemaker.model import ModelPackage

ROLE = "arn:aws:iam::898322960370:role/service-role/AmazonSageMaker-ExecutionRole-20260406T143837"
REGION = "us-east-2"
ENDPOINT_NAME = "docusense-endpoint"
MODEL_PACKAGE_GROUP = "DocuSenseModels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--role", type=str, default=ROLE)
    parser.add_argument("--endpoint-name", type=str, default=ENDPOINT_NAME)
    return parser.parse_args()


def get_latest_approved_model(sm_client, group_name: str) -> str:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(
            f"No approved models found in {group_name}. "
            "Approve a model in the AWS console first."
        )
    arn = packages[0]["ModelPackageArn"]
    print(f"Using model: {arn}")
    return arn


def main() -> None:
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_client = boto_session.client("sagemaker")
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    model_package_arn = get_latest_approved_model(sm_client, MODEL_PACKAGE_GROUP)

    model = ModelPackage(
        role=args.role,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )

    print(f"Deploying to endpoint: {args.endpoint_name}")
    print("This will take ~5-10 minutes...")

    model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=args.endpoint_name,
    )

    print(f"\nEndpoint is live: {args.endpoint_name}")
    print("Set SAGEMAKER_ENDPOINT_NAME={args.endpoint_name} in your FastAPI env.")
    print("Run delete_endpoint.py when done to avoid idle charges.")


if __name__ == "__main__":
    main()
