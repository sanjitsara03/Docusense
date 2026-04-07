
#Manually registers a model artifact in the SageMaker Model Registry.


import argparse

import boto3
import sagemaker
from sagemaker.model import Model

ROLE = "arn:aws:iam::898322960370:role/service-role/AmazonSageMaker-ExecutionRole-20260406T143837"
REGION = "us-east-2"
MODEL_PACKAGE_GROUP = "DocuSenseModels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-data", type=str, required=True,
                        help="S3 URI to model.tar.gz")
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--role", type=str, default=ROLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_client = boto_session.client("sagemaker")

    image_uri = f"763104351884.dkr.ecr.{args.region}.amazonaws.com/pytorch-inference:2.1.0-cpu-py310"

    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageDescription="Manually registered DocuSense model",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": args.model_data,
                }
            ],
            "SupportedContentTypes": ["image/png", "image/jpeg"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
        },
        ModelApprovalStatus="PendingManualApproval",
    )

    arn = response["ModelPackageArn"]
    print(f"Registered: {arn}")
    print(f"Approve at: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/model-registry/{MODEL_PACKAGE_GROUP}")


if __name__ == "__main__":
    main()
