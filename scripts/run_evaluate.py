"""
Run the evaluation step standalone against an existing model artifact.

Usage:
  uv run scripts/run_evaluate.py
  uv run scripts/run_evaluate.py --model-s3 s3://... --test-s3 s3://...
"""

import argparse

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str,
                        default="arn:aws:iam::898322960370:role/service-role/AmazonSageMaker-ExecutionRole-20260406T143837")
    parser.add_argument("--region", type=str, default="us-east-2")
    parser.add_argument("--bucket", type=str, default="docusense-data")
    parser.add_argument("--model-s3", type=str,
                        default="s3://sagemaker-us-east-2-898322960370/pipelines-s33nr7mmrwhq-Train-SXbRhAvK7V/output/model.tar.gz")
    parser.add_argument("--test-s3", type=str,
                        default="s3://docusense-data/pipeline-output/preprocessed/test")
    parser.add_argument("--output-s3", type=str,
                        default="s3://docusense-data/pipeline-output/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boto3.setup_default_session(region_name=args.region)
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    processor = ScriptProcessor(
        image_uri=f"763104351884.dkr.ecr.{args.region}.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=args.role,
        sagemaker_session=sagemaker_session,
        env={"PYTHONPATH": "/opt/ml/processing/input/deps"},
    )

    processor.run(
        code="training/evaluate.py",
        inputs=[
            ProcessingInput(
                source=args.model_s3,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=args.test_s3,
                destination="/opt/ml/processing/input/data",
            ),
            ProcessingInput(
                source=f"s3://{args.bucket}/code/",
                destination="/opt/ml/processing/input/deps",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics",
                source="/opt/ml/processing/output",
                destination=args.output_s3,
            ),
        ],
        arguments=[
            "--model-path", "/opt/ml/processing/input/model/best_model.pt",
            "--data-dir",   "/opt/ml/processing/input/data",
            "--output-dir", "/opt/ml/processing/output",
        ],
        wait=False,
        logs=False,
    )

    job_name = processor.latest_job.job_name
    print(f"Evaluation job started: {job_name}")
    print(f"Monitor at: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/processing-jobs/{job_name}")


if __name__ == "__main__":
    main()
