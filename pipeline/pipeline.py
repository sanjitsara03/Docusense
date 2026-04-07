"""
DocuSense SageMaker Pipeline.

Steps:
  1. PreprocessingStep  — validate + resize images (ml.m5.xlarge)
  2. TrainingStep       — EfficientNet-B0 fine-tune (ml.p3.2xlarge spot)
  3. EvaluationStep     — metrics + Grad-CAM (ml.m5.xlarge)
  4. ConditionStep      — register model only if accuracy > 0.90

Usage:
  python pipeline/pipeline.py --role <iam-role-arn> --bucket <s3-bucket>

"""

import argparse

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str,
                        default="arn:aws:iam::898322960370:role/service-role/AmazonSageMaker-ExecutionRole-20260406T143837")
    parser.add_argument("--bucket", type=str, default="docusense-data")
    parser.add_argument("--prefix", type=str, default="rvl-cdip")
    parser.add_argument("--region", type=str, default="us-east-2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accuracy-threshold", type=float, default=0.90)
    parser.add_argument("--run", action="store_true",
                        help="Start the pipeline execution after creating it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boto3.setup_default_session(region_name=args.region)
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    pipeline_session = PipelineSession(boto_session=boto_session)

    s3_data_uri = f"s3://{args.bucket}/{args.prefix}"
    s3_output_uri = f"s3://{args.bucket}/pipeline-output"

    # Pipeline parameters 
    p_epochs = ParameterInteger(name="Epochs", default_value=args.epochs)
    p_batch_size = ParameterInteger(name="BatchSize", default_value=args.batch_size)
    p_lr = ParameterFloat(name="LearningRate", default_value=args.lr)
    p_accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=args.accuracy_threshold)


    # Step 1: Preprocessing
    processor = ScriptProcessor(
        image_uri=f"763104351884.dkr.ecr.{args.region}.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=args.role,
        sagemaker_session=sagemaker_session,
    )

    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=processor,
        code="pipeline/preprocess.py",
        inputs=[
            ProcessingInput(
                source=s3_data_uri,
                destination="/opt/ml/processing/input/data",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"{s3_output_uri}/preprocessed/train",
            ),
            ProcessingOutput(
                output_name="val",
                source="/opt/ml/processing/output/val",
                destination=f"{s3_output_uri}/preprocessed/val",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"{s3_output_uri}/preprocessed/test",
            ),
        ],
    )

    # Step 2: Training (spot instance)
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=args.role,
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        framework_version="2.1.0",
        py_version="py310",
        use_spot_instances=False,
        max_run=3600,           # 1 hour max
        checkpoint_s3_uri=f"{s3_output_uri}/checkpoints",
        checkpoint_local_path="/opt/ml/checkpoints",
        hyperparameters={
            "epochs": p_epochs,
            "batch-size": p_batch_size,
            "lr": p_lr,
        },
        sagemaker_session=sagemaker_session,
    )

    train_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            ),
            "val": sagemaker.inputs.TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
            ),
        },
    )

    # Step 3: Evaluation
    eval_processor = ScriptProcessor(
        image_uri=f"763104351884.dkr.ecr.{args.region}.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=args.role,
        sagemaker_session=sagemaker_session,
    )

    eval_report = PropertyFile(
        name="EvalReport",
        output_name="metrics",
        path="metrics.json",
    )

    eval_step = ProcessingStep(
        name="Evaluate",
        processor=eval_processor,
        code="training/evaluate.py",
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics",
                source="/opt/ml/processing/output",
                destination=f"{s3_output_uri}/evaluation",
            ),
        ],
        job_arguments=[
            "--model-path", "/opt/ml/processing/input/model/best_model.pt",
            "--data-dir", "/opt/ml/processing/input/data",
            "--output-dir", "/opt/ml/processing/output",
        ],
        property_files=[eval_report],
    )

    # 
    # Step 4: Conditional registration (accuracy > threshold)
    model = Model(
        image_uri=f"763104351884.dkr.ecr.{args.region}.amazonaws.com/pytorch-inference:2.1.0-cpu-py310",
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=args.role,
        sagemaker_session=pipeline_session,
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["image/png", "image/jpeg"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="DocuSenseModels",
            approval_status="PendingManualApproval",
        ),
    )

    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=eval_report,
            json_path="accuracy",
        ),
        right=p_accuracy_threshold,
    )

    condition_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[accuracy_condition],
        if_steps=[register_step],
        else_steps=[],
    )

    # Build + run pipeline
    pipeline = Pipeline(
        name="DocuSensePipeline",
        parameters=[p_epochs, p_batch_size, p_lr, p_accuracy_threshold],
        steps=[preprocess_step, train_step, eval_step, condition_step],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=args.role)
    print("Pipeline upserted: DocuSensePipeline")

    if args.run:
        execution = pipeline.start()
        print(f"Pipeline started: {execution.arn}")
        print(f"Monitor at: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/pipelines/DocuSensePipeline")


if __name__ == "__main__":
    main()

