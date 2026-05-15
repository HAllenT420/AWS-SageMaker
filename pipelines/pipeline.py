"""
SageMaker Pipeline Definition

This is the MLOps version of the notebook workflow. It defines a repeatable,
automated pipeline that can be triggered on schedule or by new data.

Pipeline Steps:
    1. Preprocessing (Processing Job)
    2. Training (Training Job)
    3. Evaluation (Processing Job)
    4. Conditional Registration (if metrics pass threshold)

Run this script to create/update and execute the pipeline:
    python pipelines/pipeline.py
"""

import json
import os
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.config import get_config, get_s3_paths


def create_pipeline():
    """Create the SageMaker Pipeline."""
    config = get_config()
    region = config['region']

    # Initialize session
    boto_session = boto3.Session(region_name=region)
    session = sagemaker.Session(boto_session=boto_session)
    bucket = session.default_bucket()
    prefix = config['project_name']
    role = config['role_arn']
    s3_paths = get_s3_paths(bucket, prefix)

    scripts_dir = str(Path(__file__).parent.parent / 'scripts')

    # =========================================================================
    # Pipeline Parameters (can be overridden at runtime)
    # =========================================================================
    param_processing_instance_type = ParameterString(
        name='ProcessingInstanceType',
        default_value=config['processing_instance_type']
    )
    param_training_instance_type = ParameterString(
        name='TrainingInstanceType',
        default_value=config['training_instance_type']
    )
    param_input_data = ParameterString(
        name='InputDataUrl',
        default_value=f'{s3_paths["raw_data"]}/raw_data.csv'
    )
    param_model_approval_status = ParameterString(
        name='ModelApprovalStatus',
        default_value=config['model_approval_status']
    )
    param_auc_threshold = ParameterFloat(
        name='AucThreshold',
        default_value=0.7
    )

    # =========================================================================
    # Step 1: Preprocessing
    # =========================================================================
    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role,
        instance_type=param_processing_instance_type,
        instance_count=1,
        sagemaker_session=session,
        base_job_name=f'{prefix}-preprocess'
    )

    step_preprocess = ProcessingStep(
        name='Preprocess',
        processor=sklearn_processor,
        code=os.path.join(scripts_dir, 'preprocess.py'),
        inputs=[
            ProcessingInput(
                source=param_input_data,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/output/train'),
            ProcessingOutput(output_name='validation', source='/opt/ml/processing/output/validation'),
            ProcessingOutput(output_name='test', source='/opt/ml/processing/output/test'),
        ],
        job_arguments=['--test-size', '0.3', '--val-size', '0.5']
    )

    # =========================================================================
    # Step 2: Training
    # =========================================================================
    xgb_image_uri = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.7-1'
    )

    xgb_estimator = Estimator(
        image_uri=xgb_image_uri,
        role=role,
        instance_count=1,
        instance_type=param_training_instance_type,
        output_path=s3_paths['models'],
        sagemaker_session=session,
        base_job_name=f'{prefix}-train'
    )
    xgb_estimator.set_hyperparameters(**config['hyperparameters'])

    step_train = TrainingStep(
        name='Train',
        estimator=xgb_estimator,
        inputs={
            'train': TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                content_type='text/csv'
            ),
            'validation': TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )

    # =========================================================================
    # Step 3: Evaluation
    # =========================================================================
    evaluation_report = PropertyFile(
        name='EvaluationReport',
        output_name='evaluation',
        path='evaluation.json'
    )

    step_evaluate = ProcessingStep(
        name='Evaluate',
        processor=sklearn_processor,
        code=os.path.join(scripts_dir, 'evaluate.py'),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[
            ProcessingOutput(output_name='evaluation', source='/opt/ml/processing/evaluation')
        ],
        property_files=[evaluation_report]
    )

    # =========================================================================
    # Step 4: Conditional Model Registration
    # =========================================================================
    model_metrics = sagemaker.model_metrics.ModelMetrics(
        model_statistics=sagemaker.model_metrics.MetricsSource(
            s3_uri='{}/evaluation.json'.format(
                step_evaluate.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
            ),
            content_type='application/json'
        )
    )

    step_register = RegisterModel(
        name='RegisterModel',
        estimator=xgb_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=['ml.m5.large', 'ml.m5.xlarge'],
        transform_instances=['ml.m5.large'],
        model_package_group_name=config['model_package_group_name'],
        approval_status=param_model_approval_status,
        model_metrics=model_metrics
    )

    # Condition: Only register if AUC >= threshold
    cond_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path='binary_classification_metrics.auc.value'
        ),
        right=param_auc_threshold
    )

    step_condition = ConditionStep(
        name='CheckAUC',
        conditions=[cond_auc],
        if_steps=[step_register],
        else_steps=[]  # Do nothing if AUC is below threshold
    )

    # =========================================================================
    # Assemble Pipeline
    # =========================================================================
    pipeline = Pipeline(
        name=config['pipeline_name'],
        parameters=[
            param_processing_instance_type,
            param_training_instance_type,
            param_input_data,
            param_model_approval_status,
            param_auc_threshold,
        ],
        steps=[step_preprocess, step_train, step_evaluate, step_condition],
        sagemaker_session=session
    )

    return pipeline


def main():
    """Create/update and start the pipeline."""
    print("=" * 60)
    print("SageMaker Pipeline - Customer Churn Prediction")
    print("=" * 60)

    config = get_config()

    print(f"\nCreating pipeline: {config['pipeline_name']}")
    pipeline = create_pipeline()

    # Upsert (create or update) the pipeline
    print("Upserting pipeline definition...")
    pipeline.upsert(role_arn=config['role_arn'])
    print("Pipeline upserted successfully.")

    # Start execution
    print("\nStarting pipeline execution...")
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    print(f"\nMonitor in AWS Console:")
    print(f"  https://{config['region']}.console.aws.amazon.com/sagemaker/home?region={config['region']}#/pipelines")

    # Optionally wait for completion
    # execution.wait()
    # print("Pipeline execution complete!")

    return execution


if __name__ == '__main__':
    main()
