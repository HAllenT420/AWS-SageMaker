"""
Module 5: Model Evaluation

Responsible for:
- Running evaluation on test data
- Checking quality gates
- Producing evaluation report
"""

import json
import logging

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

logger = logging.getLogger(__name__)


def evaluate_model(
    session,
    role_arn: str,
    model_data_s3: str,
    test_s3_uri: str,
    output_s3_uri: str,
    instance_type: str = 'ml.m5.xlarge',
    scripts_dir: str = 'scripts'
) -> str:
    """
    Launch evaluation Processing Job.

    Args:
        session: SageMaker session
        role_arn: IAM role
        model_data_s3: S3 path to model.tar.gz
        test_s3_uri: S3 path to test data
        output_s3_uri: S3 path for evaluation output
        instance_type: Processing instance type
        scripts_dir: Path to scripts directory

    Returns:
        S3 URI of evaluation report
    """
    logger.info("=" * 50)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("=" * 50)

    processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=session,
        base_job_name='churn-evaluate'
    )

    processor.run(
        code=f'{scripts_dir}/evaluate.py',
        inputs=[
            ProcessingInput(
                source=model_data_s3,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=test_s3_uri,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='evaluation',
                source='/opt/ml/processing/evaluation',
                destination=output_s3_uri
            )
        ]
    )

    eval_report_uri = f'{output_s3_uri}/evaluation.json'
    logger.info(f"Evaluation report: {eval_report_uri}")
    return eval_report_uri


def check_quality_gates(eval_report_s3: str, session, quality_gates: dict) -> bool:
    """
    Check if model metrics pass quality thresholds.

    Args:
        eval_report_s3: S3 URI of evaluation.json
        session: SageMaker session
        quality_gates: Dict with min thresholds

    Returns:
        True if all gates pass, False otherwise
    """
    import boto3

    logger.info("Checking quality gates...")

    # Download evaluation report
    bucket = eval_report_s3.split('/')[2]
    key = '/'.join(eval_report_s3.split('/')[3:])

    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    metrics = json.loads(response['Body'].read().decode('utf-8'))

    # Extract metrics
    auc = metrics['binary_classification_metrics']['auc']['value']
    precision = metrics['binary_classification_metrics']['precision']['value']
    recall = metrics['binary_classification_metrics']['recall']['value']

    logger.info(f"  AUC:       {auc:.4f} (min: {quality_gates['min_auc']})")
    logger.info(f"  Precision: {precision:.4f} (min: {quality_gates['min_precision']})")
    logger.info(f"  Recall:    {recall:.4f} (min: {quality_gates['min_recall']})")

    # Check gates
    passed = (
        auc >= quality_gates['min_auc'] and
        precision >= quality_gates['min_precision'] and
        recall >= quality_gates['min_recall']
    )

    if passed:
        logger.info("✅ ALL quality gates PASSED")
    else:
        logger.warning("❌ Quality gates FAILED — model will NOT be registered")

    return passed
