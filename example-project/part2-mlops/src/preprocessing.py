"""
Module 2: Preprocessing

Responsible for:
- Launching a SageMaker Processing Job
- The actual logic lives in scripts/preprocess.py (runs on SageMaker infra)
- This module just configures and submits the job
"""

import logging

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

logger = logging.getLogger(__name__)


def run_preprocessing(
    session,
    role_arn: str,
    input_s3_uri: str,
    output_s3_base: str,
    instance_type: str = 'ml.m5.xlarge',
    scripts_dir: str = 'scripts'
) -> dict:
    """
    Launch a SageMaker Processing Job for data preprocessing.

    What happens:
    1. SageMaker spins up an ml.m5.xlarge instance
    2. Installs scikit-learn
    3. Downloads your data from S3
    4. Runs scripts/preprocess.py
    5. Uploads results back to S3
    6. Terminates the instance

    Args:
        session: SageMaker session
        role_arn: IAM role ARN
        input_s3_uri: S3 path to raw data
        output_s3_base: S3 base path for processed outputs
        instance_type: Processing instance type
        scripts_dir: Path to scripts directory

    Returns:
        Dict with S3 URIs for train, validation, test data
    """
    logger.info("=" * 50)
    logger.info("STEP 2: PREPROCESSING (SageMaker Processing Job)")
    logger.info("=" * 50)

    processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=session,
        base_job_name='churn-preprocess'
    )

    logger.info(f"Input: {input_s3_uri}")
    logger.info(f"Output: {output_s3_base}")
    logger.info(f"Instance: {instance_type}")

    processor.run(
        code=f'{scripts_dir}/preprocess.py',
        inputs=[
            ProcessingInput(
                source=input_s3_uri,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='train',
                source='/opt/ml/processing/output/train',
                destination=f'{output_s3_base}/train'
            ),
            ProcessingOutput(
                output_name='validation',
                source='/opt/ml/processing/output/validation',
                destination=f'{output_s3_base}/validation'
            ),
            ProcessingOutput(
                output_name='test',
                source='/opt/ml/processing/output/test',
                destination=f'{output_s3_base}/test'
            ),
        ],
        arguments=['--test-size', '0.3', '--val-size', '0.5']
    )

    output_paths = {
        'train': f'{output_s3_base}/train',
        'validation': f'{output_s3_base}/validation',
        'test': f'{output_s3_base}/test',
    }

    logger.info("Preprocessing complete.")
    for name, path in output_paths.items():
        logger.info(f"  {name}: {path}")

    return output_paths
