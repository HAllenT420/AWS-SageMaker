"""
Module 4: Model Training

Responsible for:
- Configuring the SageMaker Training Job
- Launching training on managed infrastructure
- Returning the model artifact location
"""

import logging

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

logger = logging.getLogger(__name__)


def train_model(
    session,
    role_arn: str,
    region: str,
    train_s3_uri: str,
    validation_s3_uri: str,
    output_s3_uri: str,
    hyperparameters: dict,
    instance_type: str = 'ml.m5.xlarge',
    instance_count: int = 1,
    use_spot: bool = False
) -> Estimator:
    """
    Launch a SageMaker Training Job.

    What happens:
    1. SageMaker provisions the instance
    2. Pulls the XGBoost Docker container
    3. Downloads training data from S3
    4. Trains the model with your hyperparameters
    5. Uploads model.tar.gz to S3
    6. Terminates the instance

    Args:
        session: SageMaker session
        role_arn: IAM role
        region: AWS region
        train_s3_uri: S3 path to training data
        validation_s3_uri: S3 path to validation data
        output_s3_uri: S3 path for model output
        hyperparameters: Dict of XGBoost hyperparameters
        instance_type: Training instance type
        instance_count: Number of instances
        use_spot: Whether to use spot instances (up to 90% savings)

    Returns:
        Fitted Estimator object (contains model_data path)
    """
    logger.info("=" * 50)
    logger.info("STEP 4: MODEL TRAINING (SageMaker Training Job)")
    logger.info("=" * 50)

    # Get XGBoost container image
    xgb_image = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.7-1'
    )
    logger.info(f"XGBoost image: {xgb_image}")

    # Configure estimator
    estimator_kwargs = {
        'image_uri': xgb_image,
        'role': role_arn,
        'instance_count': instance_count,
        'instance_type': instance_type,
        'output_path': output_s3_uri,
        'sagemaker_session': session,
        'base_job_name': 'churn-train',
    }

    # Add spot instance config if enabled
    if use_spot:
        estimator_kwargs['use_spot_instances'] = True
        estimator_kwargs['max_wait'] = 7200
        estimator_kwargs['max_run'] = 3600
        logger.info("Spot instances ENABLED (up to 90% cost savings)")

    estimator = Estimator(**estimator_kwargs)
    estimator.set_hyperparameters(**hyperparameters)

    logger.info(f"Instance: {instance_type} x {instance_count}")
    logger.info(f"Hyperparameters: {hyperparameters}")
    logger.info("Starting training job...")

    # Launch training
    estimator.fit(
        inputs={
            'train': TrainingInput(s3_data=train_s3_uri, content_type='text/csv'),
            'validation': TrainingInput(s3_data=validation_s3_uri, content_type='text/csv')
        },
        wait=True,
        logs='All'
    )

    logger.info(f"Training complete!")
    logger.info(f"Model artifact: {estimator.model_data}")

    return estimator
