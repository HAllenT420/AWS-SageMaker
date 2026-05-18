"""
Module 6: Model Registry

Responsible for:
- Registering trained models with version numbers
- Managing approval status
- Tracking model lineage
"""

import logging

import sagemaker
from sagemaker.model import Model

logger = logging.getLogger(__name__)


def register_model(
    session,
    role_arn: str,
    region: str,
    model_data_s3: str,
    model_package_group_name: str,
    approval_status: str = 'PendingManualApproval',
    description: str = ''
) -> str:
    """
    Register a model in SageMaker Model Registry.

    Args:
        session: SageMaker session
        role_arn: IAM role
        region: AWS region
        model_data_s3: S3 path to model.tar.gz
        model_package_group_name: Name of the model group
        approval_status: Initial status (PendingManualApproval/Approved)
        description: Model description

    Returns:
        Model package ARN
    """
    logger.info("=" * 50)
    logger.info("STEP 6: MODEL REGISTRATION")
    logger.info("=" * 50)

    # Get XGBoost image for inference
    xgb_image = sagemaker.image_uris.retrieve(
        framework='xgboost', region=region, version='1.7-1'
    )

    model = Model(
        image_uri=xgb_image,
        model_data=model_data_s3,
        role=role_arn,
        sagemaker_session=session
    )

    # Ensure model package group exists
    sm_client = session.boto_session.client('sagemaker')
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription='Customer churn prediction models'
        )
        logger.info(f"Created model package group: {model_package_group_name}")
    except sm_client.exceptions.ClientError:
        logger.info(f"Model package group already exists: {model_package_group_name}")

    # Register
    model_package = model.register(
        model_package_group_name=model_package_group_name,
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=['ml.m5.large', 'ml.m5.xlarge'],
        transform_instances=['ml.m5.large'],
        approval_status=approval_status,
        description=description
    )

    model_arn = model_package.model_package_arn
    logger.info(f"Model registered: {model_arn}")
    logger.info(f"Status: {approval_status}")

    return model_arn


def approve_model(session, model_package_arn: str):
    """Approve a model for deployment."""
    sm_client = session.boto_session.client('sagemaker')
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus='Approved'
    )
    logger.info(f"Model APPROVED: {model_package_arn}")
