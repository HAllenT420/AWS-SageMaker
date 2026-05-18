"""
Module 7: Model Deployment

Responsible for:
- Deploying model to SageMaker Endpoint
- Supporting multiple deployment strategies
- Handling endpoint updates
"""

import logging

import sagemaker
from sagemaker.model import Model

logger = logging.getLogger(__name__)


def deploy_model(
    session,
    role_arn: str,
    region: str,
    model_data_s3: str,
    endpoint_name: str,
    instance_type: str = 'ml.m5.large',
    instance_count: int = 1
):
    """
    Deploy model to a real-time SageMaker Endpoint.

    Args:
        session: SageMaker session
        role_arn: IAM role
        region: AWS region
        model_data_s3: S3 path to model.tar.gz
        endpoint_name: Name for the endpoint
        instance_type: Inference instance type
        instance_count: Number of instances

    Returns:
        SageMaker Predictor object
    """
    logger.info("=" * 50)
    logger.info("STEP 7: MODEL DEPLOYMENT")
    logger.info("=" * 50)

    xgb_image = sagemaker.image_uris.retrieve(
        framework='xgboost', region=region, version='1.7-1'
    )

    model = Model(
        image_uri=xgb_image,
        model_data=model_data_s3,
        role=role_arn,
        sagemaker_session=session
    )

    logger.info(f"Deploying to endpoint: {endpoint_name}")
    logger.info(f"Instance: {instance_type} x {instance_count}")

    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        serializer=sagemaker.serializers.CSVSerializer(),
        deserializer=sagemaker.deserializers.CSVDeserializer()
    )

    logger.info(f"✅ Endpoint deployed: {endpoint_name}")
    return predictor


def delete_endpoint(session, endpoint_name: str):
    """Delete an endpoint to stop charges."""
    sm_client = session.boto_session.client('sagemaker')
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    logger.info(f"Endpoint deleted: {endpoint_name}")


def update_endpoint(session, endpoint_name: str, new_model_data_s3: str, role_arn: str, region: str):
    """
    Update an existing endpoint with a new model (zero-downtime).

    This is how you deploy a retrained model without downtime.
    """
    logger.info(f"Updating endpoint {endpoint_name} with new model...")

    xgb_image = sagemaker.image_uris.retrieve(
        framework='xgboost', region=region, version='1.7-1'
    )

    model = Model(
        image_uri=xgb_image,
        model_data=new_model_data_s3,
        role=role_arn,
        sagemaker_session=session
    )

    # Create new model
    model_name = model.create(instance_type='ml.m5.large')

    # Update endpoint config
    sm_client = session.boto_session.client('sagemaker')
    new_config_name = f'{endpoint_name}-config-new'

    sm_client.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large',
            'InitialVariantWeight': 1.0
        }]
    )

    # Update endpoint (zero-downtime rolling update)
    sm_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=new_config_name
    )

    logger.info(f"✅ Endpoint updated with new model (zero-downtime)")
