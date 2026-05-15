"""
Pipeline Configuration

Centralizes all configuration for the SageMaker Pipeline.
Reads from config/pipeline_params.json and environment variables.
"""

import json
import os
from pathlib import Path


def get_config():
    """Load pipeline configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'pipeline_params.json'

    with open(config_path) as f:
        config = json.load(f)

    # Allow environment variable overrides (useful for CI/CD)
    config['region'] = os.environ.get('AWS_REGION', config['region'])
    config['role_arn'] = os.environ.get(
        'SAGEMAKER_ROLE_ARN',
        'arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE'
    )

    return config


# S3 paths helper
def get_s3_paths(bucket, prefix):
    """Generate standard S3 paths for the project."""
    return {
        'raw_data': f's3://{bucket}/{prefix}/data/raw',
        'processed_train': f's3://{bucket}/{prefix}/data/processed/train',
        'processed_validation': f's3://{bucket}/{prefix}/data/processed/validation',
        'processed_test': f's3://{bucket}/{prefix}/data/processed/test',
        'models': f's3://{bucket}/{prefix}/models',
        'evaluation': f's3://{bucket}/{prefix}/evaluation',
    }
