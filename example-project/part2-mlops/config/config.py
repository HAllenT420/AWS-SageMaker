"""
Centralized Configuration

Single source of truth for all project settings.
Reads from params.json and allows environment variable overrides.
"""

import json
import os
from pathlib import Path

import boto3
import sagemaker


def load_config() -> dict:
    """Load configuration from params.json with env var overrides."""
    config_path = Path(__file__).parent / 'params.json'
    with open(config_path) as f:
        config = json.load(f)

    # Allow environment variable overrides
    config['region'] = os.environ.get('AWS_REGION', config['region'])
    config['role_arn'] = os.environ.get('SAGEMAKER_ROLE_ARN', config['role_arn'])

    return config


def get_session(config: dict = None):
    """Create SageMaker session."""
    if config is None:
        config = load_config()

    boto_session = boto3.Session(region_name=config['region'])
    sm_session = sagemaker.Session(boto_session=boto_session)
    return sm_session


def get_s3_paths(config: dict = None) -> dict:
    """Generate standard S3 paths."""
    if config is None:
        config = load_config()

    session = get_session(config)
    bucket = session.default_bucket()
    prefix = config['project_name']

    return {
        'bucket': bucket,
        'prefix': prefix,
        'raw_data': f's3://{bucket}/{prefix}/data/raw',
        'processed': f's3://{bucket}/{prefix}/data/processed',
        'train': f's3://{bucket}/{prefix}/data/processed/train',
        'validation': f's3://{bucket}/{prefix}/data/processed/validation',
        'test': f's3://{bucket}/{prefix}/data/processed/test',
        'models': f's3://{bucket}/{prefix}/models',
        'predictions': f's3://{bucket}/{prefix}/predictions',
        'evaluation': f's3://{bucket}/{prefix}/evaluation',
    }
