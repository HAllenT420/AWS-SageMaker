"""
Module 1: Data Ingestion

Responsible for:
- Connecting to data sources (DB, API, files)
- Pulling raw data
- Saving to S3 (data lake)

In production, replace generate_synthetic_data() with your actual data source.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer churn data.

    In production, replace this with:
        - pd.read_sql("SELECT * FROM customers", db_connection)
        - pd.read_parquet("s3://data-lake/customers/")
        - requests.get("https://api.company.com/customers")

    Args:
        n_samples: Number of records to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with customer features and churn label
    """
    logger.info(f"Generating {n_samples} synthetic records (seed={seed})")
    np.random.seed(seed)

    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 100, n_samples).round(2),
        'total_charges': np.random.uniform(100, 7000, n_samples).round(2),
        'contract_type': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]
        ),
        'payment_method': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples
        ),
        'internet_service': np.random.choice(
            ['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.2]
        ),
        'online_security': np.random.choice(['Yes', 'No'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples),
        'num_support_tickets': np.random.poisson(2, n_samples),
    })

    # Create target with realistic logic
    churn_prob = (
        0.3 * (data['contract_type'] == 'Month-to-month').astype(float) +
        0.2 * (data['tenure_months'] < 12).astype(float) +
        0.15 * (data['monthly_charges'] > 70).astype(float) +
        0.1 * (data['num_support_tickets'] > 3).astype(float) +
        np.random.uniform(0, 0.3, n_samples)
    )
    data['churn'] = (churn_prob > 0.5).astype(int)

    logger.info(f"Data shape: {data.shape}, Churn rate: {data['churn'].mean():.2%}")
    return data


def upload_raw_data(data: pd.DataFrame, session, s3_path: str) -> str:
    """
    Upload raw data to S3.

    Args:
        data: Raw DataFrame
        session: SageMaker session
        s3_path: S3 destination path

    Returns:
        S3 URI of uploaded file
    """
    local_path = Path('/tmp/raw_data.csv')
    data.to_csv(local_path, index=False)

    bucket = s3_path.split('/')[2]
    key_prefix = '/'.join(s3_path.split('/')[3:])

    s3_uri = session.upload_data(
        path=str(local_path),
        bucket=bucket,
        key_prefix=key_prefix
    )
    logger.info(f"Raw data uploaded to: {s3_uri}")
    return s3_uri


def ingest(session, s3_paths: dict, n_samples: int = 5000) -> str:
    """
    Main ingestion function. Generates/pulls data and uploads to S3.

    Args:
        session: SageMaker session
        s3_paths: Dict of S3 paths from config
        n_samples: Number of samples

    Returns:
        S3 URI of raw data
    """
    logger.info("=" * 50)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 50)

    data = generate_synthetic_data(n_samples=n_samples)
    s3_uri = upload_raw_data(data, session, s3_paths['raw_data'])

    logger.info(f"Ingestion complete. Data at: {s3_uri}")
    return s3_uri
