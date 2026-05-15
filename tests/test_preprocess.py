"""
Unit tests for preprocessing logic.

Run locally with: pytest tests/
These tests validate your preprocessing logic before running on SageMaker.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'customer_id': range(n),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(20, 100, n).round(2),
        'total_charges': np.random.uniform(100, 7000, n).round(2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check'], n),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
        'online_security': np.random.choice(['Yes', 'No'], n),
        'tech_support': np.random.choice(['Yes', 'No'], n),
        'num_support_tickets': np.random.poisson(2, n),
        'churn': np.random.choice([0, 1], n),
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_preprocess_creates_splits(sample_data, temp_dir):
    """Test that preprocessing creates train/val/test splits."""
    from preprocess import preprocess

    input_path = os.path.join(temp_dir, 'raw_data.csv')
    output_path = os.path.join(temp_dir, 'output')
    sample_data.to_csv(input_path, index=False)

    # Monkey-patch the SageMaker paths
    preprocess(input_path, output_path)

    assert os.path.exists(os.path.join(output_path, 'train', 'train.csv'))
    assert os.path.exists(os.path.join(output_path, 'validation', 'validation.csv'))
    assert os.path.exists(os.path.join(output_path, 'test', 'test.csv'))


def test_preprocess_drops_id_columns(sample_data, temp_dir):
    """Test that ID columns are dropped."""
    from preprocess import preprocess

    input_path = os.path.join(temp_dir, 'raw_data.csv')
    output_path = os.path.join(temp_dir, 'output')
    sample_data.to_csv(input_path, index=False)

    preprocess(input_path, output_path)

    train_df = pd.read_csv(os.path.join(output_path, 'train', 'train.csv'), header=None)
    # Original has 11 columns, minus customer_id = 10 columns
    assert train_df.shape[1] == 10


def test_preprocess_target_first_column(sample_data, temp_dir):
    """Test that target (churn) is the first column."""
    from preprocess import preprocess

    input_path = os.path.join(temp_dir, 'raw_data.csv')
    output_path = os.path.join(temp_dir, 'output')
    sample_data.to_csv(input_path, index=False)

    preprocess(input_path, output_path)

    train_df = pd.read_csv(os.path.join(output_path, 'train', 'train.csv'), header=None)
    # First column should be binary (0 or 1)
    assert set(train_df.iloc[:, 0].unique()).issubset({0, 1})


def test_preprocess_split_ratios(sample_data, temp_dir):
    """Test that split ratios are approximately correct."""
    from preprocess import preprocess

    input_path = os.path.join(temp_dir, 'raw_data.csv')
    output_path = os.path.join(temp_dir, 'output')
    sample_data.to_csv(input_path, index=False)

    preprocess(input_path, output_path, test_size=0.3, val_size=0.5)

    train_df = pd.read_csv(os.path.join(output_path, 'train', 'train.csv'), header=None)
    val_df = pd.read_csv(os.path.join(output_path, 'validation', 'validation.csv'), header=None)
    test_df = pd.read_csv(os.path.join(output_path, 'test', 'test.csv'), header=None)

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(sample_data)

    # Approximate ratios (70/15/15)
    assert 0.6 < len(train_df) / total < 0.8
    assert 0.1 < len(val_df) / total < 0.2
    assert 0.1 < len(test_df) / total < 0.2
