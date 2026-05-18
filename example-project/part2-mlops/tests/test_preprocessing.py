"""
Unit tests for preprocessing logic.

Run with: uv run pytest tests/ -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_data():
    """Create sample data matching our schema."""
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


def test_data_ingestion_generates_correct_shape():
    """Test that synthetic data has expected columns."""
    from src.data_ingestion import generate_synthetic_data
    data = generate_synthetic_data(n_samples=100)
    assert data.shape == (100, 11)
    assert 'churn' in data.columns
    assert 'customer_id' in data.columns


def test_data_ingestion_churn_is_binary():
    """Test that target is binary."""
    from src.data_ingestion import generate_synthetic_data
    data = generate_synthetic_data(n_samples=1000)
    assert set(data['churn'].unique()).issubset({0, 1})


def test_data_ingestion_no_nulls():
    """Test that generated data has no missing values."""
    from src.data_ingestion import generate_synthetic_data
    data = generate_synthetic_data(n_samples=500)
    assert data.isnull().sum().sum() == 0


def test_config_loads():
    """Test that configuration loads without error."""
    from config.config import load_config
    config = load_config()
    assert 'project_name' in config
    assert 'hyperparameters' in config
    assert 'quality_gate' in config
