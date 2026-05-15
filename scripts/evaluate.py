"""
SageMaker Processing Script: Model Evaluation

This script runs as a Processing job to evaluate a trained model
against a test dataset and produce evaluation metrics.

Inputs:
    /opt/ml/processing/model/model.tar.gz  - Trained model artifact
    /opt/ml/processing/test/test.csv       - Test dataset

Outputs:
    /opt/ml/processing/evaluation/evaluation.json - Metrics report
"""

import json
import os
import tarfile
import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def extract_model(model_path):
    """Extract model from tar.gz archive."""
    print(f"Extracting model from: {model_path}")
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall(path='/opt/ml/processing/model_extracted')

    # Find the model file
    extracted_dir = '/opt/ml/processing/model_extracted'
    model_files = list(pathlib.Path(extracted_dir).rglob('xgboost-model'))
    if not model_files:
        # Try any file
        model_files = list(pathlib.Path(extracted_dir).iterdir())

    if not model_files:
        raise FileNotFoundError(f"No model file found in {extracted_dir}")

    return str(model_files[0])


def evaluate(model_path, test_path, output_path):
    """Run evaluation and save metrics."""
    # Load model
    model_file = extract_model(model_path)
    print(f"Loading model from: {model_file}")
    model = xgb.Booster()
    model.load_model(model_file)

    # Load test data (target is first column)
    print(f"Loading test data from: {test_path}")
    test_files = [f for f in os.listdir(test_path) if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(test_path, f), header=None) for f in test_files]
    test_data = pd.concat(dfs, ignore_index=True)

    y_true = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values

    # Predict
    dtest = xgb.DMatrix(X_test)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'binary_classification_metrics': {
            'auc': {
                'value': float(roc_auc_score(y_true, y_prob)),
                'standard_deviation': 'NaN'
            },
            'accuracy': {
                'value': float(accuracy_score(y_true, y_pred)),
                'standard_deviation': 'NaN'
            },
            'precision': {
                'value': float(precision_score(y_true, y_pred)),
                'standard_deviation': 'NaN'
            },
            'recall': {
                'value': float(recall_score(y_true, y_pred)),
                'standard_deviation': 'NaN'
            },
            'f1': {
                'value': float(f1_score(y_true, y_pred)),
                'standard_deviation': 'NaN'
            }
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'dataset_size': len(y_true),
        'positive_rate': float(y_true.mean())
    }

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric_name, metric_data in metrics['binary_classification_metrics'].items():
        print(f"  {metric_name}: {metric_data['value']:.4f}")
    print(f"\n  Dataset size: {metrics['dataset_size']}")
    print(f"  Positive rate: {metrics['positive_rate']:.3f}")

    # Save metrics
    os.makedirs(output_path, exist_ok=True)
    eval_file = os.path.join(output_path, 'evaluation.json')
    with open(eval_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {eval_file}")

    return metrics


if __name__ == '__main__':
    model_path = '/opt/ml/processing/model/model.tar.gz'
    test_path = '/opt/ml/processing/test'
    output_path = '/opt/ml/processing/evaluation'

    evaluate(model_path, test_path, output_path)
