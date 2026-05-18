"""
SageMaker Processing Script: Evaluation

Runs INSIDE a SageMaker Processing container.

Input:  /opt/ml/processing/model/model.tar.gz
        /opt/ml/processing/test/test.csv
Output: /opt/ml/processing/evaluation/evaluation.json
"""

import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)


def main():
    # Extract model
    model_dir = '/opt/ml/processing/model'
    tar_path = os.path.join(model_dir, 'model.tar.gz')
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=model_dir)

    # Find model file
    model_file = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_file)
    print("Model loaded.")

    # Load test data
    test_dir = '/opt/ml/processing/test'
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    test_data = pd.concat([
        pd.read_csv(os.path.join(test_dir, f), header=None) for f in test_files
    ])

    y_true = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values

    # Predict
    dtest = xgb.DMatrix(X_test)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'binary_classification_metrics': {
            'auc': {'value': float(roc_auc_score(y_true, y_prob)), 'standard_deviation': 'NaN'},
            'accuracy': {'value': float(accuracy_score(y_true, y_pred)), 'standard_deviation': 'NaN'},
            'precision': {'value': float(precision_score(y_true, y_pred)), 'standard_deviation': 'NaN'},
            'recall': {'value': float(recall_score(y_true, y_pred)), 'standard_deviation': 'NaN'},
            'f1': {'value': float(f1_score(y_true, y_pred)), 'standard_deviation': 'NaN'},
        },
        'dataset_size': len(y_true),
        'positive_rate': float(y_true.mean())
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for name, data in metrics['binary_classification_metrics'].items():
        print(f"  {name}: {data['value']:.4f}")

    # Save
    output_dir = '/opt/ml/processing/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {output_dir}/evaluation.json")


if __name__ == '__main__':
    main()
