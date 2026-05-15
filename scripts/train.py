"""
SageMaker Training Script: XGBoost (Script Mode)

This script demonstrates using XGBoost in "script mode" where you bring
your own training logic. For the built-in XGBoost algorithm, you don't
need this file — just use the Estimator with the XGBoost image directly.

Use this approach when you need custom training logic, custom metrics,
or want to use a framework estimator.

SageMaker Environment Variables:
    SM_MODEL_DIR: /opt/ml/model (where to save the trained model)
    SM_CHANNEL_TRAIN: /opt/ml/input/data/train
    SM_CHANNEL_VALIDATION: /opt/ml/input/data/validation
    SM_OUTPUT_DATA_DIR: /opt/ml/output/data
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def parse_args():
    """Parse hyperparameters passed by SageMaker."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--num-round', type=int, default=100)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval-metric', type=str, default='auc')
    parser.add_argument('--early-stopping-rounds', type=int, default=10)

    # SageMaker environment
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    return parser.parse_args()


def load_data(data_dir):
    """Load CSV data from a directory. Assumes target is first column."""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f), header=None)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    return X, y


def train(args):
    """Main training function."""
    print("Loading training data...")
    X_train, y_train = load_data(args.train)
    print(f"  Train shape: {X_train.shape}")

    print("Loading validation data...")
    X_val, y_val = load_data(args.validation)
    print(f"  Validation shape: {X_val.shape}")

    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'objective': args.objective,
        'eval_metric': args.eval_metric,
    }

    print(f"\nTraining with params: {json.dumps(params, indent=2)}")

    # Train
    watchlist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=10
    )

    # Evaluate
    val_predictions = model.predict(dval)
    auc = roc_auc_score(y_val, val_predictions)
    print(f"\nFinal Validation AUC: {auc:.4f}")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    print(f"\nTop feature importances (gain):")
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {score:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'xgboost-model')
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save evaluation metrics
    metrics = {
        'validation_auc': auc,
        'best_iteration': model.best_iteration,
        'num_features': X_train.shape[1]
    }
    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return model


if __name__ == '__main__':
    args = parse_args()
    train(args)
