"""
SageMaker Processing Script: Data Preprocessing

This script runs inside a SageMaker Processing container.
It reads raw data, applies transformations, and outputs train/val/test splits.

Input:  /opt/ml/processing/input/raw_data.csv
Output: /opt/ml/processing/output/train/train.csv
        /opt/ml/processing/output/validation/validation.csv
        /opt/ml/processing/output/test/test.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess(input_path, output_path, test_size=0.3, val_size=0.5):
    """
    Main preprocessing function.

    Args:
        input_path: Path to raw input CSV
        output_path: Base path for output directories
        test_size: Fraction for test+validation split
        val_size: Fraction of test_size to use for validation
    """
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Input shape: {df.shape}")

    # Drop ID columns if present
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    if id_cols:
        print(f"Dropping ID columns: {id_cols}")
        df = df.drop(columns=id_cols)

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Remove target from numerical cols
    target_col = 'churn'
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded {col}: {le.classes_}")

    # Handle missing values
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled NaN in {col} with median: {median_val}")

    # Reorder columns: target first (required by XGBoost on SageMaker)
    feature_cols = [col for col in df.columns if col != target_col]
    df_final = df[[target_col] + feature_cols]

    # Split data
    train_df, temp_df = train_test_split(
        df_final, test_size=test_size, random_state=42, stratify=df_final[target_col]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size, random_state=42, stratify=temp_df[target_col]
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {train_df.shape}")
    print(f"  Validation: {val_df.shape}")
    print(f"  Test: {test_df.shape}")

    # Save outputs (no header, no index — SageMaker XGBoost format)
    train_output = os.path.join(output_path, 'train')
    val_output = os.path.join(output_path, 'validation')
    test_output = os.path.join(output_path, 'test')

    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    train_df.to_csv(os.path.join(train_output, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(val_output, 'validation.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(test_output, 'test.csv'), index=False, header=False)

    print(f"\nPreprocessing complete. Files saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--val-size', type=float, default=0.5)
    args = parser.parse_args()

    # SageMaker Processing paths
    input_path = '/opt/ml/processing/input/raw_data.csv'
    output_path = '/opt/ml/processing/output'

    preprocess(input_path, output_path, args.test_size, args.val_size)
