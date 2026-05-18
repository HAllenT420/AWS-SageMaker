"""
SageMaker Processing Script: Preprocessing

This script runs INSIDE a SageMaker Processing container.
It does NOT run on your laptop.

Input:  /opt/ml/processing/input/raw_data.csv
Output: /opt/ml/processing/output/train/train.csv
        /opt/ml/processing/output/validation/validation.csv
        /opt/ml/processing/output/test/test.csv
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--val-size', type=float, default=0.5)
    args = parser.parse_args()

    # Read input
    input_dir = '/opt/ml/processing/input'
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(input_dir, files[0]))
    print(f"Input shape: {df.shape}")

    # Drop ID columns
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"Dropped: {id_cols}")

    # Encode categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"Encoded {col}")

    # Target first (XGBoost requirement)
    target = 'churn'
    feature_cols = [c for c in df.columns if c != target]
    df_final = df[[target] + feature_cols]

    # Split
    train_df, temp_df = train_test_split(
        df_final, test_size=args.test_size, random_state=42, stratify=df_final[target]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=args.val_size, random_state=42, stratify=temp_df[target]
    )

    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # Save (no header, no index — SageMaker XGBoost format)
    output_base = '/opt/ml/processing/output'
    for name, data in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        out_dir = os.path.join(output_base, name)
        os.makedirs(out_dir, exist_ok=True)
        data.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False, header=False)

    print("Preprocessing complete.")


if __name__ == '__main__':
    main()
