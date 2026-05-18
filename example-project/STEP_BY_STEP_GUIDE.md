# Build This Project From ZERO (Like a 10-Year-Old)

> You are starting with an EMPTY folder. Nothing exists yet.
> We will create EVERY single file, EVERY folder, from scratch.
> Just follow along. Copy-paste everything exactly.

---

## What Are We Building?

A telecom company wants to know: **"Which customers will cancel their subscription?"**

We will:
1. Create a project from scratch
2. Write code that predicts which customers will leave
3. Run it on AWS (Amazon's cloud computers)
4. Make it production-ready (so it runs automatically every week)

---

---

# PHASE 1: INSTALL TOOLS (One-Time Setup)

---

## Step 1: Open VS Code

Open VS Code on your computer. If you don't have it, download from https://code.visualstudio.com

---

## Step 2: Open Terminal

Press `Ctrl + backtick` (the key above Tab, left of number 1).

A terminal panel opens at the bottom. This is where you type commands.

---

## Step 3: Check if Python is installed

Type this in terminal and press Enter:

```
python --version
```

✅ **Good**: You see `Python 3.9.x` or `Python 3.10.x` or `Python 3.11.x`

❌ **Bad**: You see an error → Go to https://python.org/downloads, install Python. **CHECK the box "Add Python to PATH"** during install. Then close and reopen VS Code.

---

## Step 4: Install uv (our package manager)

**What is uv?** It's a tool that installs Python libraries (like pandas, numpy, etc.) super fast.

**Why not pip?** uv is 10x faster and creates a clean environment automatically.

Paste this in terminal and press Enter:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Wait for it to finish. Then verify:

```
uv --version
```

You should see something like `uv 0.6.14`.

---

## Step 5: Install AWS CLI

**What is AWS CLI?** It lets you talk to Amazon's cloud from your terminal.

```
winget install Amazon.AWSCLI
```

Close terminal and reopen it (click the trash icon, then `Ctrl + backtick` again).

Verify:

```
aws --version
```

You should see `aws-cli/2.x.x`.

---

## Step 6: Configure AWS Credentials

**What are credentials?** They're like a username and password for AWS.

You need:
- An AWS account (https://aws.amazon.com)
- An Access Key (go to AWS Console → your name top-right → Security Credentials → Create Access Key)

Now type:

```
aws configure
```

It asks 4 questions. Answer them:

```
AWS Access Key ID: (paste your access key)
AWS Secret Access Key: (paste your secret key)
Default region name: eu-west-1
Default output format: json
```

Verify it works:

```
aws sts get-caller-identity
```

You should see your account number. If error → your keys are wrong, try again.

---

## Step 7: Create a SageMaker Role

**What is a role?** It's permission for SageMaker to access your data in S3.

1. Go to https://console.aws.amazon.com/iam/
2. Click "Roles" on the left
3. Click "Create role" (orange button)
4. Select "AWS service" → choose "SageMaker" → click Next
5. Search `AmazonSageMakerFullAccess` → check it ✓
6. Search `AmazonS3FullAccess` → check it ✓
7. Click Next
8. Role name: type `SageMakerExecutionRole`
9. Click "Create role"
10. Click on the role you just created
11. **Copy the ARN** at the top (looks like `arn:aws:iam::123456789012:role/SageMakerExecutionRole`)

**Save this ARN somewhere. You'll need it soon.**

---

# ✅ TOOLS INSTALLED! Now let's build the project.

---
---

# PHASE 2: CREATE THE PROJECT FROM SCRATCH

---

## Step 8: Create the project folder

In terminal, type:

```
mkdir "C:\Users\Allen.Harry\OneDrive - Entain Group\Desktop\churn-prediction"
cd "C:\Users\Allen.Harry\OneDrive - Entain Group\Desktop\churn-prediction"
```

Now open this folder in VS Code: File → Open Folder → navigate to `Desktop\churn-prediction` → Select Folder.

Open terminal again (`Ctrl + backtick`).

---

## Step 9: Create the folder structure

We need organized folders. Type these commands one by one:

```
mkdir notebooks
mkdir scripts
mkdir config
mkdir data
mkdir src
mkdir tests
mkdir pipeline
```

Your folder now looks like:

```
churn-prediction/
├── notebooks/     ← Jupyter notebooks (experiments)
├── scripts/       ← Code that runs ON SageMaker (not your laptop)
├── config/        ← Settings and configuration
├── data/          ← Local data files (temporary)
├── src/           ← Main source code (your modules)
├── tests/         ← Tests to make sure code works
└── pipeline/      ← Automated pipeline definition
```

---

## Step 10: Create pyproject.toml (The Shopping List)

**What is pyproject.toml?** It's a file that tells `uv` what libraries to install. Think of it as a shopping list for your project.

In VS Code: File → New File → save it as `pyproject.toml` in the root folder.

Paste this ENTIRE content:

```toml
[project]
name = "churn-prediction"
version = "0.1.0"
description = "Predict which customers will cancel their subscription"
requires-python = ">=3.9"
dependencies = [
    "boto3>=1.28.0",
    "sagemaker>=2.200.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.4.0",
]
```

Save it (Ctrl+S).

**What each library does:**
- `boto3` → talks to AWS
- `sagemaker` → talks to SageMaker specifically
- `pandas` → works with tables of data
- `numpy` → does math
- `scikit-learn` → machine learning tools
- `matplotlib` + `seaborn` → makes charts
- `jupyter` + `ipykernel` → runs notebooks
- `pytest` → runs tests
- `ruff` → checks code quality

---

## Step 11: Install all dependencies

In terminal (make sure you're in the `churn-prediction` folder):

```
uv sync --all-extras
```

**What just happened?**
- uv read your `pyproject.toml`
- Created a virtual environment (`.venv` folder)
- Downloaded and installed ALL the libraries
- Created a `uv.lock` file (exact versions, for reproducibility)

This takes 30-60 seconds. Wait for it to finish.

---

## Step 12: Register Jupyter kernel

**What is a kernel?** It's the Python environment that Jupyter notebooks use.

```
uv run python -m ipykernel install --user --name churn --display-name "Churn Project"
```

Now when you open a notebook, you can select "Churn Project" as the kernel.

---

## Step 13: Create requirements.txt (Alternative to pyproject.toml)

**Why both?** Some tools only understand `requirements.txt`. It's good to have both.

Create a new file called `requirements.txt` in the root folder:

```
boto3>=1.28.0
sagemaker>=2.200.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.25.0
pytest>=7.4.0
```

Save it.

**When would you use this instead of pyproject.toml?**
- If someone doesn't have `uv`, they can do: `pip install -r requirements.txt`
- Some CI/CD systems prefer requirements.txt

---

## Step 14: Create the config file

Create file: `config/params.json`

```json
{
    "project_name": "churn-prediction",
    "region": "eu-west-1",
    "role_arn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole",

    "processing_instance_type": "ml.m5.xlarge",
    "training_instance_type": "ml.m5.xlarge",
    "inference_instance_type": "ml.m5.large",

    "hyperparameters": {
        "objective": "binary:logistic",
        "num_round": "100",
        "max_depth": "5",
        "eta": "0.2",
        "subsample": "0.8",
        "colsample_bytree": "0.8",
        "eval_metric": "auc"
    },

    "quality_gate": {
        "min_auc": 0.75
    }
}
```

**⚠️ IMPORTANT**: Replace `YOUR_ACCOUNT_ID` with your actual AWS account number, and make sure the role name matches what you created in Step 7.

Save it.

---

## Step 15: Create .gitignore

**What is .gitignore?** It tells Git which files to NOT upload (like passwords, big data files, etc.)

Create file: `.gitignore` in the root folder:

```
__pycache__/
*.pyc
.venv/
.python-version
data/*.csv
.env
*.tar.gz
.ipynb_checkpoints/
```

Save it.

---

# ✅ PROJECT STRUCTURE COMPLETE!

Your folder now looks like:

```
churn-prediction/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock              ← (auto-created by uv sync)
├── .venv/               ← (auto-created by uv sync)
├── config/
│   └── params.json
├── notebooks/
├── scripts/
├── data/
├── src/
├── tests/
└── pipeline/
```

---
---

# PHASE 3: BUILD THE NOTEBOOK (Experiment First)

Before writing production code, we ALWAYS experiment in a notebook first.

---

## Step 16: Create the notebook file

In VS Code: right-click on `notebooks/` folder → New File → name it `experiment.ipynb`

VS Code will open it as a Jupyter notebook.

At the top-right, click "Select Kernel" → choose "Churn Project".

---

## Step 17: Add cells to the notebook

A notebook has "cells". Each cell is a block of code you can run independently.

Click "+ Code" to add a new code cell. Paste the code below into each cell, one at a time.

---

### Cell 1: Import libraries and connect to AWS

```python
# Cell 1: Setup
import boto3
import sagemaker
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load our config
with open('../config/params.json') as f:
    config = json.load(f)

# Connect to AWS
REGION = config['region']
ROLE_ARN = config['role_arn']

boto_session = boto3.Session(region_name=REGION)
session = sagemaker.Session(boto_session=boto_session)
bucket = session.default_bucket()
prefix = config['project_name']

print(f"Connected to AWS!")
print(f"Region: {REGION}")
print(f"S3 Bucket: {bucket}")
print(f"SageMaker SDK: {sagemaker.__version__}")
```

Press `Shift+Enter` to run it. You should see "Connected to AWS!" and your bucket name.

---

### Cell 2: Create fake customer data

```python
# Cell 2: Generate data (in real life, you'd query a database here)
np.random.seed(42)
n = 5000  # 5000 customers

data = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'tenure_months': np.random.randint(1, 72, n),          # How long they've been a customer
    'monthly_charges': np.random.uniform(20, 100, n).round(2),  # Monthly bill
    'total_charges': np.random.uniform(100, 7000, n).round(2),  # Total spent
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.2]),
    'online_security': np.random.choice(['Yes', 'No'], n),
    'tech_support': np.random.choice(['Yes', 'No'], n),
    'num_support_tickets': np.random.poisson(2, n),        # How many times they called support
})

# Create the target: will they churn? (based on realistic logic)
churn_prob = (
    0.3 * (data['contract_type'] == 'Month-to-month').astype(float) +
    0.2 * (data['tenure_months'] < 12).astype(float) +
    0.15 * (data['monthly_charges'] > 70).astype(float) +
    0.1 * (data['num_support_tickets'] > 3).astype(float) +
    np.random.uniform(0, 0.3, n)
)
data['churn'] = (churn_prob > 0.5).astype(int)  # 1 = will leave, 0 = will stay

print(f"Created {n} customer records")
print(f"Churn rate: {data['churn'].mean():.1%} of customers will leave")
print(f"\nFirst 5 rows:")
data.head()
```

Run it (`Shift+Enter`). You should see 5000 records with ~40% churn rate.

---

### Cell 3: Explore the data (EDA)

```python
# Cell 3: Visualize the data
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Chart 1: Churn by contract type
sns.countplot(data=data, x='contract_type', hue='churn', ax=axes[0,0])
axes[0,0].set_title('Month-to-month customers churn more')

# Chart 2: Tenure (how long they stay)
sns.histplot(data=data, x='tenure_months', hue='churn', kde=True, ax=axes[0,1])
axes[0,1].set_title('New customers churn more')

# Chart 3: Monthly charges
sns.boxplot(data=data, x='churn', y='monthly_charges', ax=axes[1,0])
axes[1,0].set_title('Higher bills = more churn')

# Chart 4: Support tickets
sns.countplot(data=data, x='num_support_tickets', hue='churn', ax=axes[1,1])
axes[1,1].set_title('More complaints = more churn')

plt.tight_layout()
plt.show()
```

Run it. You'll see 4 charts showing patterns in the data.

---

### Cell 4: Preprocess the data

```python
# Cell 4: Clean and prepare data for the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Drop customer_id (it's not a useful feature)
df = data.drop('customer_id', axis=1)

# Step 2: Convert text columns to numbers (models only understand numbers)
text_columns = ['contract_type', 'payment_method', 'internet_service', 'online_security', 'tech_support']

for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"Converted '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Step 3: Put target column FIRST (SageMaker XGBoost requires this)
target = 'churn'
features = [c for c in df.columns if c != target]
df_final = df[[target] + features]

# Step 4: Split into train (70%), validation (15%), test (15%)
train_df, temp_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final[target])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[target])

print(f"\nData split:")
print(f"  Train:      {train_df.shape[0]} rows (for learning)")
print(f"  Validation: {val_df.shape[0]} rows (for tuning)")
print(f"  Test:       {test_df.shape[0]} rows (for final grade)")
```

Run it. You'll see how text is converted to numbers and data is split.

---

### Cell 5: Save data and upload to S3

```python
# Cell 5: Save locally, then upload to AWS S3
data_dir = Path('../data')
data_dir.mkdir(exist_ok=True)

# Save as CSV (no header, no index — SageMaker format)
train_df.to_csv(data_dir / 'train.csv', index=False, header=False)
val_df.to_csv(data_dir / 'validation.csv', index=False, header=False)
test_df.to_csv(data_dir / 'test.csv', index=False, header=False)

# Upload to S3 (Amazon's cloud storage)
train_s3 = session.upload_data(str(data_dir / 'train.csv'), bucket=bucket, key_prefix=f'{prefix}/data/train')
val_s3 = session.upload_data(str(data_dir / 'validation.csv'), bucket=bucket, key_prefix=f'{prefix}/data/validation')
test_s3 = session.upload_data(str(data_dir / 'test.csv'), bucket=bucket, key_prefix=f'{prefix}/data/test')

print("Data uploaded to S3!")
print(f"  Train: {train_s3}")
print(f"  Val:   {val_s3}")
print(f"  Test:  {test_s3}")
```

Run it. Your data is now in the cloud.

---

### Cell 6: Train the model on SageMaker

```python
# Cell 6: Train the model (THIS RUNS ON AWS, NOT YOUR LAPTOP)
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# Get the XGBoost container (pre-built by Amazon)
xgb_image = sagemaker.image_uris.retrieve('xgboost', REGION, '1.7-1')

# Configure the training job
estimator = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',  # A powerful machine in the cloud
    output_path=f's3://{bucket}/{prefix}/models',
    sagemaker_session=session,
    base_job_name='churn-xgb'
)

# Set hyperparameters (settings for the algorithm)
estimator.set_hyperparameters(**config['hyperparameters'])

# START TRAINING (this takes 5-8 minutes)
print("Starting training job... (wait 5-8 minutes)")
print("AWS is spinning up a machine, downloading data, and training the model.")
print("")

estimator.fit({
    'train': TrainingInput(s3_data=train_s3, content_type='text/csv'),
    'validation': TrainingInput(s3_data=val_s3, content_type='text/csv')
}, wait=True, logs='All')

print(f"\n✅ Training complete!")
print(f"Model saved at: {estimator.model_data}")
```

Run it. **This takes 5-8 minutes.** AWS is:
1. Starting a computer for you
2. Downloading your data
3. Training the model
4. Saving the result
5. Shutting down the computer

---

### Cell 7: Test how good the model is

```python
# Cell 7: Evaluate the model
from sklearn.metrics import roc_auc_score, classification_report

# Use Batch Transform to get predictions
transformer = estimator.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/{prefix}/predictions'
)

# Prepare test data (remove the target column)
test_features = test_df.iloc[:, 1:]
test_features.to_csv(data_dir / 'test_features.csv', index=False, header=False)
test_features_s3 = session.upload_data(
    str(data_dir / 'test_features.csv'), bucket=bucket, key_prefix=f'{prefix}/data/test_features'
)

print("Getting predictions... (3-5 minutes)")
transformer.transform(data=test_features_s3, content_type='text/csv', split_type='Line')
transformer.wait()

# Download predictions
import io
s3_client = boto3.client('s3', region_name=REGION)
pred_key = f'{prefix}/predictions/test_features.csv.out'
response = s3_client.get_object(Bucket=bucket, Key=pred_key)
preds_raw = response['Body'].read().decode('utf-8')
y_prob = np.array([float(x) for x in preds_raw.strip().split('\n')])

y_true = test_df.iloc[:, 0].values
y_pred = (y_prob > 0.5).astype(int)

# Print results
auc = roc_auc_score(y_true, y_prob)
print("\n" + "=" * 50)
print("MODEL REPORT CARD")
print("=" * 50)
print(f"\nAUC Score: {auc:.4f}")
print(f"  (1.0 = perfect, 0.5 = random guessing, >0.75 = good)")
print(f"\nDetailed breakdown:")
print(classification_report(y_true, y_pred, target_names=['Will Stay', 'Will Churn']))

if auc >= 0.75:
    print("✅ MODEL IS GOOD ENOUGH FOR PRODUCTION!")
else:
    print("❌ Model needs improvement. Try better features or more data.")
```

Run it. You should see AUC > 0.75 (good model!).

---

### Cell 8: Deploy as a live API

```python
# Cell 8: Deploy (makes the model available 24/7 for predictions)
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.CSVDeserializer()
)

print(f"✅ Model is LIVE!")
print(f"Endpoint name: {predictor.endpoint_name}")
print(f"\n⚠️  This costs ~$0.14/hour. Delete when done testing!")
```

---

### Cell 9: Make predictions

```python
# Cell 9: Ask the model to predict
samples = test_features.iloc[:5].values

print("Predictions for 5 customers:")
print("-" * 50)
for i, row in enumerate(samples):
    result = predictor.predict(row.tolist())
    prob = float(result[0][0])
    actual = "CHURNED" if int(y_true[i]) == 1 else "STAYED"
    predicted = "WILL CHURN" if prob > 0.5 else "WILL STAY"
    print(f"  Customer {i+1}: {prob:.1%} chance of leaving → {predicted} (actually {actual})")
```

---

### Cell 10: DELETE THE ENDPOINT (Save money!)

```python
# Cell 10: CLEANUP - Run this when done!
predictor.delete_endpoint()
print("✅ Endpoint deleted. No more charges.")
```

**⚠️ ALWAYS run this cell when you're done testing!**

---

# ✅ NOTEBOOK COMPLETE!

You just built an ML model from scratch, trained it on the cloud, and made predictions!

---
---

# PHASE 4: CONVERT TO PRODUCTION CODE (MLOps)

Now we take the notebook code and organize it into proper modules.

**Why?** Because notebooks are messy. In production you need:
- Code that can be tested
- Code that can run automatically (no human clicking cells)
- Code that multiple people can work on
- Code that handles errors gracefully

---

## Step 18: Create the config loader

Create file: `src/__init__.py` (empty file, tells Python this is a package)

```python
```

(Yes, it's empty. Just create it and save.)

Now create file: `config/config.py`

```python
"""Loads settings from params.json."""
import json
import os
from pathlib import Path

import boto3
import sagemaker


def load_config():
    """Read params.json and return as dictionary."""
    config_path = Path(__file__).parent / 'params.json'
    with open(config_path) as f:
        config = json.load(f)
    # Allow override from environment variable
    config['region'] = os.environ.get('AWS_REGION', config['region'])
    return config


def get_session(config=None):
    """Create a connection to SageMaker."""
    if config is None:
        config = load_config()
    boto_session = boto3.Session(region_name=config['region'])
    return sagemaker.Session(boto_session=boto_session)
```

Save it.

---

## Step 19: Create the data ingestion module

Create file: `src/data_ingestion.py`

```python
"""
Module 1: Data Ingestion
Job: Get data from somewhere and put it in S3.

In real life, replace generate_data() with:
  - pd.read_sql("SELECT * FROM customers", database_connection)
  - pd.read_csv("s3://company-data-lake/customers.csv")
  - requests.get("https://api.company.com/customers")
"""
import numpy as np
import pandas as pd


def generate_data(n_samples=5000, seed=42):
    """Create synthetic customer data. Replace with real data source."""
    np.random.seed(seed)
    n = n_samples

    data = pd.DataFrame({
        'customer_id': range(1, n + 1),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(20, 100, n).round(2),
        'total_charges': np.random.uniform(100, 7000, n).round(2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.2]),
        'online_security': np.random.choice(['Yes', 'No'], n),
        'tech_support': np.random.choice(['Yes', 'No'], n),
        'num_support_tickets': np.random.poisson(2, n),
    })

    churn_prob = (
        0.3 * (data['contract_type'] == 'Month-to-month').astype(float) +
        0.2 * (data['tenure_months'] < 12).astype(float) +
        0.15 * (data['monthly_charges'] > 70).astype(float) +
        0.1 * (data['num_support_tickets'] > 3).astype(float) +
        np.random.uniform(0, 0.3, n)
    )
    data['churn'] = (churn_prob > 0.5).astype(int)

    return data


def upload_to_s3(data, session, bucket, prefix):
    """Upload raw data to S3."""
    local_path = 'data/raw_data.csv'
    data.to_csv(local_path, index=False)
    s3_uri = session.upload_data(local_path, bucket=bucket, key_prefix=f'{prefix}/data/raw')
    print(f"  Uploaded to: {s3_uri}")
    return s3_uri
```

Save it.

---

## Step 20: Create the preprocessing script (runs ON SageMaker)

Create file: `scripts/preprocess.py`

**This file does NOT run on your laptop.** SageMaker uploads it to a cloud machine and runs it there.

```python
"""
Runs on SageMaker Processing instance (NOT your laptop).
Reads from /opt/ml/processing/input/
Writes to /opt/ml/processing/output/
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--val-size', type=float, default=0.5)
    args = parser.parse_args()

    # Read data (SageMaker puts it here)
    input_dir = '/opt/ml/processing/input'
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(input_dir, files[0]))
    print(f"Loaded {df.shape[0]} rows")

    # Drop ID
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    df = df.drop(columns=id_cols)

    # Encode text columns to numbers
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Target first
    target = 'churn'
    features = [c for c in df.columns if c != target]
    df_final = df[[target] + features]

    # Split
    train_df, temp_df = train_test_split(df_final, test_size=args.test_size, random_state=42, stratify=df_final[target])
    val_df, test_df = train_test_split(temp_df, test_size=args.val_size, random_state=42, stratify=temp_df[target])

    # Save (SageMaker reads from these paths)
    for name, data in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        out_dir = f'/opt/ml/processing/output/{name}'
        os.makedirs(out_dir, exist_ok=True)
        data.to_csv(f'{out_dir}/{name}.csv', index=False, header=False)

    print(f"Done! Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
```

Save it.

---

## Step 21: Create the evaluation script (runs ON SageMaker)

Create file: `scripts/evaluate.py`

```python
"""
Runs on SageMaker Processing instance.
Loads the trained model, tests it, saves metrics.
"""
import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


if __name__ == '__main__':
    # Extract model
    model_dir = '/opt/ml/processing/model'
    with tarfile.open(f'{model_dir}/model.tar.gz', 'r:gz') as tar:
        tar.extractall(path=model_dir)

    model = xgb.Booster()
    model.load_model(f'{model_dir}/xgboost-model')

    # Load test data
    test_dir = '/opt/ml/processing/test'
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    test_data = pd.concat([pd.read_csv(f'{test_dir}/{f}', header=None) for f in test_files])

    y_true = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values

    # Predict
    dtest = xgb.DMatrix(X_test)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'binary_classification_metrics': {
            'auc': {'value': float(roc_auc_score(y_true, y_prob))},
            'accuracy': {'value': float(accuracy_score(y_true, y_pred))},
            'precision': {'value': float(precision_score(y_true, y_pred))},
            'recall': {'value': float(recall_score(y_true, y_pred))},
            'f1': {'value': float(f1_score(y_true, y_pred))},
        }
    }

    print(f"AUC: {metrics['binary_classification_metrics']['auc']['value']:.4f}")

    # Save
    output_dir = '/opt/ml/processing/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=2)
```

Save it.

---

## Step 22: Create the main pipeline runner

Create file: `run_pipeline.py` (in the root folder)

```python
"""
THE BIG RED BUTTON.
Run this one file and it does EVERYTHING:
  1. Gets data
  2. Uploads to S3
  3. Runs preprocessing on SageMaker
  4. Trains model on SageMaker
  5. Evaluates model
  6. Checks if model is good enough
  7. Registers model (if good)
"""
import json
import sys
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# Add our modules to path
sys.path.insert(0, str(Path(__file__).parent))
from config.config import get_session, load_config
from src.data_ingestion import generate_data, upload_to_s3


def main():
    # === SETUP ===
    print("=" * 60)
    print("  CHURN PREDICTION PIPELINE")
    print("=" * 60)

    config = load_config()
    session = get_session(config)
    bucket = session.default_bucket()
    prefix = config['project_name']
    role = config['role_arn']
    region = config['region']

    print(f"\n  Region: {region}")
    print(f"  Bucket: {bucket}")

    # === STEP 1: GET DATA ===
    print("\n\n--- STEP 1: Data Ingestion ---")
    data = generate_data(n_samples=5000)
    raw_s3_uri = upload_to_s3(data, session, bucket, prefix)

    # === STEP 2: PREPROCESS ===
    print("\n\n--- STEP 2: Preprocessing (on SageMaker) ---")
    processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role,
        instance_type=config['processing_instance_type'],
        instance_count=1,
        sagemaker_session=session,
        base_job_name='churn-preprocess'
    )

    processor.run(
        code='scripts/preprocess.py',
        inputs=[ProcessingInput(source=raw_s3_uri, destination='/opt/ml/processing/input')],
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/output/train'),
            ProcessingOutput(output_name='validation', source='/opt/ml/processing/output/validation'),
            ProcessingOutput(output_name='test', source='/opt/ml/processing/output/test'),
        ]
    )

    # Get output paths
    job_desc = processor.latest_job.describe()
    outputs = {o['OutputName']: o['S3Output']['S3Uri'] for o in job_desc['ProcessingOutputConfig']['Outputs']}
    print(f"  Train: {outputs['train']}")
    print(f"  Val:   {outputs['validation']}")
    print(f"  Test:  {outputs['test']}")

    # === STEP 3: TRAIN ===
    print("\n\n--- STEP 3: Training (on SageMaker) ---")
    xgb_image = sagemaker.image_uris.retrieve('xgboost', region, '1.7-1')

    estimator = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=1,
        instance_type=config['training_instance_type'],
        output_path=f's3://{bucket}/{prefix}/models',
        sagemaker_session=session,
        base_job_name='churn-train'
    )
    estimator.set_hyperparameters(**config['hyperparameters'])

    estimator.fit({
        'train': TrainingInput(s3_data=outputs['train'], content_type='text/csv'),
        'validation': TrainingInput(s3_data=outputs['validation'], content_type='text/csv')
    }, wait=True, logs='All')

    print(f"  Model: {estimator.model_data}")

    # === STEP 4: EVALUATE ===
    print("\n\n--- STEP 4: Evaluation (on SageMaker) ---")
    processor.run(
        code='scripts/evaluate.py',
        inputs=[
            ProcessingInput(source=estimator.model_data, destination='/opt/ml/processing/model'),
            ProcessingInput(source=outputs['test'], destination='/opt/ml/processing/test'),
        ],
        outputs=[ProcessingOutput(output_name='evaluation', source='/opt/ml/processing/evaluation')]
    )

    # === STEP 5: CHECK QUALITY ===
    print("\n\n--- STEP 5: Quality Gate ---")
    eval_desc = processor.latest_job.describe()
    eval_s3 = eval_desc['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']

    import boto3
    s3 = boto3.client('s3')
    eval_bucket = eval_s3.split('/')[2]
    eval_key = '/'.join(eval_s3.split('/')[3:]) + '/evaluation.json'
    response = s3.get_object(Bucket=eval_bucket, Key=eval_key)
    metrics = json.loads(response['Body'].read())

    auc = metrics['binary_classification_metrics']['auc']['value']
    min_auc = config['quality_gate']['min_auc']

    print(f"  AUC: {auc:.4f} (minimum required: {min_auc})")

    if auc >= min_auc:
        print(f"  ✅ PASSED! Model is good enough.")
    else:
        print(f"  ❌ FAILED. Model needs improvement.")
        print(f"  Pipeline stopped. No deployment.")
        return

    # === STEP 6: REGISTER ===
    print("\n\n--- STEP 6: Register Model ---")
    from sagemaker.model import Model
    model = Model(image_uri=xgb_image, model_data=estimator.model_data, role=role, sagemaker_session=session)

    try:
        sm = session.boto_session.client('sagemaker')
        sm.create_model_package_group(ModelPackageGroupName='churn-models')
    except Exception:
        pass  # Already exists

    model_package = model.register(
        model_package_group_name='churn-models',
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=['ml.m5.large'],
        transform_instances=['ml.m5.large'],
        approval_status='PendingManualApproval'
    )
    print(f"  Registered: {model_package.model_package_arn}")

    # === DONE ===
    print("\n\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Model: {estimator.model_data}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Status: PendingManualApproval")
    print(f"\n  Next: Approve in AWS Console → SageMaker → Model Registry")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

Save it.

---

## Step 23: Create a test file

Create file: `tests/test_data.py`

```python
"""Tests to make sure our code works before running on AWS."""
import sys
sys.path.insert(0, '..')

from src.data_ingestion import generate_data


def test_data_has_correct_shape():
    data = generate_data(n_samples=100)
    assert data.shape == (100, 11), "Should have 100 rows and 11 columns"


def test_churn_is_binary():
    data = generate_data(n_samples=100)
    assert set(data['churn'].unique()).issubset({0, 1}), "Churn should be 0 or 1"


def test_no_missing_values():
    data = generate_data(n_samples=100)
    assert data.isnull().sum().sum() == 0, "Should have no missing values"
```

Save it.

---

## Step 24: Run the tests

In terminal:

```
uv run pytest tests/ -v
```

You should see all tests PASS (green). If they fail, check your file paths.

---

## Step 25: Run the full pipeline

```
uv run python run_pipeline.py
```

This takes 20-30 minutes. Watch the logs. It will:
1. Generate data ✓
2. Upload to S3 ✓
3. Preprocess on SageMaker (5 min) ✓
4. Train on SageMaker (5-8 min) ✓
5. Evaluate on SageMaker (5 min) ✓
6. Check quality gate ✓
7. Register model ✓

---

# ✅ YOU BUILT A PRODUCTION ML SYSTEM FROM SCRATCH!

---
---

# PHASE 5: WHAT TO DO NEXT

| Step | What | Command |
|------|------|---------|
| Push to GitHub | Save your code online | `git init` → `git add .` → `git commit -m "Initial"` → `git push` |
| Schedule retraining | Run pipeline every week | Set up EventBridge cron rule |
| Add monitoring | Watch for model drift | Add Model Monitor (see `src/monitoring.py` in the other project) |
| Use real data | Replace `generate_data()` | Connect to your actual database |
| Deploy endpoint | Make predictions live | Add `--deploy` flag to run_pipeline.py |

---

## COST SUMMARY

| Action | Cost | Duration |
|--------|------|----------|
| Running this pipeline once | ~$1.50 | 25 min |
| Leaving an endpoint running | $0.14/hour | Until you delete it |
| S3 storage | $0.02/month | Always |

**Golden rule: Always delete endpoints when done testing.**

---

*You did it. From empty folder to production ML pipeline. 🎉*
