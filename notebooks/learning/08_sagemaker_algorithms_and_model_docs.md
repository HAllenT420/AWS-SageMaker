# SageMaker Algorithms & Model Documentation

> Complete reference for every built-in algorithm, how to use them, when to pick which,
> and how to bring your own model.

---

## Table of Contents

1. [How SageMaker Training Works (Under the Hood)](#1-how-training-works)
2. [Built-in Algorithms (Complete List)](#2-built-in-algorithms)
3. [XGBoost (Our Project's Algorithm — Deep Dive)](#3-xgboost-deep-dive)
4. [Other Key Algorithms Explained](#4-other-algorithms)
5. [Framework Estimators (Bring Your Own Script)](#5-framework-estimators)
6. [Custom Containers (Full Control)](#6-custom-containers)
7. [How to Choose the Right Algorithm](#7-how-to-choose)
8. [Hyperparameter Reference](#8-hyperparameter-reference)
9. [Model Artifacts — What's Inside model.tar.gz](#9-model-artifacts)
10. [Where to Find Everything in AWS Console](#10-aws-console-locations)

---

## 1. How Training Works (Under the Hood)

### What Happens When You Call `estimator.fit()`

```
YOUR LAPTOP                          AWS CLOUD
──────────                          ─────────

1. You run:                         
   estimator.fit(data)              
        │                           
        ▼                           
2. SageMaker API receives           
   your request                     
        │                           
        ▼                           
3. ─────────────────────────────▶  4. AWS provisions an EC2 instance
                                       (e.g., ml.m5.xlarge)
                                           │
                                           ▼
                                   5. Pulls Docker container
                                      (e.g., XGBoost 1.7-1 image)
                                           │
                                           ▼
                                   6. Downloads training data
                                      from S3 → /opt/ml/input/data/
                                           │
                                           ▼
                                   7. Runs training algorithm
                                      (reads data, fits model)
                                           │
                                           ▼
                                   8. Saves model artifact
                                      → /opt/ml/model/
                                           │
                                           ▼
                                   9. Uploads model.tar.gz to S3
                                           │
                                           ▼
                                   10. Terminates instance
                                       (you stop paying)
        │                           
        ▼                           
11. estimator.model_data            
    now points to S3 location       
    of your trained model           
```

### The Container's File System

When training runs, the container sees:

```
/opt/ml/
├── input/
│   ├── config/
│   │   ├── hyperparameters.json    ← Your hyperparameters
│   │   └── resourceConfig.json     ← Instance info
│   └── data/
│       ├── train/                  ← Training data from S3
│       │   └── train.csv
│       └── validation/             ← Validation data from S3
│           └── validation.csv
├── model/                          ← WHERE TO SAVE YOUR MODEL
│   └── (your model files go here)
└── output/
    └── (logs, metrics go here)
```

### Three Ways to Train

| Approach | You Provide | SageMaker Provides | Best For |
|----------|------------|-------------------|----------|
| **Built-in Algorithm** | Data + hyperparameters | Everything else | Quick start, standard problems |
| **Script Mode** | Training script + data | Container + infrastructure | Custom logic, familiar frameworks |
| **Custom Container** | Docker image + script + data | Infrastructure only | Special dependencies, proprietary code |

---

## 2. Built-in Algorithms (Complete List)

### Supervised Learning — Classification & Regression

| Algorithm | Problem Type | Use Case | Data Type |
|-----------|-------------|----------|-----------|
| **XGBoost** | Classification, Regression | Tabular data, competitions | CSV, Parquet, LibSVM |
| **Linear Learner** | Classification, Regression | Simple linear problems | CSV, RecordIO |
| **KNN** | Classification, Regression | Small datasets, baselines | CSV, RecordIO |
| **Factorization Machines** | Classification, Regression | Sparse data, recommendations | RecordIO |
| **TabTransformer** | Classification, Regression | Tabular with categoricals | CSV |
| **AutoGluon-Tabular** | Classification, Regression | AutoML for tabular | CSV |
| **CatBoost** | Classification, Regression | Lots of categorical features | CSV |
| **LightGBM** | Classification, Regression | Large datasets, fast training | CSV |

### Supervised Learning — Time Series

| Algorithm | Problem Type | Use Case |
|-----------|-------------|----------|
| **DeepAR** | Forecasting | Multiple related time series |
| **Prophet** | Forecasting | Single time series with seasonality |

### Unsupervised Learning

| Algorithm | Problem Type | Use Case |
|-----------|-------------|----------|
| **K-Means** | Clustering | Customer segmentation |
| **PCA** | Dimensionality reduction | Feature reduction |
| **Random Cut Forest** | Anomaly detection | Fraud, system failures |
| **IP Insights** | Anomaly detection | Suspicious IP addresses |

### Computer Vision

| Algorithm | Problem Type | Use Case |
|-----------|-------------|----------|
| **Image Classification** | Label images | Cat vs dog, product categorization |
| **Object Detection** | Find objects in images | Self-driving cars, retail |
| **Semantic Segmentation** | Pixel-level labeling | Medical imaging |

### Natural Language Processing

| Algorithm | Problem Type | Use Case |
|-----------|-------------|----------|
| **BlazingText** | Text classification, Word2Vec | Sentiment, embeddings |
| **Sequence-to-Sequence** | Translation, summarization | Language tasks |
| **Object2Vec** | Embeddings | Similarity, recommendations |
| **Neural Topic Model** | Topic modeling | Document categorization |
| **LDA** | Topic modeling | Discover themes in text |

---

## 3. XGBoost Deep Dive (Our Project's Algorithm)

### What Is XGBoost?

XGBoost = e**X**treme **G**radient **Boost**ing

It builds many small decision trees, one after another. Each new tree tries to fix the mistakes of the previous trees.

```
Tree 1: Makes predictions (lots of errors)
    ↓
Tree 2: Focuses on fixing Tree 1's errors
    ↓
Tree 3: Focuses on fixing remaining errors
    ↓
... (100 trees later)
    ↓
Final prediction = Sum of all trees' predictions
```

### Why XGBoost?

- **Best for tabular data** (rows and columns, like spreadsheets)
- **Wins most Kaggle competitions** for structured data
- **Fast** — optimized C++ implementation
- **Handles missing values** automatically
- **Built-in regularization** — less overfitting
- **Feature importance** — tells you which columns matter

### XGBoost on SageMaker — Two Modes

#### Mode 1: Built-in Algorithm (Simplest)

```python
# SageMaker handles everything. You just provide data + hyperparameters.
from sagemaker.estimator import Estimator

xgb_image = sagemaker.image_uris.retrieve('xgboost', 'eu-west-1', '1.7-1')

estimator = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/models/'
)

estimator.set_hyperparameters(
    objective='binary:logistic',  # Binary classification
    num_round=100,                # Number of trees
    max_depth=5,                  # How deep each tree can be
    eta=0.2,                      # Learning rate
    eval_metric='auc'             # What to optimize
)

estimator.fit({
    'train': TrainingInput(s3_data=train_uri, content_type='text/csv'),
    'validation': TrainingInput(s3_data=val_uri, content_type='text/csv')
})
```

**Data format required**: CSV, first column = target, no header.

#### Mode 2: Script Mode (More Control)

```python
# You write your own train.py with custom logic
from sagemaker.xgboost import XGBoost

estimator = XGBoost(
    entry_point='scripts/train.py',    # YOUR script
    framework_version='1.7-1',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters={
        'max-depth': 5,
        'eta': 0.2,
        'num-round': 100
    }
)
```

### XGBoost Hyperparameters (Complete)

| Parameter | What It Controls | Range | Default | Effect of Increasing |
|-----------|-----------------|-------|---------|---------------------|
| `num_round` | Number of trees | 1-10000 | 100 | More trees = better fit, slower, risk overfit |
| `max_depth` | Tree depth | 1-20 | 6 | Deeper = more complex patterns, risk overfit |
| `eta` (learning_rate) | Step size | 0.001-1.0 | 0.3 | Lower = more robust, needs more trees |
| `subsample` | % of rows per tree | 0.1-1.0 | 1.0 | Lower = more regularization |
| `colsample_bytree` | % of columns per tree | 0.1-1.0 | 1.0 | Lower = more regularization |
| `min_child_weight` | Min samples in leaf | 0-100 | 1 | Higher = more conservative |
| `gamma` | Min loss reduction to split | 0-10 | 0 | Higher = fewer splits |
| `alpha` (reg_alpha) | L1 regularization | 0-1000 | 0 | Higher = sparser model |
| `lambda` (reg_lambda) | L2 regularization | 0-1000 | 1 | Higher = smaller weights |
| `scale_pos_weight` | Class imbalance ratio | 0-1000 | 1 | Set to neg/pos ratio for imbalanced data |

### XGBoost Objectives (What Are You Predicting?)

| Objective | Use Case | Output |
|-----------|----------|--------|
| `binary:logistic` | Binary classification (yes/no) | Probability 0-1 |
| `multi:softmax` | Multi-class (A/B/C/D) | Class label |
| `multi:softprob` | Multi-class | Probability per class |
| `reg:squarederror` | Regression (predict a number) | Continuous value |
| `reg:logistic` | Regression bounded 0-1 | Value 0-1 |
| `rank:pairwise` | Ranking (search results) | Relevance score |
| `count:poisson` | Count data | Non-negative integer |
| `survival:cox` | Survival analysis | Hazard ratio |

### XGBoost Evaluation Metrics

| Metric | Use With | What It Measures |
|--------|----------|-----------------|
| `auc` | Binary classification | Ranking quality (0.5=random, 1.0=perfect) |
| `error` | Binary classification | Misclassification rate |
| `logloss` | Classification | Probability calibration |
| `rmse` | Regression | Root mean squared error |
| `mae` | Regression | Mean absolute error |
| `merror` | Multi-class | Multi-class error rate |
| `mlogloss` | Multi-class | Multi-class log loss |

### Recommended Starting Hyperparameters

```python
# Good defaults for most tabular problems:
hyperparameters = {
    'objective': 'binary:logistic',
    'num_round': 200,
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'eval_metric': 'auc',
    'early_stopping_rounds': 20  # Stop if no improvement for 20 rounds
}
```

---

## 4. Other Key Algorithms Explained

### Linear Learner

**What**: Linear/logistic regression on steroids (distributed, regularized).

**When**: Baseline model, interpretable predictions, very large datasets.

```python
from sagemaker.estimator import Estimator

ll_image = sagemaker.image_uris.retrieve('linear-learner', region)
estimator = Estimator(image_uri=ll_image, ...)
estimator.set_hyperparameters(
    predictor_type='binary_classifier',  # or 'regressor'
    mini_batch_size=200,
    epochs=15,
    l1=0.001  # L1 regularization
)
```

### Random Cut Forest (Anomaly Detection)

**What**: Finds outliers/anomalies in data without labels.

**When**: Fraud detection, system monitoring, unusual behavior.

```python
rcf_image = sagemaker.image_uris.retrieve('randomcutforest', region)
estimator = Estimator(image_uri=rcf_image, ...)
estimator.set_hyperparameters(
    num_trees=100,
    num_samples_per_tree=256
)
# Output: anomaly score (higher = more anomalous)
```

### DeepAR (Time Series Forecasting)

**What**: Forecasts multiple related time series using deep learning.

**When**: Demand forecasting, energy consumption, stock predictions.

```python
deepar_image = sagemaker.image_uris.retrieve('forecasting-deepar', region)
estimator = Estimator(image_uri=deepar_image, ...)
estimator.set_hyperparameters(
    time_freq='D',           # Daily data
    prediction_length=30,    # Predict 30 days ahead
    context_length=90,       # Use 90 days of history
    epochs=100
)
# Input format: JSON lines with "start", "target" fields
```

### K-Means (Clustering)

**What**: Groups similar data points together.

**When**: Customer segmentation, document grouping.

```python
kmeans_image = sagemaker.image_uris.retrieve('kmeans', region)
estimator = Estimator(image_uri=kmeans_image, ...)
estimator.set_hyperparameters(
    k=5,                    # Number of clusters
    feature_dim=10,         # Number of features
    mini_batch_size=500
)
```

---

## 5. Framework Estimators (Bring Your Own Script)

### When to Use Script Mode

- You want to use PyTorch, TensorFlow, or scikit-learn
- You need custom training logic (custom loss, custom metrics)
- You want to use libraries not in the built-in containers

### PyTorch on SageMaker

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='scripts/',
    framework_version='2.1.0',
    py_version='py310',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    hyperparameters={
        'epochs': 50,
        'batch-size': 64,
        'learning-rate': 0.001
    }
)
```

### TensorFlow on SageMaker

```python
from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(
    entry_point='train.py',
    source_dir='scripts/',
    framework_version='2.14.0',
    py_version='py310',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    hyperparameters={
        'epochs': 100,
        'batch-size': 32
    }
)
```

### Scikit-Learn on SageMaker

```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point='train.py',
    framework_version='1.2-1',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters={
        'n-estimators': 100,
        'max-depth': 10
    }
)
```

### HuggingFace on SageMaker

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point='train.py',
    transformers_version='4.28.1',
    pytorch_version='2.0.0',
    py_version='py310',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    hyperparameters={
        'model_name': 'bert-base-uncased',
        'epochs': 3,
        'batch_size': 16
    }
)
```

---

## 6. Custom Containers (Full Control)

### When to Use

- Need libraries not available in pre-built containers
- Proprietary algorithms
- Complex multi-step training
- Specific OS/system requirements

### How It Works

```
1. Write a Dockerfile
2. Build the image
3. Push to Amazon ECR (container registry)
4. Reference in Estimator

Dockerfile → docker build → docker push → ECR → SageMaker uses it
```

### Example Dockerfile

```dockerfile
FROM python:3.11-slim

RUN pip install pandas numpy scikit-learn xgboost

# SageMaker expects your training script here:
COPY train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py

ENTRYPOINT ["python", "/opt/ml/code/train.py"]
```

### Using Custom Container

```python
estimator = Estimator(
    image_uri='123456789.dkr.ecr.eu-west-1.amazonaws.com/my-training:latest',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/models/'
)
```

---

## 7. How to Choose the Right Algorithm

```
START HERE: What type of data do you have?
│
├── Tabular (rows & columns, like a spreadsheet)?
│   │
│   ├── Target is Yes/No (binary)?
│   │   ├── Need interpretability? → Linear Learner
│   │   ├── Need best accuracy? → XGBoost or LightGBM
│   │   └── Lots of categorical features? → CatBoost
│   │
│   ├── Target is a number (regression)?
│   │   ├── Linear relationship? → Linear Learner
│   │   └── Non-linear? → XGBoost
│   │
│   ├── Target is A/B/C/D (multi-class)?
│   │   └── XGBoost (set objective='multi:softmax')
│   │
│   ├── No target (unsupervised)?
│   │   ├── Want to group similar items? → K-Means
│   │   ├── Want to find outliers? → Random Cut Forest
│   │   └── Want to reduce dimensions? → PCA
│   │
│   └── Don't know what to use? → AutoGluon-Tabular (tries everything)
│
├── Time series (values over time)?
│   ├── Multiple related series? → DeepAR
│   └── Single series with seasonality? → Prophet
│
├── Text?
│   ├── Classification (spam/not spam)? → BlazingText or HuggingFace BERT
│   ├── Generation/summarization? → Amazon Bedrock (LLMs)
│   └── Embeddings? → Object2Vec or BlazingText Word2Vec
│
├── Images?
│   ├── Classify whole image? → Image Classification (ResNet)
│   ├── Find objects in image? → Object Detection (SSD/YOLO)
│   └── Label every pixel? → Semantic Segmentation
│
└── Something else?
    └── Bring your own model (Script Mode or Custom Container)
```

---

## 8. Hyperparameter Reference

### Our Project's Hyperparameters Explained

```json
{
    "objective": "binary:logistic",    // We're predicting yes/no (churn/no churn)
    "num_round": "100",                // Build 100 trees
    "max_depth": "5",                  // Each tree can be 5 levels deep
    "eta": "0.2",                      // Learning rate (how much each tree contributes)
    "subsample": "0.8",               // Use 80% of data for each tree (randomness helps)
    "colsample_bytree": "0.8",        // Use 80% of features for each tree
    "eval_metric": "auc"              // Optimize for AUC (ranking quality)
}
```

### What Happens If You Change Them

| Change | Effect | Risk |
|--------|--------|------|
| Increase `num_round` (100→500) | Model learns more | Slower training, possible overfit |
| Increase `max_depth` (5→10) | Captures complex patterns | Overfitting on small data |
| Decrease `eta` (0.2→0.05) | More robust model | Needs more trees (num_round) |
| Decrease `subsample` (0.8→0.5) | More regularization | May underfit |
| Increase `min_child_weight` (1→10) | More conservative splits | May miss patterns |

### Tuning Strategy

```
Step 1: Start with defaults (our params.json)
Step 2: If overfitting (train AUC >> test AUC):
        → Decrease max_depth (5→3)
        → Decrease eta (0.2→0.1)
        → Increase min_child_weight (1→5)
Step 3: If underfitting (both train and test AUC are low):
        → Increase max_depth (5→8)
        → Increase num_round (100→300)
        → Add more features
Step 4: Use SageMaker Hyperparameter Tuning for automatic search
```

---

## 9. Model Artifacts — What's Inside model.tar.gz

After training, SageMaker saves a `model.tar.gz` file to S3.

### Contents

```
model.tar.gz
└── xgboost-model          ← The trained model (binary file)
```

For script mode, it might also contain:
```
model.tar.gz
├── model.pkl              ← Serialized model
├── preprocessor.pkl       ← Fitted preprocessor (scaler, encoder)
└── metadata.json          ← Feature names, training info
```

### How to Download and Inspect

```python
import boto3
import tarfile

# Download
s3 = boto3.client('s3')
s3.download_file(bucket, 'churn-prediction/models/.../model.tar.gz', 'model.tar.gz')

# Extract
with tarfile.open('model.tar.gz', 'r:gz') as tar:
    tar.extractall('model_dir/')

# Load and use locally
import xgboost as xgb
model = xgb.Booster()
model.load_model('model_dir/xgboost-model')

# Check feature importance
importance = model.get_score(importance_type='gain')
for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {score:.2f}")
```

---

## 10. Where to Find Everything in AWS Console

### Training Jobs

```
AWS Console → SageMaker → Training → Training Jobs
```

Here you see:
- Job name, status (Completed/Failed/InProgress)
- Duration, instance type
- Input/output S3 paths
- Hyperparameters used
- Metrics (click job → Metrics tab)
- Logs (click job → View Logs → CloudWatch)

### Model Artifacts

```
AWS Console → S3 → your-bucket → churn-prediction → models
```

Each training job creates a folder with `model.tar.gz` inside.

### Model Registry

```
AWS Console → SageMaker → Model Registry → churn-models
```

Shows all registered model versions with:
- Version number
- Approval status (Pending/Approved/Rejected)
- Metrics
- Creation date
- Model artifact location

### Endpoints

```
AWS Console → SageMaker → Inference → Endpoints
```

Shows live endpoints with:
- Status (InService/Creating/Deleting)
- Instance type and count
- Creation time
- **DELETE HERE when done testing!**

### Pipeline Executions

```
AWS Console → SageMaker → Pipelines → churn-prediction-pipeline
```

Visual DAG showing:
- Each step and its status
- Execution history
- Parameters used
- Logs for each step

---

*End of SageMaker Algorithms & Model Documentation*
