# AWS SageMaker: Complete Production Project Guide

> Your step-by-step reference for building production-grade ML projects on AWS SageMaker.
> No videos or courses needed — everything is here in sequential order.

---

## Table of Contents

1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Project Architecture Overview](#2-project-architecture-overview)
3. [Step 1: Data Storage & Management (S3)](#3-step-1-data-storage--management)
4. [Step 2: Data Exploration (SageMaker Studio / Notebooks)](#4-step-2-data-exploration)
5. [Step 3: Data Processing (Processing Jobs)](#5-step-3-data-processing)
6. [Step 4: Feature Engineering (Feature Store)](#6-step-4-feature-engineering)
7. [Step 5: Model Training (Training Jobs)](#7-step-5-model-training)
8. [Step 6: Hyperparameter Tuning](#8-step-6-hyperparameter-tuning)
9. [Step 7: Model Evaluation & Validation](#9-step-7-model-evaluation)
10. [Step 8: Model Registry](#10-step-8-model-registry)
11. [Step 9: Model Deployment (Endpoints)](#11-step-9-model-deployment)
12. [Step 10: Monitoring & Observability](#12-step-10-monitoring)
13. [Step 11: SageMaker Pipelines (Full Automation)](#13-step-11-pipelines)
14. [Step 12: CI/CD Integration](#14-step-12-cicd-integration)
15. [Cost Optimization](#15-cost-optimization)
16. [Production Checklist](#16-production-checklist)

---

## 1. Prerequisites & Setup

### What You Need Before Starting

| Requirement | Why | How to Get It |
|-------------|-----|---------------|
| AWS Account | Access to SageMaker services | aws.amazon.com |
| AWS CLI | Interact with AWS from terminal | `winget install Amazon.AWSCLI` |
| Python 3.9+ | SageMaker SDK language | python.org |
| uv | Fast package management | `irm https://astral.sh/uv/install.ps1 \| iex` |
| VS Code | IDE with Jupyter support | code.visualstudio.com |
| IAM Role | Permissions for SageMaker | Created in IAM Console |

### IAM Role Setup

Your SageMaker execution role needs these policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or scoped to your bucket)
- `CloudWatchLogsFullAccess`
- `AmazonECR` (if using custom containers)

```python
# How to create the role via CLI
import boto3

iam = boto3.client('iam')

trust_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}

# Create role
iam.create_role(
    RoleName='SageMakerExecutionRole',
    AssumeRolePolicyDocument=json.dumps(trust_policy)
)

# Attach policies
iam.attach_role_policy(
    RoleName='SageMakerExecutionRole',
    PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
)
```

### AWS Credentials Configuration

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g., eu-west-1), Output format (json)
```

### VS Code Setup

Install these extensions:
1. **Python** (ms-python.python)
2. **Jupyter** (ms-toolsai.jupyter)
3. **AWS Toolkit** (amazonwebservices.aws-toolkit-vscode)

---

## 2. Project Architecture Overview

### The Production ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Data     │───▶│  Processing  │───▶│  Feature     │               │
│  │  Sources  │    │  Job         │    │  Store       │               │
│  └──────────┘    └──────────────┘    └──────────────┘               │
│                                              │                        │
│                                              ▼                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Model   │◀───│  Training    │◀───│  Train/Val   │               │
│  │  Registry│    │  Job         │    │  Split       │               │
│  └──────────┘    └──────────────┘    └──────────────┘               │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Endpoint │◀───│  Deployment  │◀───│  Approval    │               │
│  │  (Live)  │    │  Config      │    │  Gate        │               │
│  └──────────┘    └──────────────┘    └──────────────┘               │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────────────────────────────────────┐                    │
│  │  Model Monitor → Alerts → Retrain Trigger    │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principle: Separation of Concerns

| Layer | What | SageMaker Component |
|-------|------|-------------------|
| Data | Storage, versioning, access | S3 + Glue Catalog |
| Processing | ETL, feature engineering | Processing Jobs |
| Training | Model fitting | Training Jobs |
| Evaluation | Quality checks | Processing Jobs |
| Registry | Model versioning | Model Registry |
| Serving | Real-time/batch inference | Endpoints / Batch Transform |
| Monitoring | Drift detection, alerts | Model Monitor |
| Orchestration | Pipeline automation | SageMaker Pipelines |

---

## 3. Step 1: Data Storage & Management

### S3 Bucket Structure (Best Practice)

```
s3://your-ml-bucket/
├── project-name/
│   ├── data/
│   │   ├── raw/                    # Original, untouched data
│   │   ├── processed/             # Cleaned, transformed data
│   │   │   ├── train/
│   │   │   ├── validation/
│   │   │   └── test/
│   │   └── features/             # Feature store snapshots
│   ├── models/                    # Model artifacts (model.tar.gz)
│   ├── evaluation/               # Metrics JSON files
│   ├── code/                     # Script archives
│   └── pipelines/                # Pipeline execution outputs
```

### Upload Data to S3

```python
import sagemaker

session = sagemaker.Session()
bucket = session.default_bucket()  # Creates sagemaker-<region>-<account-id>

# Upload a file
train_uri = session.upload_data(
    path='data/train.csv',
    bucket=bucket,
    key_prefix='my-project/data/raw'
)
# Returns: s3://sagemaker-eu-west-1-123456789/my-project/data/raw/train.csv

# Upload entire directory
session.upload_data(
    path='data/',
    bucket=bucket,
    key_prefix='my-project/data/raw'
)
```

### Data Versioning Strategies

| Strategy | How | When to Use |
|----------|-----|-------------|
| S3 Versioning | Enable on bucket | Small datasets, simple projects |
| Date-partitioned paths | `s3://bucket/data/2024-01-15/` | Time-series, regular updates |
| DVC (Data Version Control) | Git-like tracking for data | Large datasets, team collaboration |
| Glue Data Catalog | Metadata catalog | Enterprise, multiple consumers |

---

## 4. Step 2: Data Exploration

### Option A: SageMaker Studio (Cloud)

SageMaker Studio is a full IDE in the cloud. Use it when:
- You need GPU for exploration
- Data is too large to download locally
- Team collaboration on notebooks

### Option B: VS Code + Local (What We're Doing)

Use VS Code when:
- You prefer your local IDE
- Data fits in memory (or you sample)
- You want version control with Git
- You want to build proper project structure

### Connecting VS Code to SageMaker

```python
# You don't need SageMaker Studio. From VS Code, you interact with
# SageMaker services via the SDK. Your code runs locally, but
# training/processing jobs run on AWS infrastructure.

import boto3
import sagemaker

boto_session = boto3.Session(region_name='eu-west-1')
session = sagemaker.Session(boto_session=boto_session)

# This is your "connection" to SageMaker
# All subsequent API calls go through this session
```

### Exploration Checklist

Before building any model, answer these:

- [ ] What is the target variable?
- [ ] What type of problem? (Classification / Regression / Clustering / etc.)
- [ ] How much data do you have?
- [ ] What's the class balance? (for classification)
- [ ] Are there missing values? How many?
- [ ] Are there outliers?
- [ ] What features are categorical vs numerical?
- [ ] Is there data leakage? (future info in features)
- [ ] What's the baseline performance? (majority class, mean prediction)

---

## 5. Step 3: Data Processing (Processing Jobs)

### Why Processing Jobs?

Instead of running preprocessing on your laptop:
- **Scalable**: Use powerful instances (up to 96 vCPUs, 768 GB RAM)
- **Reproducible**: Same environment every time
- **Trackable**: SageMaker logs inputs, outputs, and parameters
- **Cost-effective**: Pay only while the job runs

### How Processing Jobs Work

```
Your Script (preprocess.py)
    ↓ uploaded to
SageMaker Processing Container
    ↓ reads from
/opt/ml/processing/input/  ← data pulled from S3
    ↓ writes to
/opt/ml/processing/output/ → pushed back to S3
```

### Running a Processing Job

```python
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

processor = SKLearnProcessor(
    framework_version='1.2-1',
    role=ROLE_ARN,
    instance_type='ml.m5.xlarge',    # 4 vCPUs, 16 GB RAM
    instance_count=1,
    sagemaker_session=session
)

processor.run(
    code='scripts/preprocess.py',     # Your local script
    inputs=[
        ProcessingInput(
            source='s3://bucket/project/data/raw/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='train',
            source='/opt/ml/processing/output/train',
            destination='s3://bucket/project/data/processed/train/'
        ),
        ProcessingOutput(
            output_name='validation',
            source='/opt/ml/processing/output/validation'
        ),
        ProcessingOutput(
            output_name='test',
            source='/opt/ml/processing/output/test'
        )
    ],
    arguments=['--test-size', '0.2']
)
```

### Processing Instance Types

| Instance | vCPUs | RAM | Use Case |
|----------|-------|-----|----------|
| ml.t3.medium | 2 | 4 GB | Small data, testing |
| ml.m5.large | 2 | 8 GB | Medium datasets |
| ml.m5.xlarge | 4 | 16 GB | Standard processing |
| ml.m5.4xlarge | 16 | 64 GB | Large datasets |
| ml.r5.4xlarge | 16 | 128 GB | Memory-intensive |

---

## 6. Step 4: Feature Engineering (Feature Store)

### What is SageMaker Feature Store?

A centralized repository for ML features that ensures:
- **Consistency**: Same features for training and inference
- **Reusability**: Share features across teams/models
- **Point-in-time correctness**: No data leakage in time-series

### Feature Store Concepts

```
Feature Group = A table of related features (like a database table)
    ├── Record = One row (one entity at one point in time)
    ├── Feature = One column (one attribute)
    ├── Record Identifier = Primary key (e.g., customer_id)
    └── Event Time = When this record was valid
```

### Creating a Feature Group

```python
from sagemaker.feature_store.feature_group import FeatureGroup

feature_group = FeatureGroup(
    name='customer-features',
    sagemaker_session=session
)

# Define schema
feature_group.load_feature_definitions(data_frame=features_df)

# Create the group (Online = real-time, Offline = batch/training)
feature_group.create(
    s3_uri=f's3://{bucket}/feature-store/',
    record_identifier_name='customer_id',
    event_time_feature_name='event_time',
    role_arn=ROLE_ARN,
    enable_online_store=True   # For real-time inference
)

# Ingest features
feature_group.ingest(data_frame=features_df, max_workers=4, wait=True)
```

### When to Use Feature Store vs Simple S3

| Scenario | Use Feature Store | Use S3 |
|----------|:-:|:-:|
| Single model, single team | | ✓ |
| Multiple models share features | ✓ | |
| Real-time inference needs features | ✓ | |
| Time-series with point-in-time joins | ✓ | |
| Simple batch predictions | | ✓ |
| Regulatory audit trail needed | ✓ | |

---

## 7. Step 5: Model Training

### Training Job Lifecycle

```
1. You call estimator.fit()
2. SageMaker provisions instance(s)
3. Pulls Docker container (XGBoost, PyTorch, etc.)
4. Downloads training data from S3
5. Runs your training script
6. Uploads model artifact (model.tar.gz) to S3
7. Terminates instance (you stop paying)
```

### Built-in Algorithms vs Custom Scripts

| Approach | When to Use | Example |
|----------|-------------|---------|
| Built-in Algorithm | Standard problems, quick start | XGBoost, Linear Learner |
| Script Mode | Custom logic, familiar frameworks | Your own train.py |
| Custom Container | Special dependencies, proprietary code | Your Dockerfile |

### Built-in Algorithm (Simplest)

```python
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# Get the pre-built XGBoost container
xgb_image = sagemaker.image_uris.retrieve('xgboost', region, '1.7-1')

estimator = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/models/',
    sagemaker_session=session
)

estimator.set_hyperparameters(
    objective='binary:logistic',
    num_round=100,
    max_depth=5,
    eta=0.2,
    eval_metric='auc'
)

estimator.fit({
    'train': TrainingInput(s3_data=train_uri, content_type='text/csv'),
    'validation': TrainingInput(s3_data=val_uri, content_type='text/csv')
})
```

### Script Mode (Recommended for Production)

```python
from sagemaker.xgboost import XGBoost

# Your own training script with custom logic
estimator = XGBoost(
    entry_point='scripts/train.py',        # Your script
    source_dir='scripts/',                  # Directory with dependencies
    framework_version='1.7-1',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/models/',
    hyperparameters={
        'max-depth': 5,
        'eta': 0.2,
        'num-round': 100
    }
)

estimator.fit({
    'train': TrainingInput(s3_data=train_uri, content_type='text/csv'),
    'validation': TrainingInput(s3_data=val_uri, content_type='text/csv')
})
```

### Custom Container (Full Control)

```python
# When you need libraries not in the pre-built containers
estimator = Estimator(
    image_uri='YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/my-training:latest',
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    output_path=f's3://{bucket}/models/'
)
```

### Training Instance Selection Guide

| Instance | GPUs | Use Case | Cost/hr (approx) |
|----------|------|----------|-------------------|
| ml.m5.large | 0 | Small tabular data | $0.13 |
| ml.m5.xlarge | 0 | Medium tabular data | $0.27 |
| ml.m5.4xlarge | 0 | Large tabular data | $1.08 |
| ml.p3.2xlarge | 1 V100 | Deep learning | $4.28 |
| ml.p3.8xlarge | 4 V100 | Large deep learning | $17.14 |
| ml.g5.xlarge | 1 A10G | Cost-effective DL | $1.41 |

### Distributed Training

```python
# For very large datasets or models
estimator = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,
    instance_count=4,              # Multiple instances
    instance_type='ml.m5.4xlarge',
    output_path=f's3://{bucket}/models/'
)
# SageMaker handles data distribution automatically for built-in algorithms
```

---

## 8. Step 6: Hyperparameter Tuning

### What It Does

Automatically searches for the best hyperparameters by running multiple training jobs in parallel.

### How It Works

```
You define:
  - Which hyperparameters to tune
  - The range for each
  - The metric to optimize
  - Max number of jobs

SageMaker:
  - Uses Bayesian optimization (smart search)
  - Runs jobs in parallel
  - Tracks all results
  - Returns the best combination
```

### Running a Tuning Job

```python
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter
)

# Define search ranges
hyperparameter_ranges = {
    'eta': ContinuousParameter(0.01, 0.3),
    'max_depth': IntegerParameter(3, 10),
    'num_round': IntegerParameter(50, 500),
    'subsample': ContinuousParameter(0.5, 1.0),
    'colsample_bytree': ContinuousParameter(0.5, 1.0),
    'min_child_weight': IntegerParameter(1, 10)
}

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='validation:auc',
    objective_type='Maximize',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=30,           # Total experiments
    max_parallel_jobs=5,   # Run 5 at a time
    strategy='Bayesian'    # Smart search (vs 'Random')
)

tuner.fit({
    'train': TrainingInput(s3_data=train_uri, content_type='text/csv'),
    'validation': TrainingInput(s3_data=val_uri, content_type='text/csv')
})

# Get best training job
best_job = tuner.best_training_job()
print(f"Best AUC: {tuner.best_score()}")
```

### Tuning Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| Bayesian | Learns from previous results | Default, most efficient |
| Random | Random combinations | Large search spaces, baseline |
| Grid | All combinations | Small spaces, exhaustive |
| Hyperband | Early stopping of bad runs | Deep learning, expensive training |

### Cost Control Tips

- Start with Random (10 jobs) to understand the landscape
- Then use Bayesian with narrower ranges
- Use Spot instances (up to 90% savings)
- Set `max_parallel_jobs` based on budget

---

## 9. Step 7: Model Evaluation & Validation

### Evaluation Processing Job

```python
# Run evaluation as a separate Processing job
processor.run(
    code='scripts/evaluate.py',
    inputs=[
        ProcessingInput(
            source=estimator.model_data,  # model.tar.gz from training
            destination='/opt/ml/processing/model'
        ),
        ProcessingInput(
            source=test_uri,
            destination='/opt/ml/processing/test'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='evaluation',
            source='/opt/ml/processing/evaluation'
        )
    ]
)
```

### Evaluation Metrics to Track

| Metric | Formula | When to Use |
|--------|---------|-------------|
| AUC-ROC | Area under ROC curve | Binary classification (default) |
| Accuracy | Correct / Total | Balanced classes only |
| Precision | TP / (TP + FP) | When false positives are costly |
| Recall | TP / (TP + FN) | When false negatives are costly |
| F1 Score | 2 × (P×R)/(P+R) | Balance precision & recall |
| RMSE | √(mean squared error) | Regression |
| MAE | Mean absolute error | Regression (robust to outliers) |

### Quality Gates (Go/No-Go)

```python
# In your pipeline, define thresholds
QUALITY_GATES = {
    'min_auc': 0.75,
    'min_precision': 0.70,
    'min_recall': 0.65,
    'max_prediction_latency_ms': 100,
    'min_test_samples': 1000
}

# Only promote model if ALL gates pass
def check_quality(metrics):
    return (
        metrics['auc'] >= QUALITY_GATES['min_auc'] and
        metrics['precision'] >= QUALITY_GATES['min_precision'] and
        metrics['recall'] >= QUALITY_GATES['min_recall']
    )
```

---

## 10. Step 8: Model Registry

### What Is It?

A catalog of trained models with:
- Version numbers
- Metadata (metrics, parameters, data used)
- Approval status (Pending → Approved → Rejected)
- Deployment history

### Registering a Model

```python
from sagemaker.model import Model

model = Model(
    image_uri=xgb_image,
    model_data=estimator.model_data,
    role=ROLE_ARN,
    sagemaker_session=session
)

# Register in a Model Package Group
model_package = model.register(
    model_package_group_name='churn-prediction-models',
    content_types=['text/csv'],
    response_types=['text/csv'],
    inference_instances=['ml.m5.large'],
    transform_instances=['ml.m5.large'],
    approval_status='PendingManualApproval',
    description='XGBoost churn model v3 - AUC 0.87'
)

print(f"Model version ARN: {model_package.model_package_arn}")
```

### Approval Workflow

```
Developer trains model
    → Registers with status "PendingManualApproval"
    → Data scientist reviews metrics
    → Approves or Rejects
    → If Approved → triggers deployment pipeline
```

```python
# Approve a model (manually or via automation)
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus='Approved'
)
```

---

## 11. Step 9: Model Deployment

### Deployment Options

| Option | Latency | Use Case | Cost Model |
|--------|---------|----------|------------|
| Real-time Endpoint | ms | Live predictions, APIs | Per-hour (always on) |
| Serverless Endpoint | ms-sec | Sporadic traffic | Per-request |
| Batch Transform | minutes | Bulk scoring | Per-job |
| Async Endpoint | seconds | Large payloads, queued | Per-hour |

### Real-time Endpoint

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='churn-prediction-prod',
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer()
)

# Make predictions
result = predictor.predict('12,45.5,1200.0,0,1,2,1,0,3')
```

### Serverless Endpoint (Cost-Effective)

```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=10
)

predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='churn-prediction-serverless'
)
# You only pay when predictions are made!
```

### Batch Transform (Offline Scoring)

```python
transformer = estimator.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/predictions/'
)

transformer.transform(
    data='s3://bucket/data/to-score/',
    content_type='text/csv',
    split_type='Line'
)
transformer.wait()
# Results appear in s3://bucket/predictions/
```

### Multi-Model Endpoints (Advanced)

Host multiple models on one endpoint to save costs:

```python
from sagemaker.multidatamodel import MultiDataModel

multi_model = MultiDataModel(
    name='multi-churn-models',
    model_data_prefix=f's3://{bucket}/multi-models/',
    model=model,
    sagemaker_session=session
)
# Add models: multi_model.add_model(model_data_source=...)
```

### A/B Testing (Traffic Splitting)

```python
# Deploy two model variants
endpoint_config = sm_client.create_endpoint_config(
    EndpointConfigName='ab-test-config',
    ProductionVariants=[
        {
            'VariantName': 'ModelA',
            'ModelName': 'model-v1',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large',
            'InitialVariantWeight': 0.8  # 80% traffic
        },
        {
            'VariantName': 'ModelB',
            'ModelName': 'model-v2',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large',
            'InitialVariantWeight': 0.2  # 20% traffic
        }
    ]
)
```

---

## 12. Step 10: Monitoring & Observability

### SageMaker Model Monitor

Detects when your model's performance degrades in production.

#### Types of Monitoring

| Monitor Type | What It Detects | Example |
|-------------|-----------------|---------|
| Data Quality | Input data drift | Feature distributions changed |
| Model Quality | Prediction accuracy drop | AUC dropped below threshold |
| Bias | Fairness issues | Model biased against a group |
| Explainability | Feature importance shift | Different features driving predictions |

### Setting Up Data Quality Monitor

```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# Create baseline from training data
monitor.suggest_baseline(
    baseline_dataset=train_uri,
    dataset_format=DatasetFormat.csv(header=False)
)

# Schedule monitoring (runs hourly/daily)
monitor.create_monitoring_schedule(
    monitor_schedule_name='churn-model-monitor',
    endpoint_input=predictor.endpoint_name,
    output_s3_uri=f's3://{bucket}/monitoring/',
    schedule_cron_expression='cron(0 * ? * * *)'  # Every hour
)
```

### CloudWatch Alerts

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Alert when invocation errors spike
cloudwatch.put_metric_alarm(
    AlarmName='ChurnModel-HighErrorRate',
    MetricName='Invocation5XXErrors',
    Namespace='AWS/SageMaker',
    Statistic='Sum',
    Period=300,
    EvaluationPeriods=2,
    Threshold=10,
    ComparisonOperator='GreaterThanThreshold',
    Dimensions=[{'Name': 'EndpointName', 'Value': 'churn-prediction-prod'}],
    AlarmActions=['arn:aws:sns:region:account:alert-topic']
)
```

---

## 13. Step 11: SageMaker Pipelines (Full Automation)

### What Is a Pipeline?

A DAG (Directed Acyclic Graph) of steps that automates your entire ML workflow.

```
Preprocess → Train → Evaluate → [If AUC > 0.75] → Register → Deploy
```

### Complete Pipeline Example

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.step_collections import RegisterModel

# Parameters (overridable at runtime)
param_instance_type = ParameterString(name='TrainingInstance', default_value='ml.m5.xlarge')
param_auc_threshold = ParameterFloat(name='AucThreshold', default_value=0.75)

# Step 1: Preprocess
step_process = ProcessingStep(name='Preprocess', ...)

# Step 2: Train
step_train = TrainingStep(name='Train', ...)

# Step 3: Evaluate
evaluation_report = PropertyFile(name='EvalReport', output_name='evaluation', path='evaluation.json')
step_evaluate = ProcessingStep(name='Evaluate', ..., property_files=[evaluation_report])

# Step 4: Register (conditional)
step_register = RegisterModel(name='Register', ...)

# Condition: Only register if AUC passes
condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name='Evaluate', property_file=evaluation_report,
                 json_path='binary_classification_metrics.auc.value'),
    right=param_auc_threshold
)
step_condition = ConditionStep(name='CheckQuality', conditions=[condition],
                               if_steps=[step_register], else_steps=[])

# Assemble
pipeline = Pipeline(
    name='churn-prediction-pipeline',
    parameters=[param_instance_type, param_auc_threshold],
    steps=[step_process, step_train, step_evaluate, step_condition]
)

# Deploy pipeline
pipeline.upsert(role_arn=ROLE_ARN)

# Run it
execution = pipeline.start()
execution.wait()  # Optional: wait for completion
```

### Pipeline Triggers

| Trigger | How | Use Case |
|---------|-----|----------|
| Manual | `pipeline.start()` | Development, ad-hoc |
| Schedule | EventBridge rule | Daily/weekly retraining |
| Data arrival | S3 event → Lambda → Pipeline | New data triggers retrain |
| Model drift | Monitor alert → Lambda → Pipeline | Auto-retrain on drift |
| Code change | CI/CD (GitHub Actions) | New code triggers pipeline |

---

## 14. Step 12: CI/CD Integration

### GitHub Actions + SageMaker

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths: ['scripts/**', 'pipelines/**']

jobs:
  deploy-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: eu-west-1
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/
      - run: uv run python pipelines/pipeline.py
```

---

## 15. Cost Optimization

### Key Strategies

| Strategy | Savings | How |
|----------|---------|-----|
| Spot Training | Up to 90% | `use_spot_instances=True` in Estimator |
| Right-size instances | 30-50% | Start small, scale up if needed |
| Serverless endpoints | 70%+ | For sporadic traffic |
| Auto-scaling | Variable | Scale endpoints with demand |
| Delete idle endpoints | 100% | Automate cleanup |
| Multi-model endpoints | 50-80% | Share infrastructure |

### Spot Training

```python
estimator = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    use_spot_instances=True,              # Enable spot
    max_wait=7200,                        # Max wait time (seconds)
    max_run=3600,                         # Max training time
    checkpoint_s3_uri=f's3://{bucket}/checkpoints/'  # Resume if interrupted
)
```

---

## 16. Production Checklist

Before going to production, verify:

### Data
- [ ] Data pipeline is automated (no manual uploads)
- [ ] Data validation checks in place
- [ ] Schema enforcement (expected columns, types)
- [ ] Data versioning enabled

### Model
- [ ] Model registered in Model Registry
- [ ] Evaluation metrics meet quality gates
- [ ] Model tested on holdout set
- [ ] Bias/fairness checked
- [ ] Model explainability documented

### Infrastructure
- [ ] Endpoint auto-scaling configured
- [ ] Error handling and retries in inference code
- [ ] Logging enabled (CloudWatch)
- [ ] Model Monitor scheduled
- [ ] Alerts configured for errors and drift

### Operations
- [ ] Rollback plan documented
- [ ] A/B testing strategy defined
- [ ] Retraining trigger defined (schedule or drift-based)
- [ ] Cost monitoring in place
- [ ] IAM permissions follow least-privilege

### Documentation
- [ ] Model card created (what, why, limitations)
- [ ] API documentation for endpoint consumers
- [ ] Runbook for common issues
- [ ] Architecture diagram up to date

---

*End of SageMaker Production Guide*
