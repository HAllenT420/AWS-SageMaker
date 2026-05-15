# Data Science Tools, Predictions & Orchestration

> Everything about how predictions are made, triggered, scheduled, and orchestrated in production.

---

## Table of Contents

1. [Types of Predictions](#1-types-of-predictions)
2. [How Predictions Are Triggered](#2-how-predictions-are-triggered)
3. [Orchestration Tools (Airflow, Step Functions, etc.)](#3-orchestration-tools)
4. [Experiment Tracking Tools](#4-experiment-tracking-tools)
5. [Data Versioning Tools](#5-data-versioning-tools)
6. [Model Serving Frameworks](#6-model-serving-frameworks)
7. [Feature Stores](#7-feature-stores)
8. [Complete Tool Landscape](#8-complete-tool-landscape)

---

## 1. Types of Predictions

### Batch Predictions (Offline)

**What**: Score a large dataset all at once, store results for later use.

**When to use**:
- Predictions don't need to be instant
- You have a fixed set of entities to score (all customers, all products)
- Results are consumed by dashboards, reports, or downstream systems
- Cost-sensitive (batch is cheaper than real-time)

**Example**: Every night, score all 1M customers for churn probability. Marketing team uses results next morning.

```python
# SageMaker Batch Transform
transformer = model.transformer(
    instance_count=2,
    instance_type='ml.m5.xlarge',
    output_path='s3://bucket/predictions/2024-01-15/'
)
transformer.transform(data='s3://bucket/data/all_customers.csv')
```

**Typical Schedule**: Daily, weekly, or on new data arrival.

---

### Real-Time Predictions (Online)

**What**: Get a prediction instantly (milliseconds) for a single request.

**When to use**:
- User-facing applications (show recommendation NOW)
- Decision needs to happen immediately (fraud detection)
- Input data is only available at request time

**Example**: User adds item to cart → call endpoint → show "Customers also bought..."

```python
# SageMaker Real-Time Endpoint
predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)
result = predictor.predict(customer_features)  # Returns in ~50ms
```

**Cost**: You pay per hour the endpoint is running (even with no traffic).

---

### Near Real-Time (Streaming)

**What**: Process events as they arrive with slight delay (seconds to minutes).

**When to use**:
- High-volume event streams
- Aggregated features (last 5 minutes of activity)
- Don't need sub-second response

**Example**: Detect anomalous login patterns by analyzing the last 10 login events.

```
Events → Kinesis Stream → Lambda (aggregate features) → SageMaker Endpoint → Action
```

---

### Async Predictions

**What**: Submit request, get result later (seconds to minutes).

**When to use**:
- Large payloads (images, documents, long text)
- Processing takes >60 seconds
- Client can poll for results

```python
# SageMaker Async Endpoint
from sagemaker.async_inference import AsyncInferenceConfig

async_config = AsyncInferenceConfig(
    output_path='s3://bucket/async-results/',
    notification_config={
        'SuccessTopic': 'arn:aws:sns:...:success',
        'ErrorTopic': 'arn:aws:sns:...:error'
    }
)
predictor = model.deploy(async_inference_config=async_config, ...)
```

---

### Comparison

| Type | Latency | Cost | Use Case |
|------|---------|------|----------|
| Batch | Minutes-hours | Lowest | Bulk scoring, reports |
| Real-time | Milliseconds | Highest | User-facing, instant decisions |
| Near real-time | Seconds | Medium | Event streams, aggregations |
| Async | Seconds-minutes | Medium | Large payloads, heavy processing |

---

## 2. How Predictions Are Triggered

### Trigger Patterns

| Pattern | Trigger | Example |
|---------|---------|---------|
| **API Call** | HTTP request | User action → API Gateway → Endpoint |
| **Schedule** | Cron/timer | Every day at 2 AM, score all customers |
| **Event-driven** | Data arrives | New file in S3 → trigger batch scoring |
| **Queue-based** | Message in queue | SQS message → Lambda → Endpoint |
| **Stream** | Real-time event | Kinesis event → process → predict |
| **Manual** | Human clicks button | Data scientist triggers from notebook |

### Pattern 1: API-Triggered (Real-Time)

```
Mobile App → API Gateway → Lambda → SageMaker Endpoint → Response
                                          ↑
                                    Feature Store (get customer features)
```

```python
# Lambda function behind API Gateway
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

def handler(event, context):
    customer_id = event['pathParameters']['customer_id']
    
    # Get features from Feature Store or DynamoDB
    features = get_features(customer_id)
    
    # Call SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName='churn-prediction-prod',
        ContentType='text/csv',
        Body=','.join(map(str, features))
    )
    
    prediction = json.loads(response['Body'].read())
    return {
        'statusCode': 200,
        'body': json.dumps({'churn_probability': prediction})
    }
```

### Pattern 2: Schedule-Triggered (Batch)

```
EventBridge (cron) → Lambda → Start SageMaker Batch Transform → S3 → Redshift
```

```python
# EventBridge rule (runs daily at 2 AM UTC)
{
    "schedule": "cron(0 2 * * ? *)",
    "target": "arn:aws:lambda:...:trigger-batch-scoring"
}
```

### Pattern 3: Event-Triggered (New Data)

```
New file in S3 → S3 Event → Lambda → Start SageMaker Pipeline
```

```python
# S3 event notification triggers Lambda
def handler(event, context):
    # New data file arrived
    s3_key = event['Records'][0]['s3']['object']['key']
    
    # Trigger the full pipeline
    sm = boto3.client('sagemaker')
    sm.start_pipeline_execution(
        PipelineName='churn-prediction-pipeline',
        PipelineParameters=[
            {'Name': 'InputDataUrl', 'Value': f's3://bucket/{s3_key}'}
        ]
    )
```

---

## 3. Orchestration Tools

### What Is Orchestration?

Orchestration = managing the execution order, dependencies, retries, and scheduling of multi-step workflows.

### Tool Comparison

| Tool | Type | Best For | Learning Curve |
|------|------|----------|---------------|
| **Apache Airflow (MWAA)** | DAG-based | Complex workflows, scheduling | Medium |
| **SageMaker Pipelines** | ML-specific | Pure SageMaker workflows | Low |
| **AWS Step Functions** | State machine | AWS-native, visual | Low |
| **Prefect** | Python-native | Modern Airflow alternative | Low |
| **Dagster** | Asset-based | Data-aware orchestration | Medium |
| **Kubeflow Pipelines** | Kubernetes | K8s-native ML pipelines | High |
| **Argo Workflows** | Kubernetes | Container-based workflows | High |

---

### Apache Airflow (Industry Standard)

**What**: The most widely used workflow orchestrator in data engineering and ML.

**Key Concepts**:
- **DAG** (Directed Acyclic Graph): Your workflow definition
- **Task**: A single unit of work
- **Operator**: Pre-built task types (BashOperator, PythonOperator, SageMakerOperator)
- **Sensor**: Wait for a condition (file exists, API returns 200)
- **XCom**: Pass data between tasks
- **Schedule**: When to run (cron expression)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.sagemaker import (
    SageMakerProcessingOperator,
    SageMakerTrainingOperator,
    SageMakerTransformOperator
)
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['team@company.com']
}

with DAG(
    'ml_churn_pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    # Wait for new data
    wait_for_data = S3KeySensor(
        task_id='wait_for_data',
        bucket_name='my-bucket',
        bucket_key='data/daily/{{ ds }}/*.csv',
        timeout=3600
    )

    # Preprocess
    preprocess = SageMakerProcessingOperator(
        task_id='preprocess',
        config={...}  # Processing job config
    )

    # Train
    train = SageMakerTrainingOperator(
        task_id='train',
        config={...}  # Training job config
    )

    # Evaluate
    evaluate = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate_model
    )

    # Deploy (conditional)
    deploy = SageMakerTransformOperator(
        task_id='batch_predict',
        config={...}
    )

    # Define order
    wait_for_data >> preprocess >> train >> evaluate >> deploy
```

**AWS Managed Airflow (MWAA)**:
- Fully managed — no server maintenance
- Auto-scaling workers
- Integrated with AWS services
- Cost: ~$0.50/hr for smallest environment

---

### SageMaker Pipelines (AWS Native)

**Best for**: Pure ML workflows that stay within SageMaker.

**Advantages over Airflow**:
- Tighter SageMaker integration
- Built-in experiment tracking
- Visual pipeline graph in console
- No separate infrastructure to manage

**Disadvantages**:
- Limited to SageMaker steps
- Less flexible for non-ML tasks
- Smaller community

---

### AWS Step Functions

**Best for**: Lightweight orchestration, visual workflows, AWS-native.

**Advantages**:
- Visual designer
- Built-in error handling and retries
- Direct integration with 200+ AWS services
- Serverless (pay per state transition)

**When to choose over Airflow**:
- Simple workflows (< 10 steps)
- Don't need complex scheduling
- Team prefers visual/low-code
- Want serverless (no infrastructure)

---

### Prefect

**What**: Modern Python-native orchestrator (often called "Airflow 2.0")

**Advantages over Airflow**:
- Pure Python (no DAG boilerplate)
- Better local development experience
- Dynamic workflows (loops, conditionals)
- Easier testing

```python
from prefect import flow, task

@task
def preprocess_data():
    # Your preprocessing logic
    return processed_data

@task
def train_model(data):
    # Your training logic
    return model

@flow(name="ml-pipeline")
def ml_pipeline():
    data = preprocess_data()
    model = train_model(data)
    return model

# Run locally or deploy to Prefect Cloud
ml_pipeline()
```

---

### Decision Matrix: Which Orchestrator?

| Criteria | Airflow | SageMaker Pipelines | Step Functions | Prefect |
|----------|---------|-------------------|----------------|---------|
| ML-specific features | Medium | High | Low | Medium |
| Non-ML tasks | High | Low | High | High |
| Scheduling | Excellent | Basic | Via EventBridge | Good |
| Visual UI | Good | Good | Excellent | Good |
| Local development | Poor | Poor | Poor | Excellent |
| Community/ecosystem | Huge | Growing | Large | Growing |
| Infrastructure | Managed (MWAA) | Serverless | Serverless | Cloud/self-host |
| Cost | $$$ | $ | $ | $-$$ |
| Learning curve | Medium | Low | Low | Low |

**My recommendation**: 
- Start with **SageMaker Pipelines** for pure ML workflows
- Add **Airflow (MWAA)** when you need complex scheduling or non-ML steps
- Use **Step Functions** for simple event-driven triggers

---

## 4. Experiment Tracking Tools

### What Is Experiment Tracking?

Recording every training run: hyperparameters, metrics, code version, data version, artifacts.

| Tool | Type | Best For |
|------|------|----------|
| **SageMaker Experiments** | AWS-native | SageMaker users |
| **MLflow** | Open source | Framework-agnostic, self-hosted |
| **Weights & Biases (W&B)** | SaaS | Deep learning, visualization |
| **Neptune.ai** | SaaS | Team collaboration |
| **Comet ML** | SaaS | Enterprise |

### MLflow (Most Popular Open Source)

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    mlflow.log_params({"max_depth": 5, "eta": 0.2, "num_round": 100})
    
    # Train model...
    
    mlflow.log_metrics({"auc": 0.87, "f1": 0.82})
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("evaluation_report.json")
```

### SageMaker Experiments

```python
from sagemaker.experiments import Run

with Run(experiment_name="churn-experiment", run_name="xgb-v3") as run:
    run.log_parameter("max_depth", 5)
    run.log_parameter("eta", 0.2)
    
    # After training
    run.log_metric("auc", 0.87)
    run.log_metric("f1", 0.82)
```

---

## 5. Data Versioning Tools

| Tool | How It Works | Best For |
|------|-------------|----------|
| **DVC** | Git-like for data files | Teams using Git |
| **LakeFS** | Git-like for data lakes | S3-based data lakes |
| **Delta Lake** | Versioned Parquet tables | Spark users |
| **S3 Versioning** | Built-in S3 feature | Simple projects |

### DVC (Data Version Control)

```bash
# Initialize DVC in your project
dvc init

# Track a large data file
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add training data v1"

# Push data to remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

# Switch to a different data version
git checkout v2-data
dvc checkout
```

---

## 6. Model Serving Frameworks

### Beyond SageMaker Endpoints

| Framework | Best For | Deployment |
|-----------|----------|------------|
| **SageMaker Endpoints** | AWS-native, managed | AWS |
| **TorchServe** | PyTorch models | Any |
| **TensorFlow Serving** | TF models | Any |
| **Triton Inference Server** | Multi-framework, GPU | Any |
| **BentoML** | Easy packaging | Any cloud |
| **Seldon Core** | Kubernetes-native | K8s |
| **FastAPI** | Simple REST API | Any |

### FastAPI (Simplest Custom Serving)

```python
from fastapi import FastAPI
import xgboost as xgb
import numpy as np

app = FastAPI()
model = xgb.Booster()
model.load_model("model.json")

@app.post("/predict")
async def predict(features: list[float]):
    dmatrix = xgb.DMatrix(np.array([features]))
    probability = float(model.predict(dmatrix)[0])
    return {
        "churn_probability": probability,
        "prediction": "churn" if probability > 0.5 else "no_churn"
    }
```

---

## 7. Feature Stores

| Tool | Type | Best For |
|------|------|----------|
| **SageMaker Feature Store** | AWS-managed | SageMaker users |
| **Feast** | Open source | Multi-cloud, flexible |
| **Tecton** | SaaS | Enterprise, real-time |
| **Hopsworks** | Open source | Full ML platform |

---

## 8. Complete Tool Landscape

### The Full MLOps Stack

```
┌─────────────────────────────────────────────────────────────┐
│                        DEVELOPMENT                            │
├─────────────────────────────────────────────────────────────┤
│  IDE: VS Code, JupyterLab, SageMaker Studio                 │
│  Version Control: Git, GitHub, GitLab                        │
│  Package Mgmt: uv, pip, conda                               │
│  Notebooks: Jupyter, VS Code Notebooks                       │
├─────────────────────────────────────────────────────────────┤
│                        DATA                                   │
├─────────────────────────────────────────────────────────────┤
│  Storage: S3, Redshift, DynamoDB                             │
│  Processing: Glue, Spark (EMR), Pandas                       │
│  Versioning: DVC, LakeFS, S3 Versioning                      │
│  Feature Store: SageMaker FS, Feast, Tecton                  │
│  Catalog: Glue Data Catalog, DataHub                         │
├─────────────────────────────────────────────────────────────┤
│                        TRAINING                               │
├─────────────────────────────────────────────────────────────┤
│  Compute: SageMaker Training, EC2, EMR                       │
│  Frameworks: XGBoost, PyTorch, TensorFlow, scikit-learn      │
│  Experiment Tracking: MLflow, SageMaker Experiments, W&B     │
│  HPO: SageMaker Tuning, Optuna, Ray Tune                     │
├─────────────────────────────────────────────────────────────┤
│                        DEPLOYMENT                             │
├─────────────────────────────────────────────────────────────┤
│  Serving: SageMaker Endpoints, Triton, FastAPI               │
│  Containers: Docker, ECR, ECS, EKS                           │
│  API: API Gateway, ALB                                       │
│  Serverless: Lambda, SageMaker Serverless                    │
├─────────────────────────────────────────────────────────────┤
│                        ORCHESTRATION                          │
├─────────────────────────────────────────────────────────────┤
│  Pipelines: SageMaker Pipelines, Airflow, Step Functions     │
│  CI/CD: GitHub Actions, CodePipeline, Jenkins                │
│  Scheduling: EventBridge, Airflow Scheduler                  │
├─────────────────────────────────────────────────────────────┤
│                        MONITORING                             │
├─────────────────────────────────────────────────────────────┤
│  Model: SageMaker Model Monitor, Evidently AI                │
│  Infra: CloudWatch, Datadog, Grafana                         │
│  Alerts: SNS, PagerDuty, Slack                               │
└─────────────────────────────────────────────────────────────┘
```

---

*End of Tools & Orchestration Guide*
