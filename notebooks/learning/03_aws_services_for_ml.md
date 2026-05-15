# AWS Services for Machine Learning & Data Science

> Complete reference of AWS services used in ML projects — what each does, when to use it, and how they connect.

---

## Table of Contents

1. [Service Map: Which Service for What?](#1-service-map)
2. [Data Storage & Databases](#2-data-storage--databases)
3. [Data Processing & ETL](#3-data-processing--etl)
4. [Machine Learning Services](#4-machine-learning-services)
5. [MLOps & Deployment](#5-mlops--deployment)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Security & Access](#7-security--access)
8. [Networking & Compute](#8-networking--compute)
9. [AI/ML Application Services](#9-aiml-application-services)
10. [How They All Connect (Architecture Patterns)](#10-architecture-patterns)

---

## 1. Service Map

### Quick Reference: "I need to ___"

| I need to... | Use This Service |
|-------------|-----------------|
| Store raw files/data | **S3** |
| Store structured data (SQL) | **RDS** or **Aurora** |
| Store NoSQL data | **DynamoDB** |
| Store time-series data | **Timestream** |
| Build a data warehouse | **Redshift** |
| Run SQL on S3 data | **Athena** |
| ETL / data transformation | **Glue** |
| Stream real-time data | **Kinesis** |
| Train ML models | **SageMaker** |
| Auto-ML (no code) | **SageMaker Canvas** or **AutoPilot** |
| Deploy ML models | **SageMaker Endpoints** |
| Run batch predictions | **SageMaker Batch Transform** |
| Schedule workflows | **Step Functions** or **MWAA (Airflow)** |
| Orchestrate ML pipelines | **SageMaker Pipelines** |
| Monitor models | **SageMaker Model Monitor** |
| Run serverless code | **Lambda** |
| Build APIs | **API Gateway** |
| Container orchestration | **ECS** or **EKS** |
| CI/CD | **CodePipeline** + **CodeBuild** |
| Store secrets | **Secrets Manager** |
| Manage permissions | **IAM** |
| Send notifications | **SNS** |
| Queue messages | **SQS** |
| Schedule events | **EventBridge** |
| Log everything | **CloudWatch** |
| Use pre-built AI (vision, NLP) | **Rekognition**, **Comprehend**, **Textract** |
| LLMs / Generative AI | **Bedrock** |

---

## 2. Data Storage & Databases

### Amazon S3 (Simple Storage Service)

**What**: Object storage for any file type (CSV, Parquet, images, model artifacts)

**Use in ML**: Store training data, model artifacts, pipeline outputs, logs

```python
import boto3

s3 = boto3.client('s3')

# Upload
s3.upload_file('data.csv', 'my-bucket', 'project/data/raw/data.csv')

# Download
s3.download_file('my-bucket', 'project/data/raw/data.csv', 'local_data.csv')

# List objects
response = s3.list_objects_v2(Bucket='my-bucket', Prefix='project/data/')
```

**Key Features**:
- Storage classes: Standard, Infrequent Access, Glacier (archival)
- Versioning: Keep history of file changes
- Lifecycle rules: Auto-move old data to cheaper storage
- Event notifications: Trigger Lambda when files arrive

---

### Amazon RDS (Relational Database Service)

**What**: Managed SQL databases (PostgreSQL, MySQL, SQL Server, Oracle)

**Use in ML**: Source data for features, store prediction results, application data

```python
import psycopg2

conn = psycopg2.connect(
    host='my-db.cluster-xyz.eu-west-1.rds.amazonaws.com',
    database='analytics',
    user='admin',
    password='...'
)
df = pd.read_sql('SELECT * FROM customers WHERE active = true', conn)
```

---

### Amazon Redshift

**What**: Data warehouse for analytics (columnar storage, massively parallel)

**Use in ML**: Large-scale feature computation, historical analysis, BI + ML

**When to use over RDS**: When you have billions of rows and need fast aggregations

```python
# Query Redshift from SageMaker
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Or use Redshift ML to train models directly in SQL!
# CREATE MODEL churn_model FROM customers FUNCTION predict_churn IAM_ROLE '...'
```

---

### Amazon DynamoDB

**What**: NoSQL key-value database (single-digit millisecond latency)

**Use in ML**: Store real-time features, cache predictions, feature flags

```python
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('customer-features')

# Store a prediction result
table.put_item(Item={
    'customer_id': '12345',
    'churn_probability': '0.73',
    'prediction_timestamp': '2024-01-15T10:30:00Z'
})

# Retrieve for real-time serving
response = table.get_item(Key={'customer_id': '12345'})
```

---

### Amazon Athena

**What**: Serverless SQL query engine for S3 data

**Use in ML**: Ad-hoc exploration of data in S3 without loading into a database

```sql
-- Query CSV/Parquet files directly in S3
SELECT customer_id, tenure_months, churn
FROM "my_database"."customer_data"
WHERE tenure_months < 12
LIMIT 1000;
```

**Cost**: $5 per TB scanned (use Parquet format to reduce scans by 90%)

---

### Comparison Table

| Service | Data Type | Latency | Scale | Cost Model | Best For |
|---------|-----------|---------|-------|------------|----------|
| S3 | Any file | 100ms+ | Unlimited | Per GB stored + requests | Raw data, artifacts |
| RDS | Relational | 1-10ms | TB scale | Per hour (instance) | Application data |
| Redshift | Analytical | Seconds | PB scale | Per hour (cluster) | Data warehouse |
| DynamoDB | Key-value | <10ms | Unlimited | Per request + storage | Real-time lookups |
| Athena | S3 files | Seconds | PB scale | Per TB scanned | Ad-hoc queries |

---

## 3. Data Processing & ETL

### AWS Glue

**What**: Serverless ETL (Extract, Transform, Load) service

**Use in ML**: Data cleaning, transformation, catalog management

**Components**:
- **Glue Crawlers**: Auto-discover schema from S3/databases
- **Glue Jobs**: Run Spark/Python ETL scripts
- **Glue Data Catalog**: Central metadata store (like a data dictionary)
- **Glue DataBrew**: Visual data preparation (no code)

```python
# Glue Job (runs on Spark)
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext

sc = SparkContext()
glueContext = GlueContext(sc)

# Read from catalog
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="my_database",
    table_name="raw_customers"
)

# Transform
mapped = ApplyMapping.apply(frame=datasource, mappings=[
    ("customer_id", "string", "customer_id", "string"),
    ("tenure", "int", "tenure_months", "int"),
])

# Write to S3
glueContext.write_dynamic_frame.from_options(
    frame=mapped,
    connection_type="s3",
    connection_options={"path": "s3://bucket/processed/"},
    format="parquet"
)
```

---

### Amazon Kinesis

**What**: Real-time data streaming

**Use in ML**: Stream predictions, real-time feature computation, event processing

**Components**:
- **Kinesis Data Streams**: Raw event ingestion
- **Kinesis Data Firehose**: Auto-deliver to S3/Redshift
- **Kinesis Data Analytics**: SQL/Flink on streams

```
User clicks → Kinesis Stream → Lambda (compute features) → SageMaker Endpoint → DynamoDB (store result)
```

---

### Amazon EMR (Elastic MapReduce)

**What**: Managed Hadoop/Spark clusters

**Use in ML**: Large-scale data processing when Glue isn't enough

**When to use over Glue**: 
- Need fine-grained Spark tuning
- Long-running clusters
- Complex ML pipelines on Spark (MLlib)

---

### AWS Step Functions

**What**: Serverless workflow orchestration (state machines)

**Use in ML**: Coordinate multi-step workflows, error handling, human approval

```json
{
  "StartAt": "PreprocessData",
  "States": {
    "PreprocessData": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:preprocess",
      "Next": "TrainModel"
    },
    "TrainModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Next": "EvaluateModel"
    },
    "EvaluateModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:evaluate",
      "Next": "ApprovalGate"
    },
    "ApprovalGate": {
      "Type": "Choice",
      "Choices": [{"Variable": "$.auc", "NumericGreaterThan": 0.75, "Next": "Deploy"}],
      "Default": "Fail"
    }
  }
}
```

---

### Amazon MWAA (Managed Workflows for Apache Airflow)

**What**: Managed Apache Airflow — the industry-standard workflow orchestrator

**Use in ML**: Complex DAGs, cross-service orchestration, scheduling

**When to use over Step Functions**: 
- Team already knows Airflow
- Need complex scheduling (cron, dependencies)
- Want Python-native DAG definitions
- Need extensive community operators

```python
# Airflow DAG for ML pipeline
from airflow import DAG
from airflow.providers.amazon.aws.operators.sagemaker import (
    SageMakerProcessingOperator,
    SageMakerTrainingOperator,
    SageMakerEndpointOperator
)

with DAG('ml_pipeline', schedule_interval='@weekly') as dag:
    preprocess = SageMakerProcessingOperator(task_id='preprocess', ...)
    train = SageMakerTrainingOperator(task_id='train', ...)
    deploy = SageMakerEndpointOperator(task_id='deploy', ...)

    preprocess >> train >> deploy
```

---

## 4. Machine Learning Services

### Amazon SageMaker (Core ML Platform)

| Component | Purpose |
|-----------|---------|
| Studio | Cloud IDE for ML |
| Notebooks | Jupyter in the cloud |
| Processing | Data prep at scale |
| Training | Model training on managed infra |
| Tuning | Hyperparameter optimization |
| Feature Store | Centralized feature management |
| Model Registry | Model versioning & approval |
| Endpoints | Real-time inference |
| Batch Transform | Offline predictions |
| Pipelines | ML workflow automation |
| Model Monitor | Production monitoring |
| Clarify | Bias detection & explainability |
| Canvas | No-code ML |
| AutoPilot | AutoML |
| JumpStart | Pre-trained models & solutions |
| Ground Truth | Data labeling |

---

### Amazon Bedrock

**What**: Managed access to foundation models (LLMs)

**Models Available**: Claude (Anthropic), Titan (Amazon), Llama (Meta), Mistral, Stable Diffusion

**Use Cases**: Text generation, summarization, RAG, chatbots, code generation

```python
import boto3

bedrock = boto3.client('bedrock-runtime')

response = bedrock.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": "Explain gradient boosting"}],
        "max_tokens": 500
    })
)
```

---

## 5. MLOps & Deployment

### AWS CodePipeline + CodeBuild

**What**: CI/CD for ML code

```
Code Push → CodePipeline → CodeBuild (test + lint) → Deploy Pipeline → SageMaker
```

### Amazon ECR (Elastic Container Registry)

**What**: Docker image storage

**Use in ML**: Store custom training/inference containers

```bash
# Build and push custom container
docker build -t my-training .
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker tag my-training:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/my-training:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/my-training:latest
```

### AWS Lambda

**What**: Serverless functions (run code without servers)

**Use in ML**: 
- Trigger pipelines on events
- Lightweight inference (small models)
- Pre/post-processing
- Glue between services

```python
# Lambda to trigger SageMaker pipeline on new data
def handler(event, context):
    import boto3
    sm = boto3.client('sagemaker')
    sm.start_pipeline_execution(
        PipelineName='churn-prediction-pipeline',
        PipelineParameters=[
            {'Name': 'InputDataUrl', 'Value': event['s3_uri']}
        ]
    )
```

---

## 6. Monitoring & Observability

### Amazon CloudWatch

**What**: Monitoring, logging, and alerting for all AWS services

**Use in ML**:
- Training job metrics (loss curves)
- Endpoint latency and errors
- Custom metrics from your code
- Log analysis

### Amazon SNS (Simple Notification Service)

**What**: Pub/sub messaging for notifications

**Use in ML**: Alert on model drift, pipeline failures, approval requests

### Amazon EventBridge

**What**: Event bus for connecting services

**Use in ML**: Schedule retraining, react to S3 uploads, trigger on model approval

---

## 7. Security & Access

### AWS IAM (Identity and Access Management)

**What**: Who can do what on which resources

**Key Concepts**:
- **User**: A person or application
- **Role**: A set of permissions that can be assumed
- **Policy**: JSON document defining permissions
- **SageMaker Execution Role**: What SageMaker can access on your behalf

### AWS Secrets Manager

**What**: Store and rotate secrets (API keys, DB passwords)

```python
import boto3

secrets = boto3.client('secretsmanager')
response = secrets.get_secret_value(SecretId='my-db-credentials')
creds = json.loads(response['SecretString'])
```

### AWS KMS (Key Management Service)

**What**: Encryption key management

**Use in ML**: Encrypt data at rest in S3, encrypt model artifacts

---

## 8. Networking & Compute

### Amazon VPC (Virtual Private Cloud)

**What**: Isolated network for your resources

**Use in ML**: Run SageMaker in private subnet (no internet access for security)

### Amazon EC2

**What**: Virtual machines

**Use in ML**: Custom training setups, development environments, GPU instances

### Amazon ECS / EKS

**What**: Container orchestration (Docker)

**Use in ML**: Deploy models as containers, microservices architecture

---

## 9. AI/ML Application Services (Pre-built AI)

These are ready-to-use AI services — no model training needed:

| Service | What It Does | Example Use |
|---------|-------------|-------------|
| **Rekognition** | Image/video analysis | Face detection, object recognition |
| **Comprehend** | NLP | Sentiment analysis, entity extraction |
| **Textract** | Document processing | Extract text from PDFs, forms |
| **Translate** | Language translation | Multi-language support |
| **Polly** | Text-to-speech | Voice interfaces |
| **Transcribe** | Speech-to-text | Meeting transcription |
| **Forecast** | Time-series forecasting | Demand prediction |
| **Personalize** | Recommendations | Product recommendations |
| **Fraud Detector** | Fraud detection | Transaction screening |
| **Kendra** | Enterprise search | Intelligent document search |

---

## 10. Architecture Patterns

### Pattern 1: Simple Batch ML Pipeline

```
S3 (raw data) → Glue (ETL) → S3 (processed) → SageMaker Training → Model Registry
                                                                          ↓
Schedule (EventBridge) → SageMaker Batch Transform → S3 (predictions) → Redshift
```

### Pattern 2: Real-Time Prediction API

```
Client → API Gateway → Lambda → SageMaker Endpoint → Response
                                       ↑
                              Model from Registry
```

### Pattern 3: Streaming ML

```
Events → Kinesis → Lambda (features) → SageMaker Endpoint → DynamoDB → Dashboard
                       ↓
                  S3 (archive) → Periodic Retraining
```

### Pattern 4: Full MLOps

```
GitHub → CodePipeline → CodeBuild (test) → SageMaker Pipeline
                                                    ↓
                                           Processing → Training → Evaluation
                                                                       ↓
                                                              [AUC > 0.75?]
                                                              Yes ↓      No ↓
                                                          Register    Alert Team
                                                              ↓
                                                    Manual Approval (SNS)
                                                              ↓
                                                    Deploy to Endpoint
                                                              ↓
                                                    Model Monitor → EventBridge → Retrain
```

---

*End of AWS Services Guide*
