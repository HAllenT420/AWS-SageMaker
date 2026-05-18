# Step-by-Step Guide: From Notebook to Production MLOps

> Follow this guide top to bottom. Each step builds on the previous one.
> No jumping ahead — the order matters.

---

## Phase 1: Setup (Do This First)

### Step 1.1: Install uv and create environment

```bash
# Open terminal in VS Code (Ctrl+`)
# Navigate to this folder
cd example-project/part2-mlops

# Install dependencies
uv sync
```

### Step 1.2: Configure AWS

```bash
aws configure
# AWS Access Key ID: <your-key>
# AWS Secret Access Key: <your-secret>
# Default region: eu-west-1 (or your region)
# Output format: json
```

### Step 1.3: Set your SageMaker Role

Edit `part2-mlops/config/params.json` and replace `YOUR_ACCOUNT_ID` and `YOUR_SAGEMAKER_ROLE` with your actual values.

To find your role:
1. Go to AWS Console → IAM → Roles
2. Search for "SageMaker"
3. Copy the ARN (looks like: `arn:aws:iam::123456789:role/SageMakerRole`)

---

## Phase 2: Experiment in Notebook (Part 1)

### Step 2.1: Open the notebook

Open `part1-notebook/churn_experiment.ipynb` in VS Code.

### Step 2.2: Select kernel

Click the kernel selector (top-right) → select your Python environment.

### Step 2.3: Run cells sequentially

The notebook does everything in one place:
1. Generates synthetic customer data
2. Explores the data (EDA)
3. Preprocesses (encode, scale, split)
4. Uploads to S3
5. Trains XGBoost on SageMaker
6. Evaluates the model
7. Deploys to endpoint
8. Makes predictions
9. Cleans up

**This is your "proof of concept" — validate the approach works before modularizing.**

---

## Phase 3: Modular MLOps Code (Part 2)

### Why Modularize?

| Notebook | Modular Code |
|----------|-------------|
| Hard to test | Each function testable |
| Hard to reuse | Import and reuse anywhere |
| No error handling | Proper try/except |
| Manual execution | Automated pipeline |
| Single developer | Team collaboration |

### Step 3.1: Understand the module structure

Each file in `src/` handles ONE step of the pipeline:

```
data_ingestion.py      → "Where does data come from?"
preprocessing.py       → "How do we clean it?"
feature_engineering.py → "What features do we create?"
training.py           → "How do we train?"
evaluation.py         → "Is the model good enough?"
registry.py           → "How do we version it?"
deployment.py         → "How do we serve it?"
monitoring.py         → "How do we watch it?"
```

### Step 3.2: Run the pipeline

```bash
cd part2-mlops
uv run python run_pipeline.py
```

This executes all steps in order, with proper logging and error handling.

### Step 3.3: Run tests

```bash
uv run pytest tests/ -v
```

---

## Phase 4: Understanding Each Module

### Module 1: `data_ingestion.py`

**What it does**: Connects to data sources, pulls raw data, saves to S3.

**In production**: This would connect to your actual database, API, or data lake.

**Key concept**: Never train directly from the source DB. Always:
1. Extract → 2. Save to data lake (S3) → 3. Train from data lake

### Module 2: `preprocessing.py`

**What it does**: Cleans data, handles missing values, encodes categoricals.

**Key concept**: This runs as a SageMaker Processing Job (not on your laptop).
The script in `scripts/preprocess.py` is what SageMaker actually executes.

### Module 3: `feature_engineering.py`

**What it does**: Creates new features from raw data.

**Key concept**: Feature engineering is where most model improvement comes from.
Good features > fancy algorithms.

### Module 4: `training.py`

**What it does**: Configures and launches a SageMaker Training Job.

**Key concept**: Your laptop just SUBMITS the job. The actual training runs on
AWS infrastructure (ml.m5.xlarge, etc.). You pay only while it runs.

### Module 5: `evaluation.py`

**What it does**: Evaluates the trained model, produces metrics, checks quality gates.

**Key concept**: If metrics don't pass the threshold, the pipeline STOPS.
No bad model ever reaches production.

### Module 6: `registry.py`

**What it does**: Registers the model in SageMaker Model Registry with version number.

**Key concept**: Every model is versioned. You can always roll back to a previous version.

### Module 7: `deployment.py`

**What it does**: Deploys the approved model to a SageMaker Endpoint.

**Key concept**: Supports multiple strategies:
- Direct replacement (simple)
- Canary (10% traffic first)
- Blue/Green (instant switch + rollback)

### Module 8: `monitoring.py`

**What it does**: Sets up Model Monitor to watch for drift.

**Key concept**: Models degrade over time. Monitoring catches this early
so you can retrain before users notice.

---

## Phase 5: The Full Pipeline (Automated)

### How `pipeline/pipeline.py` ties it all together

```
SageMaker Pipeline Definition:
    Step 1: ProcessingStep (runs preprocessing.py on SageMaker)
    Step 2: TrainingStep (runs training on SageMaker)
    Step 3: ProcessingStep (runs evaluation on SageMaker)
    Step 4: ConditionStep (if AUC > 0.75 → register, else → stop)
    Step 5: RegisterModel (add to Model Registry)
```

### Running the pipeline

```bash
# This creates/updates the pipeline in AWS and starts it
uv run python pipeline/pipeline.py
```

### Monitoring the pipeline

After starting, go to:
- AWS Console → SageMaker → Pipelines → churn-prediction-pipeline
- You'll see a visual DAG of your pipeline executing

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named sagemaker" | Run `uv sync` in part2-mlops/ |
| "Access Denied" | Check your IAM role has SageMaker permissions |
| "ResourceLimitExceeded" | Request a quota increase in AWS Console |
| "Could not assume role" | Verify the role ARN in params.json |
| Training job fails | Check CloudWatch Logs for the training job |

---

*Follow this guide top to bottom and you'll have a working production ML system.*
