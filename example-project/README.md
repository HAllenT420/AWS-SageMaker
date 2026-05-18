# Example Project: Customer Churn Prediction (End-to-End)

> This is a hands-on, follow-along project that implements the SageMaker Production Guide step by step.
> 
> **Part 1**: Notebook (experimentation, all-in-one)  
> **Part 2**: Modular MLOps code (production-ready structure)

## Project Structure

```
example-project/
├── README.md                          ← You are here
├── STEP_BY_STEP_GUIDE.md             ← Follow this guide sequentially
│
├── part1-notebook/                    ← PART 1: Single notebook experiment
│   └── churn_experiment.ipynb
│
├── part2-mlops/                       ← PART 2: Production modular code
│   ├── config/
│   │   ├── config.py                  # Centralized configuration
│   │   └── params.json                # Hyperparameters & settings
│   ├── data/
│   │   └── generate_sample_data.py    # Generate synthetic data
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py          # Step 1: Pull data from sources
│   │   ├── preprocessing.py           # Step 2: Clean & transform
│   │   ├── feature_engineering.py     # Step 3: Create features
│   │   ├── training.py                # Step 4: Train model
│   │   ├── evaluation.py             # Step 5: Evaluate model
│   │   ├── registry.py               # Step 6: Register model
│   │   ├── deployment.py             # Step 7: Deploy endpoint
│   │   └── monitoring.py             # Step 8: Set up monitoring
│   ├── scripts/
│   │   ├── preprocess.py             # SageMaker Processing script
│   │   ├── train.py                  # SageMaker Training script
│   │   └── evaluate.py              # SageMaker Evaluation script
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── pipeline.py              # Full SageMaker Pipeline
│   ├── tests/
│   │   ├── test_preprocessing.py
│   │   └── test_training.py
│   ├── run_pipeline.py               # Entry point: run everything
│   └── pyproject.toml
│
└── diagrams/                          ← Architecture visuals
    └── project_flow.mmd
```

## How to Use This

1. **Read** `STEP_BY_STEP_GUIDE.md` — explains every decision
2. **Run** `part1-notebook/churn_experiment.ipynb` — see it work interactively
3. **Study** `part2-mlops/` — see how notebook code becomes production code
4. **Execute** `part2-mlops/run_pipeline.py` — run the full automated pipeline

## Prerequisites

- AWS CLI configured (`aws configure`)
- Python 3.9+
- uv installed
- A SageMaker execution role ARN
