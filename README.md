# SageMaker MLOps Project

End-to-end ML project using AWS SageMaker, developed locally in VS Code.

## Project Structure

```
sagemaker-mlops-project/
├── notebooks/
│   └── 01_experimentation.ipynb      # Interactive exploration & prototyping
├── pipelines/
│   ├── __init__.py
│   ├── pipeline.py                   # SageMaker Pipeline definition
│   └── config.py                     # Pipeline configuration
├── scripts/
│   ├── preprocess.py                 # Data preprocessing script
│   ├── train.py                      # Training script
│   └── evaluate.py                   # Model evaluation script
├── src/
│   ├── __init__.py
│   └── inference.py                  # Custom inference logic
├── tests/
│   └── test_preprocess.py            # Unit tests
├── data/
│   └── .gitkeep                      # Local data folder (not committed)
├── config/
│   └── pipeline_params.json          # Environment-specific parameters
├── requirements.txt
├── setup.py
├── Makefile                          # Common commands
└── README.md
```

## Prerequisites

1. **uv** - Fast Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
2. **AWS CLI** configured with credentials (`aws configure`)
3. **Python 3.9+**
4. **VS Code** with Python and Jupyter extensions
5. An **IAM Role** with SageMaker permissions

## Quick Start

```bash
# Install uv (if not already installed)
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create project environment and install all dependencies
uv sync

# Install with dev dependencies (pytest, ruff)
uv sync --all-extras

# Register the kernel for Jupyter notebooks in VS Code
uv run python -m ipykernel install --user --name sagemaker-mlops --display-name "SageMaker MLOps"

# Run the notebook for experimentation
# Open notebooks/01_experimentation.ipynb in VS Code and select the "SageMaker MLOps" kernel

# Run the pipeline
uv run python pipelines/pipeline.py

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## Workflow

1. **Experiment** in the notebook (notebooks/01_experimentation.ipynb)
2. **Refactor** working code into scripts/ and pipelines/
3. **Test** locally with `uv run pytest`
4. **Deploy** the pipeline to SageMaker with `uv run python pipelines/pipeline.py`

## Learning Guides

Comprehensive reference notebooks in `notebooks/learning/`:

| # | Guide | What You'll Learn |
|---|-------|-------------------|
| 02 | [SageMaker Production Guide](notebooks/learning/02_sagemaker_production_guide.md) | Step-by-step production ML on SageMaker (all components) |
| 03 | [AWS Services for ML](notebooks/learning/03_aws_services_for_ml.md) | Every AWS service used in ML — what, when, why |
| 04 | [Tools & Orchestration](notebooks/learning/04_ds_tools_and_orchestration.md) | Airflow, predictions, scheduling, MLOps tools |
| 05 | [CI/CD & GitHub](notebooks/learning/05_cicd_and_github_guide.md) | Git, GitHub, CI/CD from absolute scratch |
| 06 | [Data Science Bible](notebooks/learning/06_data_science_bible.md) | End-to-end DS process, all metrics, algorithms, checks |
