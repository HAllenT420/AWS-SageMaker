# CI/CD Pipeline & GitHub: Complete Beginner's Guide

> Explained from scratch — as if you've never used Git or CI/CD before.
> By the end, you'll understand how code goes from your laptop to production.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Git: The Foundation](#2-git-the-foundation)
3. [GitHub: Collaboration Platform](#3-github-collaboration-platform)
4. [Branching Strategy](#4-branching-strategy)
5. [Pull Requests (Code Review)](#5-pull-requests)
6. [What Is CI/CD?](#6-what-is-cicd)
7. [CI: Continuous Integration](#7-ci-continuous-integration)
8. [CD: Continuous Delivery/Deployment](#8-cd-continuous-deliverydeployment)
9. [GitHub Actions (Hands-On)](#9-github-actions)
10. [CI/CD for ML Projects](#10-cicd-for-ml-projects)
11. [AWS CI/CD Services](#11-aws-cicd-services)
12. [Complete Example: ML Project CI/CD](#12-complete-example)
13. [Best Practices](#13-best-practices)

---

## 1. What Problem Does This Solve?

### Without CI/CD (The Old Way)

```
Developer writes code on laptop
    → Manually copies files to server
    → Hopes nothing breaks
    → If it breaks, nobody knows who changed what
    → Rollback = "does anyone have yesterday's version?"
```

### With CI/CD (The Modern Way)

```
Developer writes code
    → Pushes to Git (version controlled)
    → Automated tests run (catches bugs immediately)
    → Code review by team (quality check)
    → Automated deployment (consistent, repeatable)
    → If it breaks, auto-rollback + team notified
```

**CI/CD = Automation that ensures your code is always tested, reviewed, and deployed safely.**

---

## 2. Git: The Foundation

### What Is Git?

Git is a **version control system** — it tracks every change to every file, who made it, and when.

Think of it like "Track Changes" in Word, but for your entire project.

### Key Concepts

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **Repository (repo)** | Your project folder tracked by Git | A filing cabinet |
| **Commit** | A snapshot of your project at a point in time | A save point in a game |
| **Branch** | A parallel version of your code | A draft copy |
| **Merge** | Combining two branches | Accepting edits from a draft |
| **Clone** | Download a copy of a repo | Photocopying the cabinet |
| **Push** | Upload your commits to remote (GitHub) | Syncing to cloud |
| **Pull** | Download latest changes from remote | Getting latest from cloud |

### Essential Git Commands

```bash
# SETUP (one-time)
git config --global user.name "Your Name"
git config --global user.email "you@email.com"

# START A PROJECT
git init                          # Initialize Git in current folder
git clone <url>                   # Download existing repo from GitHub

# DAILY WORKFLOW
git status                        # See what's changed
git add .                         # Stage all changes for commit
git add file.py                   # Stage specific file
git commit -m "Add churn model"   # Save snapshot with message
git push                          # Upload to GitHub
git pull                          # Download latest from GitHub

# BRANCHING
git branch feature/new-model      # Create a branch
git checkout feature/new-model    # Switch to that branch
git checkout -b feature/new-model # Create AND switch (shortcut)
git merge feature/new-model       # Merge branch into current branch

# HISTORY
git log                           # See commit history
git log --oneline                 # Compact history
git diff                          # See uncommitted changes
```

### How a Commit Works (Visual)

```
Before commit:
  Working Directory → [git add] → Staging Area → [git commit] → Repository

Example:
  Edit train.py → git add train.py → git commit -m "Fix learning rate"
                                              ↓
                                    Saved permanently with:
                                    - Who (your name)
                                    - When (timestamp)
                                    - What (the diff)
                                    - Why (commit message)
```

---

## 3. GitHub: Collaboration Platform

### What Is GitHub?

GitHub = Git (version control) + cloud hosting + collaboration tools.

It's where your code lives online so teams can work together.

### GitHub vs Git

| Git | GitHub |
|-----|--------|
| Tool on your computer | Website/service |
| Tracks changes locally | Hosts repos in the cloud |
| Works offline | Requires internet |
| Free, open source | Free for public repos, paid for private features |

### Key GitHub Features

| Feature | What It Does |
|---------|-------------|
| **Repositories** | Host your project code |
| **Pull Requests** | Propose and review changes |
| **Issues** | Track bugs and feature requests |
| **Actions** | Automated CI/CD workflows |
| **Projects** | Kanban boards for task management |
| **Releases** | Version your software |
| **Wiki** | Documentation |
| **Secrets** | Store API keys securely |

### Creating a Repo on GitHub

1. Go to github.com → "New Repository"
2. Name it (e.g., `sagemaker-mlops-project`)
3. Choose Public or Private
4. Add .gitignore (Python template)
5. Click "Create"

```bash
# Connect your local project to GitHub
git remote add origin https://github.com/YOUR_USERNAME/sagemaker-mlops-project.git
git branch -M main
git push -u origin main
```

---

## 4. Branching Strategy

### Why Branches?

Branches let you work on features without breaking the main code.

```
main (production-ready code)
  │
  ├── feature/add-churn-model     (your work in progress)
  │     ├── commit: "Add preprocessing"
  │     ├── commit: "Add training script"
  │     └── commit: "Add evaluation"
  │
  ├── feature/update-dashboard    (teammate's work)
  │
  └── hotfix/fix-endpoint-bug     (urgent fix)
```

### Git Flow (Common Strategy)

```
main          ─────────────────────────────────────────── (production)
                    ↑ merge                    ↑ merge
develop       ──────┼──────────────────────────┼──────── (integration)
                    ↑ merge          ↑ merge
feature/xyz   ──────┘               │
feature/abc   ──────────────────────┘
```

### Simplified Strategy (Recommended for ML)

```
main              ─── (always deployable) ───
                       ↑ Pull Request
feature/branch    ─── (your work) ───
```

Rules:
1. Never commit directly to `main`
2. Create a branch for every change
3. Open a Pull Request to merge
4. Require at least 1 review before merging
5. CI must pass before merging

---

## 5. Pull Requests

### What Is a Pull Request (PR)?

A PR says: "I've made changes on my branch. Please review and merge them into main."

### PR Workflow

```
1. Create branch:     git checkout -b feature/improve-model
2. Make changes:      (edit files, commit)
3. Push branch:       git push -u origin feature/improve-model
4. Open PR:           On GitHub, click "New Pull Request"
5. Review:            Teammates comment, suggest changes
6. CI runs:           Automated tests check your code
7. Approve:           Reviewer approves
8. Merge:             Click "Merge Pull Request"
9. Delete branch:     Clean up
```

### What a Good PR Looks Like

```markdown
## Title: Add XGBoost churn prediction model

## Description
- Added preprocessing script with label encoding
- Implemented XGBoost training with early stopping
- Added evaluation metrics (AUC, F1, Precision, Recall)
- Model achieves 0.87 AUC on test set

## Changes
- `scripts/preprocess.py` - New file
- `scripts/train.py` - New file
- `scripts/evaluate.py` - New file
- `config/pipeline_params.json` - Updated hyperparameters

## Testing
- [x] Unit tests pass (`pytest tests/`)
- [x] Pipeline runs successfully on dev
- [x] Model metrics meet threshold (AUC > 0.75)

## Screenshots
(evaluation metrics, confusion matrix, etc.)
```

---

## 6. What Is CI/CD?

### The Full Picture

```
CI/CD = Continuous Integration + Continuous Delivery + Continuous Deployment

┌──────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Developer → Push Code → CI (Test) → CD (Deploy) → Production    │
│                              │              │                      │
│                         Automated       Automated                  │
│                         Quality         Delivery                   │
│                         Checks          to Users                   │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### CI vs CD vs CD

| Term | Full Name | What It Does |
|------|-----------|-------------|
| **CI** | Continuous Integration | Automatically test code on every push |
| **CD** | Continuous Delivery | Automatically prepare code for release (manual deploy) |
| **CD** | Continuous Deployment | Automatically deploy to production (no manual step) |

### Why It Matters

| Without CI/CD | With CI/CD |
|---------------|------------|
| "Works on my machine" | Works everywhere (tested in clean environment) |
| Bugs found in production | Bugs caught before merge |
| Manual, error-prone deploys | Automated, consistent deploys |
| Fear of deploying | Deploy with confidence |
| Weekly/monthly releases | Deploy multiple times per day |

---

## 7. CI: Continuous Integration

### What Happens in CI?

Every time you push code, these checks run automatically:

```
Push Code
    ↓
┌─────────────────────────────────┐
│  CI Pipeline                     │
├─────────────────────────────────┤
│  1. Checkout code                │
│  2. Install dependencies         │
│  3. Run linter (code style)      │
│  4. Run unit tests               │
│  5. Run integration tests        │
│  6. Check code coverage          │
│  7. Security scan                │
│  8. Build artifacts              │
│  9. Report results               │
└─────────────────────────────────┘
    ↓
✅ All pass → Ready to merge
❌ Any fail → Block merge, notify developer
```

### CI Checks for ML Projects

| Check | What It Validates | Tool |
|-------|------------------|------|
| Lint | Code style, formatting | ruff, flake8 |
| Type check | Type correctness | mypy |
| Unit tests | Individual functions work | pytest |
| Data validation | Schema, ranges, nulls | great_expectations |
| Model tests | Model loads, predicts, reasonable output | pytest |
| Pipeline test | Pipeline definition is valid | SageMaker SDK |
| Security | No secrets in code, safe dependencies | bandit, safety |

---

## 8. CD: Continuous Delivery/Deployment

### Continuous Delivery (with manual gate)

```
CI passes → Build artifact → Deploy to staging → Manual approval → Deploy to production
```

### Continuous Deployment (fully automated)

```
CI passes → Build artifact → Deploy to staging → Auto-tests pass → Deploy to production
```

### For ML Projects, CD Means:

```
Code change merged to main
    ↓
CI tests pass
    ↓
SageMaker Pipeline executes (train new model)
    ↓
Model evaluation passes quality gate
    ↓
Model registered in Model Registry
    ↓
[Manual approval OR auto-approval]
    ↓
Model deployed to endpoint
    ↓
Canary/shadow deployment (test with real traffic)
    ↓
Full rollout
```

---

## 9. GitHub Actions (Hands-On)

### What Are GitHub Actions?

GitHub's built-in CI/CD system. You define workflows in YAML files.

### File Location

```
your-repo/
└── .github/
    └── workflows/
        ├── ci.yml          # Runs on every push
        ├── deploy.yml      # Runs on merge to main
        └── scheduled.yml   # Runs on schedule
```

### Basic CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI

# When to run
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

# What to do
jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      # 1. Get the code
      - name: Checkout code
        uses: actions/checkout@v4

      # 2. Set up Python
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      # 3. Install uv
      - name: Install uv
        uses: astral-sh/setup-uv@v3

      # 4. Install dependencies
      - name: Install dependencies
        run: uv sync --all-extras

      # 5. Lint
      - name: Lint with ruff
        run: uv run ruff check .

      # 6. Run tests
      - name: Run tests
        run: uv run pytest tests/ -v --tb=short

      # 7. Check types (optional)
      - name: Type check
        run: uv run mypy scripts/ --ignore-missing-imports
```

### Key Concepts

| Concept | What It Is |
|---------|-----------|
| **Workflow** | A YAML file defining automation |
| **Trigger (on)** | What starts the workflow (push, PR, schedule, manual) |
| **Job** | A set of steps that run on one machine |
| **Step** | A single command or action |
| **Action** | A reusable step (e.g., `actions/checkout@v4`) |
| **Runner** | The machine that executes the job |
| **Secret** | Encrypted variable (API keys, credentials) |
| **Artifact** | File produced by a job (test reports, builds) |

### Using Secrets (for AWS Credentials)

1. Go to repo → Settings → Secrets → Actions
2. Add: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`

```yaml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: ${{ secrets.AWS_REGION }}
```

### Workflow Triggers

```yaml
on:
  # On every push to these branches
  push:
    branches: [main, develop]

  # On pull requests targeting main
  pull_request:
    branches: [main]

  # On schedule (cron)
  schedule:
    - cron: '0 2 * * 1'  # Every Monday at 2 AM

  # Manual trigger (button in GitHub UI)
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deploy to which environment?'
        required: true
        default: 'staging'
```

---

## 10. CI/CD for ML Projects

### ML CI/CD Is Different From Software CI/CD

| Software CI/CD | ML CI/CD |
|---------------|----------|
| Test code logic | Test code + data + model |
| Deploy application | Deploy model + application |
| Rollback = previous code | Rollback = previous model version |
| Triggered by code change | Triggered by code change OR data change OR drift |

### ML CI/CD Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'scripts/**'
      - 'pipelines/**'
      - 'config/**'

jobs:
  # Job 1: Test
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras
      - run: uv run pytest tests/ -v
      - run: uv run ruff check .

  # Job 2: Deploy Pipeline (only after tests pass)
  deploy-pipeline:
    needs: test  # Wait for test job to pass
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # Only on main branch

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: eu-west-1

      - name: Install dependencies
        run: uv sync

      - name: Update SageMaker Pipeline
        run: uv run python pipelines/pipeline.py

      - name: Notify team
        if: success()
        run: echo "Pipeline deployed successfully"
```

### What Gets Tested in ML CI

```python
# tests/test_pipeline.py
def test_preprocessing_script_runs():
    """Verify preprocessing doesn't crash on sample data."""
    # Run with small sample
    ...

def test_model_can_predict():
    """Verify model produces valid predictions."""
    model = load_model("tests/fixtures/sample_model.json")
    prediction = model.predict(sample_input)
    assert 0 <= prediction <= 1

def test_pipeline_definition_valid():
    """Verify pipeline YAML/definition is syntactically correct."""
    from pipelines.pipeline import create_pipeline
    pipeline = create_pipeline()
    definition = json.loads(pipeline.definition())
    assert 'Steps' in definition

def test_feature_schema():
    """Verify input data matches expected schema."""
    df = pd.read_csv("tests/fixtures/sample_data.csv")
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df['age'].between(0, 120).all()
```

---

## 11. AWS CI/CD Services

### AWS CodePipeline

```
Source (GitHub) → Build (CodeBuild) → Test → Deploy (SageMaker)
```

### AWS CodeBuild

Runs your build/test commands in a managed container.

```yaml
# buildspec.yml (CodeBuild config)
version: 0.2

phases:
  install:
    commands:
      - pip install uv
      - uv sync --all-extras
  build:
    commands:
      - uv run pytest tests/
      - uv run python pipelines/pipeline.py

artifacts:
  files:
    - '**/*'
```

### Comparison: GitHub Actions vs AWS CodePipeline

| Feature | GitHub Actions | AWS CodePipeline |
|---------|---------------|-----------------|
| Where it lives | GitHub | AWS Console |
| Config format | YAML in repo | Console + buildspec.yml |
| AWS integration | Via actions/credentials | Native |
| Cost | Free (2000 min/month) | Pay per pipeline + build minutes |
| Community actions | Huge marketplace | Limited |
| Ease of use | Easier | More complex |

**Recommendation**: Use GitHub Actions unless your org mandates AWS-native CI/CD.

---

## 12. Complete Example: ML Project CI/CD

### The Full Flow

```
Developer pushes code to feature branch
    ↓
GitHub Actions CI runs (lint + test)
    ↓ (passes)
Developer opens Pull Request
    ↓
Teammate reviews code
    ↓ (approved)
Merge to main
    ↓
GitHub Actions CD triggers
    ↓
SageMaker Pipeline updated and started
    ↓
Pipeline: Preprocess → Train → Evaluate
    ↓
If AUC > 0.75: Model registered (PendingApproval)
    ↓
SNS notification to team
    ↓
Data scientist approves model in console
    ↓
EventBridge detects approval → triggers deployment Lambda
    ↓
Lambda updates SageMaker Endpoint with new model
    ↓
Model Monitor watches for drift
    ↓
If drift detected → triggers retraining pipeline
```

---

## 13. Best Practices

### Git Best Practices

1. **Commit often, push daily** — small commits are easier to review and revert
2. **Write meaningful commit messages** — "Fix bug" is bad; "Fix null pointer in feature encoding for categorical columns" is good
3. **Never commit secrets** — use .gitignore and GitHub Secrets
4. **Use .gitignore** — don't commit data files, .venv, __pycache__
5. **One PR per feature** — keep changes focused and reviewable

### CI/CD Best Practices

1. **Fast CI** — keep it under 10 minutes (developers won't wait longer)
2. **Fail fast** — run quick checks (lint) before slow ones (integration tests)
3. **Reproducible** — pin dependency versions, use lockfiles
4. **Branch protection** — require CI to pass before merge
5. **Secrets management** — never hardcode credentials

### ML-Specific Best Practices

1. **Version everything** — code (Git), data (DVC), models (Registry)
2. **Test data quality** — not just code
3. **Automate retraining** — don't rely on manual triggers
4. **Shadow deployment** — test new models with real traffic before switching
5. **Monitor continuously** — models degrade over time

---

### Quick Reference: Git Commands You'll Use Daily

```bash
# Morning: get latest
git pull

# Start new work
git checkout -b feature/my-change

# During the day: save progress
git add .
git commit -m "Descriptive message"

# End of day: push to GitHub
git push -u origin feature/my-change

# When done: create PR on GitHub website

# After PR merged: clean up
git checkout main
git pull
git branch -d feature/my-change
```

---

*End of CI/CD & GitHub Guide*
