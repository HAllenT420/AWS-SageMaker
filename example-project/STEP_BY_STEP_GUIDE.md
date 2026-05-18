# Step-by-Step Guide: Build This Project From Scratch

> Imagine you just opened your laptop and want to run this project.
> Follow EVERY step below. Don't skip anything. Don't jump ahead.
> Copy-paste the commands exactly as shown.

---

## BEFORE YOU START: What Are We Building?

We are building a system that **predicts if a customer will leave** (churn).

Think of it like this:
- A telecom company has 5000 customers
- Some customers will cancel their subscription next month
- We want to predict WHO will cancel, so we can offer them a discount to stay

The system will:
1. Take customer data (how long they've been with us, how much they pay, etc.)
2. Feed it to a machine learning model
3. The model says: "This customer has 82% chance of leaving"
4. The company can then act on it

---

## PART A: ONE-TIME SETUP (Do This Only Once)

---

### Step A1: Make Sure Python Is Installed

Open your terminal in VS Code (press `Ctrl + backtick` — the key above Tab).

Type this and press Enter:

```
python --version
```

You should see something like `Python 3.11.5` or `Python 3.9.x`.

**If you get an error**: Go to https://www.python.org/downloads/ and install Python. Check the box "Add Python to PATH" during installation.

---

### Step A2: Install uv (Our Package Manager)

In the same terminal, paste this ENTIRE line and press Enter:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Wait for it to finish. You should see "uv installed successfully" or similar.

**To verify it worked**, type:

```
uv --version
```

You should see a version number like `uv 0.6.x`.

---

### Step A3: Install AWS CLI

Paste this in terminal:

```
winget install Amazon.AWSCLI
```

If `winget` doesn't work, go to https://aws.amazon.com/cli/ and download the installer.

**To verify**, close and reopen terminal, then type:

```
aws --version
```

You should see `aws-cli/2.x.x`.

---

### Step A4: Configure AWS Credentials

You need an AWS account with access keys. If you don't have them:
1. Go to https://aws.amazon.com and sign in
2. Click your name (top right) → Security Credentials
3. Create Access Key → Download the CSV

Now in terminal, type:

```
aws configure
```

It will ask you 4 things. Type each answer and press Enter:

```
AWS Access Key ID [None]: PASTE_YOUR_ACCESS_KEY_HERE
AWS Secret Access Key [None]: PASTE_YOUR_SECRET_KEY_HERE
Default region name [None]: eu-west-1
Default output format [None]: json
```

**To verify**, type:

```
aws sts get-caller-identity
```

You should see your account ID. If you get an error, your keys are wrong.

---

### Step A5: Find Your SageMaker Role ARN

You need a "role" that gives SageMaker permission to do things.

**Option 1: If your company already has one:**
Ask your team: "What's our SageMaker execution role ARN?"
It looks like: `arn:aws:iam::123456789012:role/SageMakerRole`

**Option 2: Create one yourself:**

1. Go to AWS Console → IAM → Roles → Create Role
2. Trusted entity: AWS Service → SageMaker
3. Permissions: Search and check `AmazonSageMakerFullAccess`
4. Also check `AmazonS3FullAccess`
5. Role name: `SageMakerExecutionRole`
6. Click Create
7. Click on the role you just created
8. Copy the "ARN" at the top (starts with `arn:aws:iam::`)

**Write this ARN down. You'll need it in the next step.**

---

### Step A6: Set Up The Project

In terminal, navigate to the project folder:

```
cd "C:\Users\Allen.Harry\OneDrive - Entain Group\Desktop\AWS SageMaker\example-project\part2-mlops"
```

Install all dependencies:

```
uv sync --all-extras
```

Wait for it to finish (should take 30-60 seconds).

---

### Step A7: Put Your Role ARN in the Config

Open this file in VS Code:
```
example-project/part2-mlops/config/params.json
```

Find this line:
```
"role_arn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE",
```

Replace it with YOUR actual ARN from Step A5. For example:
```
"role_arn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
```

**Save the file** (Ctrl+S).

Also change the region if yours is different:
```
"region": "eu-west-1",
```

---

### Step A8: Register Jupyter Kernel (For Notebooks)

In terminal (still in the part2-mlops folder):

```
uv run python -m ipykernel install --user --name churn-project --display-name "Churn Project"
```

---

## ✅ SETUP COMPLETE!

You only do the above ONCE. Now let's actually run things.

---
---

## PART B: RUN THE NOTEBOOK (Interactive Experiment)

This is where you see everything work step by step, with outputs.

---

### Step B1: Open the Notebook

In VS Code, open this file:
```
example-project/part1-notebook/churn_experiment.ipynb
```

---

### Step B2: Select the Kernel

Look at the top-right corner of the notebook. You'll see something like "Select Kernel" or "Python 3".

Click it → Select "Churn Project" (the one we created in Step A8).

If you don't see it, select "Python Environments" → pick any Python 3.9+ environment.

---

### Step B3: Update the Role ARN in the Notebook

In the SECOND code cell, find this line:
```python
ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE'
```

Replace it with your actual ARN (same one from Step A5).

---

### Step B4: Run Each Cell One by One

Click on the first code cell, then press `Shift+Enter` to run it.

Wait for it to finish (you'll see a number appear on the left like `[1]`).

Then move to the next cell and press `Shift+Enter` again.

**Keep doing this for every cell, in order, top to bottom.**

Here's what each section does:

| Section | What Happens | Time |
|---------|-------------|------|
| 1. Setup | Connects to AWS | 2 seconds |
| 2. Data Generation | Creates fake customer data | 1 second |
| 3. EDA | Shows charts about the data | 2 seconds |
| 4. Preprocessing | Cleans and prepares data | 2 seconds |
| 5. Upload to S3 | Sends data to AWS cloud storage | 10 seconds |
| 6. Train on SageMaker | **Trains the model on AWS** | **5-8 minutes** |
| 7. Evaluate | Tests how good the model is | 3-5 minutes |
| 8. Deploy | Makes the model available for predictions | 5-8 minutes |
| 9. Test Predictions | Asks the model to predict | 2 seconds |
| 10. Cleanup | Deletes the endpoint (saves money) | 10 seconds |

**⚠️ IMPORTANT**: Steps 6, 7, and 8 take several minutes because AWS is spinning up machines for you. Just wait. Don't click anything else.

---

### Step B5: Check Your Results

After Step 7 (Evaluate), you should see something like:

```
AUC-ROC: 0.8534
```

If AUC is above 0.75, your model is good! ✅

---

### Step B6: DELETE THE ENDPOINT (Save Money!)

After you're done testing, go to the LAST cell (Section 10) and **uncomment** the line:

```python
predictor.delete_endpoint()
```

(Remove the `#` at the start)

Run that cell. This stops AWS from charging you.

**If you forget this, you'll be charged ~$0.14/hour until you delete it.**

---

## ✅ NOTEBOOK EXPERIMENT COMPLETE!

You just:
- Trained a model on AWS SageMaker ✓
- Evaluated it (AUC > 0.75) ✓
- Deployed it as a live API ✓
- Made real predictions ✓
- Cleaned up ✓

---
---

## PART C: RUN THE MODULAR CODE (Production Version)

This does the SAME thing as the notebook, but as proper production code.

---

### Step C1: Open Terminal in the Right Folder

In VS Code terminal:

```
cd "C:\Users\Allen.Harry\OneDrive - Entain Group\Desktop\AWS SageMaker\example-project\part2-mlops"
```

---

### Step C2: Run Tests First (Make Sure Code Works)

```
uv run pytest tests/ -v
```

You should see all tests PASS (green). If any fail, something is wrong with setup.

---

### Step C3: Run the Full Pipeline (Without Deployment)

```
uv run python run_pipeline.py
```

This will:
1. Generate data and upload to S3
2. Run preprocessing on SageMaker
3. Train the model on SageMaker
4. Evaluate the model
5. Check quality gates (AUC > 0.75?)
6. Register the model in Model Registry

**This takes about 20-30 minutes total.** You'll see logs in the terminal showing progress.

---

### Step C4: Run With Deployment (Optional)

If you also want to deploy an endpoint:

```
uv run python run_pipeline.py --deploy
```

**Remember to delete the endpoint when done** to avoid charges.

---

### Step C5: Run the SageMaker Pipeline (Fully Automated)

This creates a pipeline in AWS that can run on schedule:

```
uv run python pipeline/pipeline.py
```

After this runs, go to:
- AWS Console → SageMaker → Pipelines
- You'll see your pipeline with a visual diagram
- You can click "Start" to run it anytime

---

## ✅ PRODUCTION CODE COMPLETE!

---
---

## PART D: UNDERSTANDING WHAT EACH FILE DOES

If someone asks "what does this file do?", here's the answer:

```
example-project/
│
├── part1-notebook/
│   └── churn_experiment.ipynb    ← "The playground" — try things here
│
├── part2-mlops/
│   ├── config/
│   │   ├── params.json           ← "The settings" — change region, role, hyperparameters
│   │   └── config.py             ← "The settings reader" — loads params.json
│   │
│   ├── src/                      ← "The brain" — each file does ONE job
│   │   ├── data_ingestion.py     ← Gets data from source (DB, API, or generates it)
│   │   ├── preprocessing.py      ← Tells SageMaker to clean the data
│   │   ├── training.py           ← Tells SageMaker to train the model
│   │   ├── evaluation.py         ← Checks if the model is good enough
│   │   ├── registry.py           ← Saves the model with a version number
│   │   ├── deployment.py         ← Makes the model available as an API
│   │   └── monitoring.py         ← Watches the model for problems
│   │
│   ├── scripts/                  ← "The workers" — these run ON SageMaker (not your laptop)
│   │   ├── preprocess.py         ← Actual preprocessing code (runs on AWS machine)
│   │   └── evaluate.py           ← Actual evaluation code (runs on AWS machine)
│   │
│   ├── pipeline/
│   │   └── pipeline.py           ← "The assembly line" — automates everything
│   │
│   ├── tests/                    ← "The safety net" — makes sure code works
│   │   └── test_preprocessing.py
│   │
│   ├── run_pipeline.py           ← "The big red button" — runs everything in order
│   ├── pyproject.toml            ← "The shopping list" — what packages to install (for uv)
│   └── requirements.txt          ← Same shopping list (for pip)
```

---

## COMMON PROBLEMS & FIXES

| What You See | What's Wrong | How to Fix |
|---|---|---|
| `uv: command not found` | uv not installed | Redo Step A2, then close and reopen terminal |
| `aws: command not found` | AWS CLI not installed | Redo Step A3, then close and reopen terminal |
| `No module named 'sagemaker'` | Dependencies not installed | Run `uv sync --all-extras` in part2-mlops folder |
| `AccessDeniedException` | Wrong AWS credentials | Run `aws configure` again with correct keys |
| `Could not assume role` | Wrong role ARN in params.json | Check Step A5, make sure ARN is correct |
| `ResourceLimitExceeded` | AWS account limit | Go to AWS Console → Service Quotas → request increase |
| Training job stuck | Normal — it takes time | Wait 5-8 minutes. Check CloudWatch Logs if >15 min |
| `Endpoint already exists` | You ran deploy twice | Delete old endpoint first, or change the name in params.json |
| Charged money unexpectedly | Forgot to delete endpoint | Go to AWS Console → SageMaker → Endpoints → Delete |

---

## COST GUIDE (So You Don't Get Surprised)

| What | Cost | When You Pay |
|------|------|-------------|
| S3 storage | ~$0.02/month | Always (tiny amount) |
| Processing job | ~$0.05 per run | Only while running (5-10 min) |
| Training job | ~$0.10 per run | Only while running (5-10 min) |
| Endpoint (ml.m5.large) | ~$0.14/hour | **Every hour it's running!** |
| Batch Transform | ~$0.05 per run | Only while running |

**The expensive thing is the ENDPOINT.** Always delete it when done testing.

Total cost to run this project once: **~$1-2** (if you delete the endpoint after testing).

---

## WHAT TO DO NEXT

After you've run everything successfully:

1. **Change the data**: Edit `src/data_ingestion.py` to use real data instead of synthetic
2. **Tune the model**: Change hyperparameters in `config/params.json`
3. **Add features**: Edit `scripts/preprocess.py` to create better features
4. **Schedule retraining**: Set up EventBridge to run the pipeline weekly
5. **Add monitoring**: Call `setup_model_monitor()` after deployment

---

*That's it. You've built a production ML system. 🎉*
