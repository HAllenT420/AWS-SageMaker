# Data Science Algorithms: Visual Guide

> How every algorithm works (with pictures), their assumptions,
> all hyperparameters, and every evaluation metric explained visually.

---

## Table of Contents

1. [Linear Regression](#1-linear-regression)
2. [Logistic Regression](#2-logistic-regression)
3. [Decision Trees](#3-decision-trees)
4. [Random Forest](#4-random-forest)
5. [Gradient Boosting (XGBoost / LightGBM)](#5-gradient-boosting)
6. [Support Vector Machines (SVM)](#6-svm)
7. [K-Nearest Neighbors (KNN)](#7-knn)
8. [Naive Bayes](#8-naive-bayes)
9. [K-Means Clustering](#9-k-means)
10. [Neural Networks](#10-neural-networks)
11. [All Hyperparameters Reference](#11-hyperparameters)
12. [All Evaluation Metrics (Visual)](#12-evaluation-metrics)
13. [ROC Curve & AUC Explained](#13-roc-curve)
14. [Confusion Matrix Deep Dive](#14-confusion-matrix)
15. [Regression Metrics Visualized](#15-regression-metrics)
16. [How to Pick the Right Metric](#16-picking-metrics)

---

## 1. Linear Regression

### How It Works (Visual)

```
    y (price)
    │
    │                    ╱ ← This line is the model
    │               ╱  •
    │          ╱ •    •
    │     ╱•      •
    │ ╱  •   •
    │╱ •
    └──────────────────── x (size of house)

The model finds the BEST STRAIGHT LINE through your data points.
Prediction = slope × input + intercept
           = 5000 × (square feet) + 20000
```

### The Math (Simple Version)

```
y = mx + b

y = what you're predicting (house price)
x = input feature (house size)
m = slope (how much y changes when x increases by 1)
b = intercept (y value when x = 0)

With multiple features:
y = w1×x1 + w2×x2 + w3×x3 + ... + b
```

### Assumptions

| Assumption | What It Means | What Happens If Violated |
|------------|--------------|------------------------|
| **Linearity** | Relationship is a straight line | Model will be wrong (use polynomial or tree) |
| **Independence** | Each data point is independent | Standard errors are wrong |
| **Homoscedasticity** | Error spread is constant | Predictions unreliable at extremes |
| **Normality of errors** | Errors follow bell curve | Confidence intervals are wrong |
| **No multicollinearity** | Features aren't highly correlated | Coefficients become unstable |

### How to Check Assumptions

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_train)
residuals = y_train - predictions

# Plot 1: Residuals vs Predicted (check homoscedasticity)
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Should look like random scatter around 0')
plt.show()

# Plot 2: Q-Q plot (check normality)
from scipy import stats
stats.probplot(residuals, plot=plt)
plt.title('Points should follow the diagonal line')
plt.show()
```

### Hyperparameters

| Parameter | What It Does | Values |
|-----------|-------------|--------|
| `fit_intercept` | Include bias term? | True/False (default: True) |
| `normalize` | Scale features first? | True/False (default: False) |

### Variants

| Variant | What It Adds | When to Use |
|---------|-------------|-------------|
| **Ridge (L2)** | Penalty on large coefficients | Many features, multicollinearity |
| **Lasso (L1)** | Penalty that zeros out features | Feature selection |
| **ElasticNet** | Both L1 + L2 | Best of both worlds |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge: alpha controls regularization strength
ridge = Ridge(alpha=1.0)

# Lasso: zeros out unimportant features
lasso = Lasso(alpha=0.1)

# ElasticNet: mix of both
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

---

## 2. Logistic Regression

### How It Works (Visual)

```
    P(churn)
    1.0 │                          ●●●●●●●●●
        │                      ●●●
        │                   ●●
    0.5 │- - - - - - - - -●- - - - - - - - ← Decision boundary
        │               ●●
        │            ●●●
    0.0 │●●●●●●●●●●
        └──────────────────────────────────── monthly_charges
                    $50              $100

NOT a straight line — it's an S-curve (sigmoid).
Output is always between 0 and 1 (probability).
```

### The Math

```
Step 1: Linear combination (same as linear regression)
        z = w1×x1 + w2×x2 + ... + b

Step 2: Squeeze through sigmoid function
        P(y=1) = 1 / (1 + e^(-z))

This converts any number into a probability between 0 and 1.

z = -5  →  P = 0.007  (very unlikely)
z =  0  →  P = 0.500  (50/50)
z = +5  →  P = 0.993  (very likely)
```

### Assumptions

| Assumption | What It Means |
|------------|--------------|
| **Binary outcome** | Target is 0 or 1 |
| **Independence** | Observations are independent |
| **No multicollinearity** | Features aren't highly correlated |
| **Linear in log-odds** | log(p/(1-p)) is linear with features |
| **Large sample size** | Need ~10 events per feature |

### Hyperparameters

| Parameter | What It Does | Values | Default |
|-----------|-------------|--------|---------|
| `C` | Regularization (inverse) | 0.001-1000 | 1.0 |
| `penalty` | Type of regularization | 'l1', 'l2', 'elasticnet' | 'l2' |
| `solver` | Optimization algorithm | 'lbfgs', 'liblinear', 'saga' | 'lbfgs' |
| `max_iter` | Max iterations | 100-10000 | 100 |
| `class_weight` | Handle imbalance | None, 'balanced' | None |

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,                    # Higher = less regularization
    penalty='l2',             # L2 regularization
    class_weight='balanced',  # Handle imbalanced classes
    max_iter=1000
)
```

---

## 3. Decision Trees

### How It Works (Visual)

```
                    ┌─────────────────────┐
                    │ Is tenure < 12 mo?  │
                    └──────────┬──────────┘
                       Yes ╱         ╲ No
                          ╱           ╲
            ┌────────────────┐   ┌────────────────┐
            │ Monthly > $70? │   │ Contract type? │
            └───────┬────────┘   └───────┬────────┘
             Yes ╱    ╲ No        M2M ╱    ╲ Yearly
                ╱      ╲             ╱      ╲
          ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
          │ CHURN  │ │ MAYBE  │ │ CHURN  │ │ STAY   │
          │  85%   │ │  45%   │ │  60%   │ │  10%   │
          └────────┘ └────────┘ └────────┘ └────────┘

The tree asks yes/no questions, splitting data at each node.
It picks the question that best separates the classes.
```

### How Splits Are Chosen

```
Before split: 100 customers (40 churn, 60 stay)
                    Impurity = 0.48 (mixed)

Split on "tenure < 12":
  Left (tenure < 12):  30 customers (25 churn, 5 stay)  → Impurity = 0.28
  Right (tenure >= 12): 70 customers (15 churn, 55 stay) → Impurity = 0.33

This split REDUCES impurity the most → tree picks it!
```

### Assumptions

| Assumption | Reality |
|------------|---------|
| **None!** | Decision trees have NO statistical assumptions |
| | They handle non-linear relationships |
| | They handle mixed data types |
| | They handle missing values (some implementations) |

**This is why trees are so popular — they just work.**

### Hyperparameters

| Parameter | What It Does | Values | Effect |
|-----------|-------------|--------|--------|
| `max_depth` | How deep the tree can grow | 1-30 | Deeper = more complex, risk overfit |
| `min_samples_split` | Min samples to split a node | 2-100 | Higher = more conservative |
| `min_samples_leaf` | Min samples in a leaf | 1-50 | Higher = smoother predictions |
| `max_features` | Features considered per split | 'sqrt', 'log2', int | Lower = more randomness |
| `criterion` | How to measure split quality | 'gini', 'entropy' | Usually doesn't matter much |

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,              # Don't grow too deep
    min_samples_split=20,     # Need at least 20 samples to split
    min_samples_leaf=10,      # Each leaf has at least 10 samples
    class_weight='balanced'
)
```

### Visualize a Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, class_names=['Stay', 'Churn'],
          filled=True, rounded=True, max_depth=3)
plt.title('Decision Tree (first 3 levels)')
plt.show()
```

---

## 4. Random Forest

### How It Works (Visual)

```
Training Data
     │
     ├──── Random Sample 1 ──── Tree 1 ──── Prediction: CHURN
     │     (different rows      (different
     │      & columns)           structure)
     │
     ├──── Random Sample 2 ──── Tree 2 ──── Prediction: STAY
     │
     ├──── Random Sample 3 ──── Tree 3 ──── Prediction: CHURN
     │
     ├──── Random Sample 4 ──── Tree 4 ──── Prediction: CHURN
     │
     └──── Random Sample 5 ──── Tree 5 ──── Prediction: STAY

                                              VOTE: 3 CHURN vs 2 STAY
                                              Final: CHURN (majority wins)

Each tree sees DIFFERENT data (random rows + random columns).
Each tree makes its own prediction.
Final answer = majority vote of all trees.
```

### Why It's Better Than One Tree

```
One tree:     Overfits (memorizes noise)
100 trees:    Each overfits differently
              Averaging them cancels out the noise
              → More robust predictions
```

### Assumptions

Same as Decision Trees — **NONE**. That's why Random Forest is the go-to "just works" algorithm.

### Hyperparameters

| Parameter | What It Does | Values | Default |
|-----------|-------------|--------|---------|
| `n_estimators` | Number of trees | 50-1000 | 100 |
| `max_depth` | Max depth per tree | 3-30, None | None (unlimited) |
| `min_samples_split` | Min samples to split | 2-100 | 2 |
| `min_samples_leaf` | Min samples in leaf | 1-50 | 1 |
| `max_features` | Features per split | 'sqrt', 'log2', float | 'sqrt' |
| `bootstrap` | Sample with replacement? | True/False | True |
| `class_weight` | Handle imbalance | None, 'balanced' | None |
| `n_jobs` | Parallel trees | -1 (all CPUs) | None |

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,         # 200 trees
    max_depth=10,             # Each tree max 10 levels
    min_samples_leaf=5,       # At least 5 samples per leaf
    max_features='sqrt',      # sqrt(n_features) per split
    class_weight='balanced',
    n_jobs=-1,                # Use all CPU cores
    random_state=42
)
```

---

## 5. Gradient Boosting (XGBoost / LightGBM)

### How It Works (Visual)

```
Unlike Random Forest (parallel trees), Gradient Boosting builds trees SEQUENTIALLY.
Each tree fixes the mistakes of the previous one.

Step 1: Make initial prediction (e.g., average)
        Prediction: 0.4 for everyone
        Error: some customers should be 0.9, some 0.1

Step 2: Tree 1 learns the ERRORS
        "Customers with tenure < 6 have error +0.3"
        New prediction: 0.4 + 0.2×(Tree 1 output)

Step 3: Tree 2 learns the REMAINING errors
        "Customers with high charges still have error +0.2"
        New prediction: 0.4 + 0.2×(Tree 1) + 0.2×(Tree 2)

... 100 trees later ...

Final: Very accurate predictions!


VISUAL COMPARISON:

Random Forest:          Gradient Boosting:
Tree1 ─┐               Tree1
Tree2 ─┤               ↓ (learn errors)
Tree3 ─┼── VOTE ──▶    Tree2
Tree4 ─┤               ↓ (learn remaining errors)
Tree5 ─┘               Tree3
                        ↓
(parallel, independent) (sequential, each fixes previous)
```

### XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed | Fast | Fastest | Slower |
| Accuracy | Excellent | Excellent | Excellent |
| Categorical handling | Manual encoding needed | Built-in | Best built-in |
| Missing values | Handles automatically | Handles automatically | Handles automatically |
| Memory | Medium | Low | High |
| Best for | General use | Large datasets | Many categoricals |

### Assumptions

| Assumption | Reality |
|------------|---------|
| **None!** | Same as trees — no assumptions |
| | But: sensitive to hyperparameters |
| | And: can overfit if not regularized |

### Hyperparameters (XGBoost)

| Parameter | What It Does | Range | Start With |
|-----------|-------------|-------|-----------|
| `n_estimators` / `num_round` | Number of trees | 50-5000 | 200 |
| `max_depth` | Tree depth | 3-12 | 6 |
| `learning_rate` / `eta` | Contribution of each tree | 0.01-0.3 | 0.1 |
| `subsample` | % rows per tree | 0.5-1.0 | 0.8 |
| `colsample_bytree` | % features per tree | 0.5-1.0 | 0.8 |
| `min_child_weight` | Min samples in leaf | 1-20 | 5 |
| `gamma` | Min gain to split | 0-5 | 0 |
| `reg_alpha` | L1 regularization | 0-10 | 0 |
| `reg_lambda` | L2 regularization | 0-10 | 1 |
| `scale_pos_weight` | Class imbalance | 1-100 | neg_count/pos_count |

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=3,       # If 3x more negatives than positives
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=10)
```

---

## 6. Support Vector Machines (SVM)

### How It Works (Visual)

```
    Feature 2
    │
    │    ○ ○ ○
    │  ○ ○ ○ ○ ○
    │    ○ ○ ○
    │─ ─ ─ ─ ─ ─ ─ ─ ← Decision boundary (hyperplane)
    │    ● ● ●         ↕ Margin (maximize this gap)
    │  ● ● ● ● ●
    │    ● ● ●
    └──────────────── Feature 1

SVM finds the line (or hyperplane) that MAXIMIZES the gap
between the two classes. The points closest to the line
are called "support vectors" — they define the boundary.
```

### Kernel Trick (For Non-Linear Data)

```
Original space (not separable):     After kernel transform (separable!):

    ● ○ ● ○ ●                           ○ ○ ○
    ○ ● ○ ● ○                         ○ ○ ○ ○ ○
    ● ○ ● ○ ●                       ─────────────
    ○ ● ○ ● ○                         ● ● ● ● ●
                                        ● ● ●

The kernel maps data to a higher dimension where it IS separable.
```

### Assumptions

| Assumption | What It Means |
|------------|--------------|
| **Feature scaling required** | All features must be on same scale (use StandardScaler) |
| **Works best with clear margin** | Classes should be somewhat separable |
| **Sensitive to outliers** | Outliers can shift the boundary |

### Hyperparameters

| Parameter | What It Does | Values |
|-----------|-------------|--------|
| `C` | Regularization (trade-off: margin width vs errors) | 0.01-1000 |
| `kernel` | Shape of decision boundary | 'linear', 'rbf', 'poly' |
| `gamma` | How far influence of single point reaches | 'scale', 'auto', 0.001-10 |
| `degree` | Degree for polynomial kernel | 2-5 |

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# MUST scale features for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = SVC(
    C=1.0,
    kernel='rbf',        # Radial basis function (most common)
    gamma='scale',
    probability=True     # Enable predict_proba()
)
```

---

## 7. K-Nearest Neighbors (KNN)

### How It Works (Visual)

```
    Feature 2
    │
    │    ○       ○
    │  ○   ○
    │      ╔═══╗
    │    ○ ║ ? ║ ●      ← New point: who are my 5 nearest neighbors?
    │      ╚═══╝
    │  ●     ● ●        Neighbors: 3 ● and 2 ○
    │    ●               Vote: ● wins (3 vs 2)
    │        ●           Prediction: ●
    └──────────────── Feature 1

KNN doesn't "learn" anything. It just remembers all training data.
For a new point, it finds the K closest points and takes a vote.
```

### Assumptions

| Assumption | What It Means |
|------------|--------------|
| **Feature scaling required** | Distance-based → features must be on same scale |
| **Meaningful distance** | Euclidean distance must make sense for your features |
| **Sufficient data** | Needs enough neighbors to vote reliably |
| **Low dimensionality** | Struggles with many features (curse of dimensionality) |

### Hyperparameters

| Parameter | What It Does | Values |
|-----------|-------------|--------|
| `n_neighbors` (K) | How many neighbors to consider | 3-50 (odd numbers) |
| `weights` | How to weight neighbors | 'uniform', 'distance' |
| `metric` | Distance measure | 'euclidean', 'manhattan', 'minkowski' |

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,        # Look at 5 nearest neighbors
    weights='distance',   # Closer neighbors count more
    metric='euclidean'
)
```

---

## 8. Naive Bayes

### How It Works (Visual)

```
Given: Customer has tenure=3, charges=$90, contract=month-to-month

Question: What's the probability they'll churn?

P(churn | features) ∝ P(tenure=3 | churn) × P(charges=$90 | churn) × P(month | churn) × P(churn)

It calculates probability of each feature given each class,
then multiplies them together (assumes features are INDEPENDENT).

P(churn | this customer) = 0.82
P(stay  | this customer) = 0.18
→ Prediction: CHURN
```

### Assumptions

| Assumption | What It Means | Reality |
|------------|--------------|---------|
| **Feature independence** | Features don't affect each other | Almost never true! |
| **Feature distribution** | Gaussian, Multinomial, or Bernoulli | Must match your data |

**Despite violating assumptions, Naive Bayes often works surprisingly well!**

### Variants

| Variant | Use When |
|---------|----------|
| `GaussianNB` | Continuous features (numbers) |
| `MultinomialNB` | Count data (word frequencies) |
| `BernoulliNB` | Binary features (yes/no) |

### Hyperparameters

| Parameter | What It Does |
|-----------|-------------|
| `var_smoothing` (Gaussian) | Prevents zero probabilities |
| `alpha` (Multinomial) | Laplace smoothing |

```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()  # Almost no hyperparameters!
```

---

## 9. K-Means Clustering

### How It Works (Visual)

```
Step 1: Place K random centers    Step 2: Assign points to nearest center

    ×₁                                ○○○ ×₁ ○○
                                      ○○○○○○
         ×₂                              ●●● ×₂ ●●
                                         ●●●●●
              ×₃                              ▲▲▲ ×₃
                                              ▲▲▲▲

Step 3: Move centers to middle    Step 4: Repeat until stable

      ×₁ (moved)                  Final clusters:
    ○○○○○○○                       ┌─────────────────┐
                                  │ Cluster 1: ○○○○ │
        ×₂ (moved)               │ Cluster 2: ●●●● │
    ●●●●●●●                      │ Cluster 3: ▲▲▲▲ │
                                  └─────────────────┘
           ×₃ (moved)
    ▲▲▲▲▲▲
```

### Assumptions

| Assumption | What It Means |
|------------|--------------|
| **Spherical clusters** | Clusters are roughly round |
| **Similar size** | Clusters have similar number of points |
| **You know K** | Must specify number of clusters |

### How to Choose K (Elbow Method)

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('K (number of clusters)')
plt.ylabel('Inertia (lower = tighter clusters)')
plt.title('Elbow Method: Pick K where curve bends')
plt.show()

# The "elbow" point is your best K
```

---

## 10. Neural Networks

### How It Works (Visual)

```
INPUT LAYER          HIDDEN LAYERS           OUTPUT LAYER
(your features)      (learned patterns)      (prediction)

  tenure ──────┐
               ├──→ [neuron] ──┐
  charges ─────┤               ├──→ [neuron] ──┐
               ├──→ [neuron] ──┤               ├──→ P(churn)
  contract ────┤               ├──→ [neuron] ──┘
               ├──→ [neuron] ──┘
  tickets ─────┘

Each connection has a WEIGHT (learned during training).
Each neuron applies: output = activation(sum of weighted inputs)
```

### Assumptions

| Assumption | What It Means |
|------------|--------------|
| **Lots of data** | Needs 10K+ samples to work well |
| **Feature scaling** | Must normalize/standardize inputs |
| **No specific distribution** | Very flexible |

### Hyperparameters

| Parameter | What It Does | Values |
|-----------|-------------|--------|
| `hidden_layer_sizes` | Architecture | (100,), (64, 32), (128, 64, 32) |
| `activation` | Non-linearity | 'relu', 'tanh', 'sigmoid' |
| `learning_rate_init` | Step size | 0.0001-0.01 |
| `batch_size` | Samples per update | 32, 64, 128, 256 |
| `epochs` / `max_iter` | Training passes | 50-1000 |
| `dropout` | Regularization | 0.1-0.5 |
| `optimizer` | How to update weights | 'adam', 'sgd' |

---

## 11. All Hyperparameters Reference

### Universal Tuning Strategy

```
1. START with defaults
2. IDENTIFY the problem:
   - Overfitting? → More regularization
   - Underfitting? → More complexity
3. TUNE one parameter at a time
4. USE cross-validation to evaluate
5. USE automated search (Optuna, GridSearch) for final tuning
```

### Regularization Parameters (Prevent Overfitting)

| Algorithm | Parameter | Effect of Increasing |
|-----------|-----------|---------------------|
| Linear/Logistic | `C` (inverse) | LESS regularization |
| Ridge/Lasso | `alpha` | MORE regularization |
| Decision Tree | `max_depth` ↓ | MORE regularization |
| Random Forest | `max_depth` ↓, `min_samples_leaf` ↑ | MORE regularization |
| XGBoost | `eta` ↓, `max_depth` ↓, `lambda` ↑ | MORE regularization |
| Neural Network | `dropout` ↑, `weight_decay` ↑ | MORE regularization |
| SVM | `C` ↓ | MORE regularization |

### Automated Hyperparameter Tuning

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=10, verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 12. All Evaluation Metrics (Visual)

---

## 13. ROC Curve & AUC Explained

### What Is the ROC Curve?

```
    True Positive Rate (Recall)
    1.0 │         ╱─────────────── Perfect model (AUC = 1.0)
        │       ╱╱
        │     ╱╱
        │   ╱╱  ╱── Your model (AUC = 0.85)
        │  ╱╱ ╱╱
        │ ╱╱╱╱
        │╱╱╱
    0.5 │╱╱ ╱─────────────────── Random guessing (AUC = 0.5)
        │╱╱
        │╱
    0.0 │
        └──────────────────────── False Positive Rate
        0.0                  1.0

The ROC curve shows: "If I catch X% of churners (TPR),
how many non-churners do I falsely flag (FPR)?"

AUC = Area Under this Curve
  - 1.0 = perfect (never wrong)
  - 0.5 = random (useless)
  - <0.5 = worse than random (model is inverted)
```

### How to Read It

```
Point on curve at (FPR=0.1, TPR=0.8) means:
  "I can catch 80% of churners while only
   falsely flagging 10% of non-churners"
```

### Code to Plot ROC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### AUC Interpretation

| AUC | Meaning | Action |
|-----|---------|--------|
| 0.90-1.00 | Excellent | Deploy with confidence |
| 0.80-0.90 | Good | Acceptable for most use cases |
| 0.70-0.80 | Fair | May need improvement |
| 0.60-0.70 | Poor | Needs better features or algorithm |
| 0.50-0.60 | Failing | Model is barely better than random |

---

## 14. Confusion Matrix Deep Dive

### The Matrix (Visual)

```
                        PREDICTED
                    Positive    Negative
                 ┌───────────┬───────────┐
    ACTUAL       │           │           │
    Positive     │    TP     │    FN     │
    (Churn=1)    │   (Hit!)  │  (Missed!)│
                 │           │           │
                 ├───────────┼───────────┤
    ACTUAL       │           │           │
    Negative     │    FP     │    TN     │
    (Churn=0)    │(False     │(Correct   │
                 │  alarm!)  │  reject)  │
                 └───────────┴───────────┘

Example with 1000 customers:
                    Predicted    Predicted
                    Churn        Stay
                 ┌───────────┬───────────┐
    Actually     │           │           │
    Churned      │    120    │     30    │  ← 150 actually churned
                 │   (TP)    │   (FN)    │
                 ├───────────┼───────────┤
    Actually     │           │           │
    Stayed       │     50    │    800    │  ← 850 actually stayed
                 │   (FP)    │   (TN)    │
                 └───────────┴───────────┘
                      170         830        ← Predicted totals
```

### All Metrics Derived From Confusion Matrix

```
From the example above:
TP=120, FP=50, FN=30, TN=800

Accuracy    = (TP+TN) / Total        = (120+800)/1000  = 0.920  (92%)
Precision   = TP / (TP+FP)           = 120/170         = 0.706  (71%)
Recall      = TP / (TP+FN)           = 120/150         = 0.800  (80%)
Specificity = TN / (TN+FP)           = 800/850         = 0.941  (94%)
F1 Score    = 2×(P×R)/(P+R)          = 2×(0.706×0.8)/(0.706+0.8) = 0.750
FPR         = FP / (FP+TN)           = 50/850          = 0.059  (6%)
```

### Visual: What Each Metric Focuses On

```
PRECISION: "Of everyone I PREDICTED as churn, how many actually churned?"
┌───────────┬───────────┐
│    TP     │           │
│  ████████ │           │  Precision = TP / (TP + FP)
├───────────┤           │            = "How trustworthy are my alarms?"
│    FP     │           │
│  ████████ │           │
└───────────┴───────────┘
 ↑ Precision looks at this column


RECALL: "Of everyone who ACTUALLY churned, how many did I catch?"
┌───────────┬───────────┐
│    TP     │    FN     │
│  ████████ │  ████████ │  Recall = TP / (TP + FN)
│           │           │        = "How many churners did I miss?"
├───────────┼───────────┤
│           │           │
│           │           │
└───────────┴───────────┘
 ↑ Recall looks at this row
```

### Code to Plot Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
```

---

### Precision-Recall Curve

```
    Precision
    1.0 │●
        │ ●●
        │   ●●●
        │      ●●●
    0.8 │         ●●●
        │            ●●●
        │               ●●●
    0.6 │                  ●●●
        │                     ●●●
    0.4 │                        ●●●
        │
    0.0 │
        └──────────────────────────── Recall
        0.0                       1.0

As you try to catch MORE churners (higher recall),
your precision drops (more false alarms).

The curve shows this trade-off.
AUC-PR = area under this curve (better for imbalanced data than ROC)
```

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2, label=f'Model (AP = {ap:.3f})')
plt.xlabel('Recall (How many churners caught)')
plt.ylabel('Precision (How accurate are alarms)')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 15. Regression Metrics Visualized

### MAE (Mean Absolute Error)

```
    Actual vs Predicted:

    Actual:    100   200   150   300   250
    Predicted: 110   180   160   290   270

    Errors:    |10|  |20|  |10|  |10|  |20|  = 10, 20, 10, 10, 20

    MAE = average of errors = (10+20+10+10+20)/5 = 14

    "On average, predictions are off by $14"
```

### RMSE (Root Mean Squared Error)

```
    Same errors: 10, 20, 10, 10, 20

    Squared:     100, 400, 100, 100, 400
    Mean:        (100+400+100+100+400)/5 = 220
    Root:        √220 = 14.83

    RMSE = 14.83

    RMSE penalizes BIG errors more than MAE.
    If one prediction is off by 100, RMSE goes up a lot.
```

### R² (R-Squared)

```
    R² = 1 - (model errors / baseline errors)

    Baseline = always predict the average

    If R² = 0.85:
      "My model explains 85% of the variation in the data"
      "Only 15% is unexplained (noise)"

    R² = 1.0  → Perfect predictions
    R² = 0.0  → No better than predicting the average
    R² < 0.0  → Worse than predicting the average (bad model!)
```

### Visual Comparison

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted (R²={r2:.3f}, RMSE={rmse:.1f})')
plt.legend()
plt.show()
```

---

## 16. How to Pick the Right Metric

### Decision Flowchart

```
What type of problem?
│
├── CLASSIFICATION
│   │
│   ├── Classes balanced (50/50)?
│   │   └── Use: Accuracy, F1, or AUC-ROC
│   │
│   ├── Classes imbalanced (95/5)?
│   │   └── Use: AUC-PR, F1, or Recall
│   │
│   ├── False positives are expensive?
│   │   │ (e.g., blocking a legitimate transaction)
│   │   └── Use: PRECISION
│   │
│   ├── False negatives are expensive?
│   │   │ (e.g., missing a cancer diagnosis)
│   │   └── Use: RECALL
│   │
│   ├── Need to rank predictions?
│   │   │ (e.g., "top 100 most likely to churn")
│   │   └── Use: AUC-ROC
│   │
│   └── Need calibrated probabilities?
│       │ (e.g., "this customer has exactly 73% chance")
│       └── Use: Log Loss or Brier Score
│
└── REGRESSION
    │
    ├── Want interpretable error?
    │   └── Use: MAE ("off by $14 on average")
    │
    ├── Want to penalize big errors?
    │   └── Use: RMSE
    │
    ├── Want percentage error?
    │   └── Use: MAPE ("off by 5% on average")
    │
    └── Want to compare models?
        └── Use: R² ("explains 85% of variance")
```

### Real-World Examples

| Business Problem | Best Metric | Why |
|-----------------|-------------|-----|
| Fraud detection | **Recall** + AUC-PR | Missing fraud costs millions |
| Email spam filter | **Precision** | Blocking real email is terrible |
| Customer churn | **AUC-ROC** or **F1** | Balance both errors |
| Medical screening | **Recall** (sensitivity) | Missing disease is dangerous |
| Ad click prediction | **Log Loss** | Need accurate probabilities for bidding |
| House price prediction | **MAE** or **RMSE** | Interpretable dollar error |
| Demand forecasting | **MAPE** | Percentage error across different scales |
| Credit scoring | **AUC-ROC** + **KS statistic** | Ranking ability matters |
| Recommendation ranking | **NDCG** or **MAP** | Order of results matters |

### Complete Metrics Code

```python
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix,
    # Regression
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np

# === CLASSIFICATION METRICS ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("CLASSIFICATION METRICS")
print("=" * 40)
print(f"Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:    {precision_score(y_test, y_pred):.4f}")
print(f"Recall:       {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:     {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:      {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR:       {average_precision_score(y_test, y_prob):.4f}")
print(f"Log Loss:     {log_loss(y_test, y_prob):.4f}")
print(f"MCC:          {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen Kappa:  {cohen_kappa_score(y_test, y_pred):.4f}")

print(f"\n{classification_report(y_test, y_pred)}")


# === REGRESSION METRICS ===
# y_pred_reg = model.predict(X_test)  # For regression models

# print("REGRESSION METRICS")
# print("=" * 40)
# print(f"MAE:   {mean_absolute_error(y_test, y_pred_reg):.4f}")
# print(f"RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred_reg)):.4f}")
# print(f"R²:    {r2_score(y_test, y_pred_reg):.4f}")
# print(f"MAPE:  {mean_absolute_percentage_error(y_test, y_pred_reg):.4f}")
```

### Plotting All Metrics Together

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0,0].plot(fpr, tpr, 'b-', linewidth=2)
axes[0,0].plot([0,1], [0,1], 'r--')
axes[0,0].set_title(f'ROC Curve (AUC={roc_auc_score(y_test, y_prob):.3f})')
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')

# 2. Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[0,1].plot(rec, prec, 'g-', linewidth=2)
axes[0,1].set_title(f'Precision-Recall (AP={average_precision_score(y_test, y_prob):.3f})')
axes[0,1].set_xlabel('Recall')
axes[0,1].set_ylabel('Precision')

# 3. Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1,0],
                                         display_labels=['Stay','Churn'], cmap='Blues')
axes[1,0].set_title('Confusion Matrix')

# 4. Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
axes[1,1].plot(prob_pred, prob_true, 'bo-', label='Model')
axes[1,1].plot([0,1], [0,1], 'r--', label='Perfect calibration')
axes[1,1].set_title('Calibration Curve')
axes[1,1].set_xlabel('Predicted Probability')
axes[1,1].set_ylabel('Actual Probability')
axes[1,1].legend()

plt.tight_layout()
plt.show()
```

---

### Quick Reference Card

| Metric | Perfect | Random | Formula | Use When |
|--------|:---:|:---:|---------|----------|
| **Accuracy** | 1.0 | 0.5 | (TP+TN)/All | Balanced classes |
| **Precision** | 1.0 | pos_rate | TP/(TP+FP) | FP is costly |
| **Recall** | 1.0 | pos_rate | TP/(TP+FN) | FN is costly |
| **F1** | 1.0 | ~0.5 | 2PR/(P+R) | Balance P & R |
| **AUC-ROC** | 1.0 | 0.5 | Area under ROC | Ranking ability |
| **AUC-PR** | 1.0 | pos_rate | Area under PR | Imbalanced data |
| **Log Loss** | 0.0 | 0.693 | -mean(log(p)) | Probability quality |
| **MCC** | 1.0 | 0.0 | (TP×TN-FP×FN)/√... | Single best metric |
| **MAE** | 0 | — | mean(\|error\|) | Interpretable |
| **RMSE** | 0 | — | √mean(error²) | Penalize big errors |
| **R²** | 1.0 | 0.0 | 1-SS_res/SS_tot | % variance explained |

---

*End of Algorithms & Metrics Visual Guide*
