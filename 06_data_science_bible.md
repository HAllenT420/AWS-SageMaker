# The Data Science Bible: End-to-End Guide

> Everything you need to know about the data science process — from raw data to production model.
> Sequential, comprehensive, with all metrics, checks, and decision frameworks.

---

## Table of Contents

1. [The Data Science Process (Overview)](#1-the-data-science-process)
2. [Step 1: Problem Definition](#2-step-1-problem-definition)
3. [Step 2: Data Collection](#3-step-2-data-collection)
4. [Step 3: Exploratory Data Analysis (EDA)](#4-step-3-eda)
5. [Step 4: Data Cleaning](#5-step-4-data-cleaning)
6. [Step 5: Feature Engineering](#6-step-5-feature-engineering)
   - [Scalers: Types, When to Use, and How They Work](#scalers-types-when-to-use-and-how-they-work)
   - [Feature Selection: Detailed Notes](#feature-selection-methods)
   - [PCA: Dimensionality Reduction for High-Dimensional Data](#pca-dimensionality-reduction-for-high-dimensional-data)
7. [Step 6: Data Splitting](#7-step-6-data-splitting)
   - [Preprocessing After Splitting (Avoiding Data Leakage)](#preprocessing-after-splitting-avoiding-data-leakage)
8. [Step 7: Model Selection](#8-step-7-model-selection)
9. [Step 8: Model Training](#9-step-8-model-training)
10. [Step 9: Model Evaluation (All Metrics)](#10-step-9-model-evaluation)
11. [Step 10: Hyperparameter Tuning](#11-step-10-hyperparameter-tuning)
12. [Step 11: Model Interpretation & Explainability](#12-step-11-model-interpretation)
    - [Feature Importance: Why It Matters and How to Use It](#feature-importance-why-it-matters-and-how-to-use-it)
13. [Step 12: Model Validation & Testing](#13-step-12-model-validation)
14. [Step 13: Deployment](#14-step-13-deployment)
15. [Step 14: Monitoring & Maintenance](#15-step-14-monitoring)
16. [Algorithm Reference (When to Use What)](#16-algorithm-reference)
17. [Metrics Reference (Complete)](#17-metrics-reference)
18. [Common Pitfalls & How to Avoid Them](#18-common-pitfalls)

---

## 1. The Data Science Process

### CRISP-DM Framework (Industry Standard)

```
┌─────────────────────────────────────────────────┐
│                                                   │
│    1. Business Understanding                      │
│         ↓                                         │
│    2. Data Understanding                          │
│         ↓                                         │
│    3. Data Preparation                            │
│         ↓                                         │
│    4. Modeling                                    │
│         ↓                                         │
│    5. Evaluation                                  │
│         ↓                                         │
│    6. Deployment                                  │
│         ↓                                         │
│    (Loop back to 1 — iterate!)                   │
│                                                   │
└─────────────────────────────────────────────────┘
```

### The Detailed Pipeline

```
Problem → Data → EDA → Clean → Features → Split → Model → Evaluate → Tune → Deploy → Monitor
   ↑                                                                                      │
   └──────────────────────── Feedback Loop ───────────────────────────────────────────────┘
```

---

## 2. Step 1: Problem Definition

### Questions to Answer BEFORE Touching Data

| Question | Why It Matters |
|----------|---------------|
| What business problem are we solving? | Keeps you focused |
| What decision will this model inform? | Defines the output |
| What does success look like? (metric + threshold) | Defines "done" |
| What's the baseline? (current approach without ML) | Benchmark to beat |
| Who will use the predictions? | Defines deployment type |
| How often do predictions need to refresh? | Batch vs real-time |
| What's the cost of wrong predictions? | Guides metric choice |
| Is there enough data? | Feasibility check |

### Problem Types

| Type | Output | Example |
|------|--------|---------|
| **Binary Classification** | Yes/No, 0/1 | Churn prediction, fraud detection |
| **Multi-class Classification** | Category (3+) | Product category, disease diagnosis |
| **Regression** | Continuous number | Price prediction, demand forecasting |
| **Ranking** | Ordered list | Search results, recommendations |
| **Clustering** | Group assignment | Customer segmentation |
| **Anomaly Detection** | Normal/Abnormal | Fraud, system failures |
| **Time Series Forecasting** | Future values | Sales forecast, stock prices |
| **NLP** | Text output/class | Sentiment, summarization |
| **Computer Vision** | Image labels/boxes | Object detection, classification |

### Defining Success Criteria

```
"The model is successful if:
  - AUC > 0.80 on holdout test set
  - Reduces customer churn by 15% in A/B test
  - Prediction latency < 100ms for real-time use
  - False positive rate < 5% (don't annoy good customers)"
```

---

## 3. Step 2: Data Collection

### Data Sources

| Source | Examples | Access Method |
|--------|----------|--------------|
| Databases | PostgreSQL, MySQL, Redshift | SQL queries |
| Data Lakes | S3, HDFS | Spark, Athena |
| APIs | REST endpoints, third-party | HTTP requests |
| Files | CSV, Excel, JSON, Parquet | pandas.read_* |
| Streaming | Kafka, Kinesis | Stream consumers |
| Web Scraping | Websites | BeautifulSoup, Scrapy |

### Data Quality Checks at Collection

```python
# Immediately after loading data, check:
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
```

### Data Documentation

For every dataset, document:
- **Source**: Where does it come from?
- **Freshness**: How often is it updated?
- **Schema**: Column names, types, descriptions
- **Volume**: How many rows? Growing how fast?
- **Quality**: Known issues, missing data patterns
- **Access**: Who owns it? Permissions needed?

---

## 4. Step 3: Exploratory Data Analysis (EDA)

### EDA Checklist

#### 4.1 Univariate Analysis (One Variable at a Time)

```python
# Numerical columns
df.describe()  # count, mean, std, min, 25%, 50%, 75%, max

# For each numerical column:
# - Distribution shape (normal? skewed? bimodal?)
# - Outliers (values beyond 1.5×IQR)
# - Range (min/max reasonable?)

# Categorical columns
df['category'].value_counts()
df['category'].nunique()

# For each categorical column:
# - Number of unique values
# - Frequency of each value
# - Any rare categories?
```

#### 4.2 Bivariate Analysis (Relationship with Target)

```python
# Numerical feature vs target
df.groupby('target')['feature'].mean()  # Difference in means
# Correlation: df['feature'].corr(df['target'])

# Categorical feature vs target
pd.crosstab(df['category'], df['target'], normalize='index')
# Shows: what % of each category churns?
```

#### 4.3 Multivariate Analysis

```python
# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Look for:
# - Highly correlated features (>0.8) → may need to drop one
# - Features correlated with target → good predictors
# - Unexpected correlations → data leakage?
```

#### 4.4 Key Visualizations

| Plot | Use For | What to Look For |
|------|---------|-----------------|
| Histogram | Distribution of numerical features | Skewness, outliers, gaps |
| Box plot | Outliers, spread by group | Extreme values, group differences |
| Bar chart | Categorical frequencies | Imbalanced categories |
| Scatter plot | Relationship between 2 numerical | Correlation, clusters |
| Heatmap | Correlation matrix | Multicollinearity |
| Pair plot | All pairwise relationships | Patterns, separability |
| Time series plot | Trends over time | Seasonality, trends, anomalies |

---

## 5. Step 4: Data Cleaning

### Handling Missing Values

| Strategy | When to Use | Code |
|----------|-------------|------|
| **Drop rows** | <5% missing, random | `df.dropna()` |
| **Drop column** | >50% missing | `df.drop(columns=['col'])` |
| **Mean/Median imputation** | Numerical, random missing | `df['col'].fillna(df['col'].median())` |
| **Mode imputation** | Categorical | `df['col'].fillna(df['col'].mode()[0])` |
| **Forward/backward fill** | Time series | `df['col'].ffill()` |
| **KNN imputation** | Complex patterns | `KNNImputer(n_neighbors=5)` |
| **Indicator variable** | Missingness is informative | Add `col_is_missing` column |

### Handling Outliers

| Strategy | When to Use |
|----------|-------------|
| **Keep them** | They're real and informative |
| **Cap/Floor (Winsorize)** | Extreme but valid values |
| **Remove** | Data entry errors |
| **Log transform** | Right-skewed distributions |
| **Robust scaling** | Use median/IQR instead of mean/std |

```python
# IQR method to detect outliers
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers
df['col'] = df['col'].clip(lower=lower_bound, upper=upper_bound)
```

### Handling Duplicates

```python
# Check for duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Remove exact duplicates
df = df.drop_duplicates()

# Remove duplicates based on key columns
df = df.drop_duplicates(subset=['customer_id', 'date'], keep='last')
```

### Data Type Corrections

```python
# Fix types
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
```

---

## 6. Step 5: Feature Engineering

### What Is Feature Engineering?

Transforming raw data into features that better represent the underlying patterns for your model.

**This is often the most impactful step** — good features matter more than fancy algorithms.

### Numerical Transformations

| Transformation | When | Code |
|---------------|------|------|
| **Log transform** | Right-skewed data | `np.log1p(df['col'])` |
| **Square root** | Moderate skew | `np.sqrt(df['col'])` |
| **Standardization** | Different scales, for linear models | `StandardScaler()` |
| **Min-Max scaling** | Need 0-1 range | `MinMaxScaler()` |
| **Robust scaling** | Outliers present | `RobustScaler()` |
| **Binning** | Convert continuous to categorical | `pd.cut(df['age'], bins=[0,18,35,50,65,100])` |
| **Polynomial** | Non-linear relationships | `PolynomialFeatures(degree=2)` |

### Scalers: Types, When to Use, and How They Work

Scaling is the process of bringing numerical features to a comparable range so that no single feature dominates the model just because of its magnitude.

#### Why Scaling Matters

- **Linear models** (Logistic Regression, SVM, Linear Regression) are directly affected by feature scale — a feature ranging 0-1000 will dominate one ranging 0-1
- **Distance-based models** (KNN, K-Means, SVM with RBF kernel) compute distances between points — unscaled features distort distances
- **Gradient-based models** (Neural Networks) converge faster with scaled inputs
- **Tree-based models** (Random Forest, XGBoost, LightGBM) are **NOT affected** by scaling — they split on thresholds, not magnitudes

#### Scaler Comparison

| Scaler | Formula | Centers Data? | Handles Outliers? | Output Range | Best For |
|--------|---------|:---:|:---:|:---:|----------|
| **StandardScaler** | (x - mean) / std | Yes | No | ~(-3, +3) | Normally distributed features, linear models |
| **MinMaxScaler** | (x - min) / (max - min) | No | No | [0, 1] | Neural networks, image pixel values |
| **RobustScaler** | (x - median) / IQR | Yes | Yes | Varies | Data with outliers |
| **MaxAbsScaler** | x / max(\|x\|) | No | No | [-1, 1] | Sparse data (preserves zeros) |
| **PowerTransformer** | Box-Cox or Yeo-Johnson | Yes | Partially | ~(-3, +3) | Making data more Gaussian |
| **QuantileTransformer** | Maps to uniform/normal | Yes | Yes | [0, 1] or normal | Non-linear, heavy outliers |
| **Normalizer** | x / \|\|x\|\| (row-wise) | No | No | Unit norm | Text (TF-IDF), when direction matters |

#### Detailed Breakdown of Each Scaler

**1. StandardScaler (Z-score normalization)**

```python
from sklearn.preprocessing import StandardScaler

# Formula: z = (x - mean) / std
# After scaling: mean = 0, std = 1

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- **Use when**: Features are approximately normally distributed, no extreme outliers
- **Avoid when**: Heavy outliers (they distort mean and std)
- **Used by**: Logistic Regression, SVM, PCA, Neural Networks

**2. MinMaxScaler**

```python
from sklearn.preprocessing import MinMaxScaler

# Formula: x_scaled = (x - min) / (max - min)
# After scaling: all values in [0, 1]

scaler = MinMaxScaler()  # default range (0, 1)
# scaler = MinMaxScaler(feature_range=(-1, 1))  # custom range
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- **Use when**: Need bounded range (0-1), neural networks, image data
- **Avoid when**: Outliers exist (a single outlier compresses all other values)
- **Problem**: If test data has values outside train min/max, they exceed [0,1]

**3. RobustScaler**

```python
from sklearn.preprocessing import RobustScaler

# Formula: x_scaled = (x - median) / IQR
# IQR = Q3 - Q1 (interquartile range)
# After scaling: median = 0, IQR = 1

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- **Use when**: Data has significant outliers (fraud amounts, income, prices)
- **Why it works**: Median and IQR are not affected by extreme values (unlike mean/std)
- **Trade-off**: Does not bound the output range — outliers still exist, just less influential

**4. MaxAbsScaler**

```python
from sklearn.preprocessing import MaxAbsScaler

# Formula: x_scaled = x / max(|x|)
# After scaling: values in [-1, 1], zeros stay as zeros

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- **Use when**: Sparse data (CSR matrices from text vectorization) — preserves sparsity
- **Avoid when**: Features have very different ranges

**5. PowerTransformer**

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson: works with positive and negative values
# Box-Cox: works only with strictly positive values

scaler = PowerTransformer(method='yeo-johnson')  # or 'box-cox'
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- **Use when**: Features are heavily skewed and you want to make them more Gaussian
- **How it works**: Applies a power transformation (like log, but automatically finds the best exponent)
- **Bonus**: Also standardizes the output (mean=0, std=1)

**6. QuantileTransformer**

```python
from sklearn.preprocessing import QuantileTransformer

# Maps data to a uniform [0,1] or normal distribution
scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- **Use when**: Extreme outliers, non-linear distributions, nothing else works
- **How it works**: Ranks all values, then maps ranks to the desired distribution
- **Trade-off**: Destroys the original distance relationships between values (non-linear transform)

#### Decision Framework: Which Scaler to Choose

```
Does your data have significant outliers?
│
├── YES
│   ├── Need Gaussian-like output? → PowerTransformer (Yeo-Johnson)
│   ├── Extreme outliers, nothing works? → QuantileTransformer
│   └── Just need robust centering? → RobustScaler  (most common choice)
│
└── NO
    ├── Need [0,1] range? → MinMaxScaler
    ├── Sparse data? → MaxAbsScaler
    ├── Approximately normal? → StandardScaler  (most common choice)
    └── Need Gaussian output? → PowerTransformer
```

#### Which Models Need Scaling?

| Model | Needs Scaling? | Why |
|-------|:---:|------|
| Logistic Regression | Yes | Coefficients are scale-dependent |
| SVM | Yes | Distance-based, kernel sensitive to scale |
| KNN | Yes | Distance-based |
| K-Means | Yes | Distance-based |
| Neural Networks | Yes | Gradient descent converges faster |
| PCA | Yes | Variance-based, large-scale features dominate |
| Linear/Ridge/Lasso Regression | Yes | Regularization penalizes large coefficients |
| Random Forest | No | Threshold-based splits, scale-invariant |
| XGBoost / LightGBM / CatBoost | No | Threshold-based splits, scale-invariant |
| Decision Tree | No | Threshold-based splits |
| Naive Bayes | No | Probability-based |

### Categorical Encoding

| Method | When | Cardinality | Code |
|--------|------|-------------|------|
| **Label Encoding** | Ordinal categories | Any | `LabelEncoder()` |
| **One-Hot Encoding** | Nominal, low cardinality | <10 | `pd.get_dummies()` |
| **Target Encoding** | High cardinality | >10 | `TargetEncoder()` |
| **Frequency Encoding** | Count matters | Any | `df['col'].map(df['col'].value_counts())` |
| **Binary Encoding** | Very high cardinality | >50 | `BinaryEncoder()` |

### Date/Time Features

```python
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days
df['hour'] = df['timestamp'].dt.hour
df['is_business_hours'] = df['hour'].between(9, 17)
```

### Aggregation Features

```python
# Customer-level aggregations
customer_features = df.groupby('customer_id').agg(
    total_purchases=('amount', 'sum'),
    avg_purchase=('amount', 'mean'),
    purchase_count=('order_id', 'count'),
    days_since_last_purchase=('date', lambda x: (pd.Timestamp.now() - x.max()).days),
    unique_products=('product_id', 'nunique'),
    max_purchase=('amount', 'max')
).reset_index()
```

### Interaction Features

```python
# Combine features that might interact
df['price_per_unit'] = df['total_price'] / df['quantity']
df['tenure_x_charges'] = df['tenure_months'] * df['monthly_charges']
df['support_rate'] = df['num_tickets'] / df['tenure_months']
```

### Feature Selection Methods

| Method | Type | How It Works |
|--------|------|-------------|
| **Correlation filter** | Filter | Remove features correlated >0.9 with each other |
| **Variance threshold** | Filter | Remove near-zero variance features |
| **Chi-squared test** | Filter | Statistical test for categorical features |
| **Mutual information** | Filter | Non-linear dependency measure |
| **Recursive Feature Elimination (RFE)** | Wrapper | Train model, remove least important, repeat |
| **L1 Regularization (Lasso)** | Embedded | Model zeros out unimportant features |
| **Feature importance (tree-based)** | Embedded | Use model's built-in importance |
| **SHAP values** | Model-agnostic | Contribution of each feature to predictions |

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Select top 20 features by mutual information
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
```

#### Feature Selection: Detailed Notes and When to Use What

##### Why Feature Selection Matters

- **Reduces overfitting** — fewer features means less noise for the model to memorize
- **Improves speed** — fewer features = faster training and prediction
- **Improves interpretability** — easier to explain a model with 10 features than 200
- **Removes multicollinearity** — correlated features confuse linear models
- **Handles curse of dimensionality** — with too many features relative to samples, models struggle

##### The Three Types of Feature Selection

**1. Filter Methods (Fast, Model-Independent)**

Evaluate features independently of any model. Fast but may miss feature interactions.

```python
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest,
    mutual_info_classif, chi2, f_classif
)

# Remove near-zero variance features
# (features that are almost constant — they add no information)
var_selector = VarianceThreshold(threshold=0.01)
X_filtered = var_selector.fit_transform(X_train)

# Select top K features by statistical test
# For classification:
selector = SelectKBest(score_func=mutual_info_classif, k=20)
# For regression:
# selector = SelectKBest(score_func=f_regression, k=20)

X_selected = selector.fit_transform(X_train, y_train)
selected_mask = selector.get_support()
selected_features = X_train.columns[selected_mask].tolist()

# Correlation-based removal (drop one of highly correlated pairs)
corr_matrix = X_train.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper_triangle.columns
           if any(upper_triangle[col] > 0.90)]
X_train = X_train.drop(columns=to_drop)
```

**2. Wrapper Methods (Accurate, Slow)**

Use a model to evaluate subsets of features. More accurate but computationally expensive.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
# Trains model → removes least important feature → repeats
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator, n_features_to_select=15, step=1)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_].tolist()
feature_ranking = pd.Series(rfe.ranking_, index=X_train.columns).sort_values()

# RFECV — automatically finds optimal number of features using CV
rfecv = RFECV(estimator, step=1, cv=5, scoring='roc_auc', min_features_to_select=5)
rfecv.fit(X_train, y_train)
print(f"Optimal number of features: {rfecv.n_features_}")
```

**3. Embedded Methods (Best of Both Worlds)**

Feature selection happens during model training.

```python
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier

# Lasso (L1) — automatically zeros out unimportant features
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
important_features = X_train.columns[lasso.coef_ != 0].tolist()
print(f"Lasso selected {len(important_features)} features")

# Tree-based feature importance
model = XGBClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# Keep features with importance above a threshold
threshold = 0.01  # 1% importance
selected_features = importance[importance > threshold].index.tolist()
```

##### Decision Framework: Which Feature Selection Method?

| Situation | Recommended Method | Why |
|-----------|-------------------|-----|
| Quick baseline, many features | Variance Threshold + Correlation filter | Fast, removes obvious noise |
| Need best subset, have time | RFECV with cross-validation | Finds optimal count automatically |
| Using linear model | Lasso (L1 regularization) | Built-in feature selection |
| Using tree model | Feature importance + SHAP | Leverages model's own learning |
| Very high dimensionality (1000+) | Mutual Information → then RFE | Filter first to reduce, then refine |
| Need to explain to stakeholders | SHAP values | Shows direction and magnitude |

##### Important Notes on Feature Selection

1. **Always do feature selection on training data only** — selecting features using the full dataset is data leakage
2. **Feature importance != causation** — a feature being important to the model does not mean it causes the outcome
3. **Correlated features split importance** — if two features are correlated, the model may assign importance to one randomly; removing one often does not hurt performance
4. **More features is not always better** — after a point, adding features adds noise faster than signal
5. **Domain knowledge beats algorithms** — if you know a feature should not logically predict the target, remove it regardless of what the model says (it might be leakage)

### PCA: Dimensionality Reduction for High-Dimensional Data

#### What Is PCA?

PCA (Principal Component Analysis) transforms your original features into a new set of uncorrelated features called **principal components**, ordered by how much variance they explain. It is used to reduce the number of dimensions while retaining as much information as possible.

#### When to Use PCA

| Situation | Use PCA? | Why |
|-----------|:---:|------|
| 100+ features, many correlated | Yes | Reduces redundancy, speeds up training |
| Features from PCA-transformed data (V1-V28 in credit card dataset) | Already done | The V columns ARE PCA components |
| Need to visualize high-dimensional data | Yes | Reduce to 2-3 components for plotting |
| Linear model struggling with multicollinearity | Yes | PCA components are uncorrelated |
| Tree-based model (XGBoost, RF) | Rarely | Trees handle high dimensions and correlations natively |
| Need interpretable features | No | PCA components are not interpretable |
| Small dataset, many features | Yes | Reduces overfitting risk |

#### How PCA Works (Intuition)

```
Original Data: 50 features, many correlated
         ↓
Step 1: Standardize all features (REQUIRED — PCA is variance-based)
         ↓
Step 2: Compute covariance matrix (how features relate to each other)
         ↓
Step 3: Find eigenvectors (directions of maximum variance)
         ↓
Step 4: Rank by eigenvalue (how much variance each direction explains)
         ↓
Step 5: Keep top K components that explain enough variance (e.g., 95%)
         ↓
Result: K new features (components), uncorrelated, ordered by importance
```

Each principal component is a **linear combination** of the original features:

```
PC1 = 0.4*Feature_A + 0.3*Feature_B - 0.2*Feature_C + ...
PC2 = -0.1*Feature_A + 0.5*Feature_B + 0.3*Feature_C + ...
```

#### PCA Code Example

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: MUST scale before PCA (PCA is sensitive to feature magnitudes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Fit PCA — keep enough components to explain 95% variance
pca = PCA(n_components=0.95)  # keeps components until 95% variance explained
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original features: {X_train_scaled.shape[1]}")
print(f"PCA components kept: {X_train_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Step 3: Visualize how many components you need (Elbow/Scree plot)
pca_full = PCA().fit(X_train_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - How Many Components Do You Need?')
plt.legend()
plt.grid(True)
plt.show()
```

#### Decoding PCA: Understanding What the Components Mean

When you have a large number of columns (like 50-200 features) and PCA reduces them to 10-15 components, you lose direct interpretability. Here is how to decode what each component represents:

**Method 1: Loadings Matrix (Which original features contribute to each component)**

```python
# The loadings tell you how much each original feature contributes to each PC
loadings = pd.DataFrame(
    pca.components_.T,  # Transpose: rows = original features, cols = components
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X_train.columns  # Original feature names
)

# For PC1: which original features have the highest weight?
pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
print("Top features contributing to PC1:")
print(pc1_loadings.head(10))

# Visualize loadings for first 3 components
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ax in enumerate(axes):
    loadings[f'PC{i+1}'].sort_values().plot(kind='barh', ax=ax)
    ax.set_title(f'PC{i+1} Loadings')
    ax.set_xlabel('Loading Weight')
plt.tight_layout()
plt.show()
```

**Method 2: Naming Components by Dominant Features**

```python
# Automatically label each component by its top contributing features
for i in range(min(5, pca.n_components_)):
    pc = loadings[f'PC{i+1}']
    top_positive = pc.nlargest(3).index.tolist()
    top_negative = pc.nsmallest(3).index.tolist()

    print(f"\nPC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% variance):")
    print(f"  Driven by (positive): {top_positive}")
    print(f"  Driven by (negative): {top_negative}")

# Example output:
# PC1 (25.3% variance):
#   Driven by (positive): ['total_spend', 'num_transactions', 'account_age']
#   Driven by (negative): ['days_inactive', 'complaints', 'late_payments']
# Interpretation: PC1 represents "customer engagement" — high = active, low = disengaged
```

**Method 3: Inverse Transform (Reconstruct Original Features)**

```python
# If you need to go back from PCA space to original feature space
X_reconstructed = pca.inverse_transform(X_train_pca)
X_reconstructed = scaler.inverse_transform(X_reconstructed)

# This gives approximate original values (some info lost)
# Useful for: understanding what a specific PCA prediction means in real terms

# Reconstruction error tells you how much information was lost
reconstruction_error = np.mean((X_train_scaled - pca.inverse_transform(X_train_pca))**2)
print(f"Mean reconstruction error: {reconstruction_error:.6f}")
```

**Method 4: Biplot (Visualize Features and Samples Together)**

```python
def biplot(score, coeff, labels, pc_x=0, pc_y=1):
    """
    Biplot: shows both samples (dots) and feature directions (arrows)
    in PCA space. Arrows pointing in similar directions = correlated features.
    """
    plt.figure(figsize=(12, 8))

    # Plot samples
    plt.scatter(score[:, pc_x], score[:, pc_y], alpha=0.3, s=10)

    # Plot feature arrows
    for i, label in enumerate(labels):
        plt.arrow(0, 0, coeff[i, pc_x]*3, coeff[i, pc_y]*3,
                  color='red', alpha=0.7, head_width=0.05)
        plt.text(coeff[i, pc_x]*3.2, coeff[i, pc_y]*3.2,
                 label, color='red', fontsize=8)

    plt.xlabel(f'PC{pc_x+1}')
    plt.ylabel(f'PC{pc_y+1}')
    plt.title('PCA Biplot')
    plt.grid(True, alpha=0.3)
    plt.show()

biplot(X_train_pca, pca.components_.T, X_train.columns)
```

#### PCA with Large Number of Columns: Practical Strategy

When you have 100-500+ columns:

```python
# Strategy for very high-dimensional data

# 1. First, remove obviously useless features
#    - Zero/near-zero variance
#    - >90% missing
#    - Duplicate columns
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X_train_scaled)
print(f"After variance filter: {X_filtered.shape[1]} features remain")

# 2. Apply PCA with variance threshold
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_filtered)
print(f"After PCA (95% variance): {X_pca.shape[1]} components")

# 3. Create a mapping document for stakeholders
component_summary = []
for i in range(pca.n_components_):
    top_features = loadings[f'PC{i+1}'].abs().nlargest(5).index.tolist()
    component_summary.append({
        'Component': f'PC{i+1}',
        'Variance Explained': f"{pca.explained_variance_ratio_[i]*100:.1f}%",
        'Cumulative Variance': f"{sum(pca.explained_variance_ratio_[:i+1])*100:.1f}%",
        'Top Features': ', '.join(top_features),
        'Interpretation': ''  # Fill manually based on domain knowledge
    })

summary_df = pd.DataFrame(component_summary)
print(summary_df.to_string())
```

#### PCA vs Feature Selection: When to Use Which?

| Criteria | PCA | Feature Selection |
|----------|-----|-------------------|
| **Interpretability** | Low (components are combinations) | High (keeps original features) |
| **Multicollinearity** | Eliminates completely | May remain if not filtered |
| **Information loss** | Minimal (controlled by n_components) | Can lose important interactions |
| **Speed** | Fast transform after fitting | Varies by method |
| **Works with tree models** | Usually not needed | Yes, always useful |
| **Works with linear models** | Excellent | Good |
| **Handles 500+ features** | Excellent | Slower (especially wrappers) |
| **Production deployment** | Need to save PCA object | Simpler (just column list) |

**Rule of thumb**: Use PCA when you have many correlated features and are using linear/distance-based models. Use feature selection when interpretability matters or you are using tree-based models.

#### Important Notes on PCA

1. **Always scale before PCA** — PCA finds directions of maximum variance; unscaled features with large ranges will dominate
2. **PCA components are uncorrelated** — this eliminates multicollinearity entirely
3. **You cannot do PCA on categorical features** — use MCA (Multiple Correspondence Analysis) for categorical data, or encode first
4. **PCA is unsupervised** — it does not use the target variable; it may discard variance that is actually predictive
5. **For supervised dimensionality reduction**, consider LDA (Linear Discriminant Analysis) which uses class labels
6. **The credit card dataset V1-V28 columns are already PCA-transformed** — the original features were transformed for privacy; you cannot decode them back to original features without the original PCA object
7. **Fit PCA on training data only** — same leakage rules apply as with scalers

---

## 7. Step 6: Data Splitting

### Why Split?

| Set | Purpose | Typical Size |
|-----|---------|-------------|
| **Training** | Model learns from this | 60-80% |
| **Validation** | Tune hyperparameters, compare models | 10-20% |
| **Test** | Final unbiased evaluation (touch ONCE) | 10-20% |

### Splitting Rules

1. **Never use test data for any decision** — only evaluate final model once
2. **Stratify for classification** — maintain class proportions in all splits
3. **Time-based split for time series** — train on past, test on future
4. **Group split if needed** — keep all records of one customer in same split

```python
from sklearn.model_selection import train_test_split

# Standard split (stratified for classification)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Time-based split (for time series)
train = df[df['date'] < '2024-01-01']
val = df[(df['date'] >= '2024-01-01') & (df['date'] < '2024-04-01')]
test = df[df['date'] >= '2024-04-01']
```

### Cross-Validation

When you don't have enough data for a separate validation set:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Preprocessing After Splitting (Avoiding Data Leakage)

#### What Is Data Leakage?

Data leakage happens when information from the test set influences the training process. The model appears to perform well during development, but fails in production because it was "cheating" — it had access to knowledge it won't have when making real predictions on unseen data.

The most common form of leakage is **preprocessing leakage**: fitting a scaler, encoder, or imputer on the full dataset (including test data) before splitting.

#### Why This Matters

When you fit a scaler on the full dataset, the scaler learns **global statistics** (mean, median, standard deviation, IQR) that include information from the test set. For example:

- `StandardScaler` learns the mean and std of the entire dataset
- `RobustScaler` learns the median and IQR of the entire dataset
- `MinMaxScaler` learns the min and max of the entire dataset
- `KNNImputer` uses test set neighbors to fill training set missing values

These statistics then "leak" test set information into the training data transformation. The result:

1. **Overly optimistic metrics** — Your test set evaluation is no longer unbiased because the model indirectly saw test data patterns during training
2. **Poor generalization** — In production, you won't have future data to compute statistics from, so the model encounters a distribution it wasn't truly prepared for
3. **False confidence** — You deploy a model thinking it has 95% AUC, but in reality it's 90%

#### The Correct Order

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│   WRONG (Leakage):                                               │
│   ─────────────────                                              │
│   1. Load full dataset                                           │
│   2. Scale/transform full dataset  ← scaler sees test data!     │
│   3. Split into train/test                                       │
│   4. Train model                                                 │
│   5. Evaluate on test  ← metrics are biased                     │
│                                                                   │
│   CORRECT (No Leakage):                                          │
│   ────────────────────                                           │
│   1. Load full dataset                                           │
│   2. Split into train/test FIRST                                 │
│   3. Fit scaler on X_train ONLY                                  │
│   4. Transform X_train with fitted scaler                        │
│   5. Transform X_test with SAME fitted scaler (no re-fitting)   │
│   6. Train model on transformed X_train                          │
│   7. Evaluate on transformed X_test  ← metrics are unbiased     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Code Example

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# Step 1: Split FIRST — before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Fit scaler on training data ONLY
scaler = RobustScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])

# Step 3: Transform test data using the SAME fitted scaler
# .transform() — NOT .fit_transform()
X_test['Amount'] = scaler.transform(X_test[['Amount']])

# Step 4: Save the scaler for production use
import joblib
joblib.dump(scaler, 'artifacts/amount_scaler.pkl')

# At inference time (production):
# loaded_scaler = joblib.load('artifacts/amount_scaler.pkl')
# new_data['Amount'] = loaded_scaler.transform(new_data[['Amount']])
```

#### The Key Distinction: fit_transform vs transform

| Method | What It Does | When to Use |
|--------|-------------|-------------|
| `scaler.fit(X_train)` | Learns statistics (mean, std, etc.) from X_train | Once, on training data only |
| `scaler.transform(X)` | Applies the learned transformation | On any data (train, test, production) |
| `scaler.fit_transform(X_train)` | Shortcut: fit + transform in one call | Only on training data |

**Rule**: `fit_transform()` on train, `transform()` on everything else.

#### What Operations Cause Leakage If Done Before Splitting?

| Operation | Leaks Because |
|-----------|--------------|
| **Scaling** (StandardScaler, MinMaxScaler, RobustScaler) | Learns global mean/std/min/max/median/IQR |
| **Imputation** (mean, median, KNN) | Fill values come from test data too |
| **Encoding** (Target encoding) | Target statistics include test labels |
| **Feature selection** (based on correlation with target) | Selection influenced by test set patterns |
| **Oversampling/SMOTE** | Synthetic samples may be near test points |
| **PCA / Dimensionality reduction** | Components learned from full data |
| **Binning** (based on quantiles) | Bin edges influenced by test data |

#### What Is Safe to Do Before Splitting?

| Operation | Why It's Safe |
|-----------|--------------|
| **Dropping columns** (e.g., ID, irrelevant) | No statistics learned |
| **Fixing data types** (string → datetime) | No statistics learned |
| **Removing duplicates** | No statistics learned |
| **Renaming columns** | No statistics learned |
| **Filtering rows** (removing known bad data) | No statistics learned |
| **One-hot encoding** (if categories are fixed/known) | No statistics from data distribution |

#### Using sklearn Pipeline to Prevent Leakage Automatically

The safest approach is to use `sklearn.pipeline.Pipeline`, which guarantees that all preprocessing steps are fit only on training data during cross-validation and model fitting:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('scale_amount', RobustScaler(), ['Amount']),
    ],
    remainder='passthrough'  # Leave other columns unchanged
)

# Create pipeline: preprocessing + model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])

# Now fit and predict — no leakage possible
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_probs = pipeline.predict_proba(X_test)[:, 1]

# The pipeline automatically:
# - Fits the scaler on X_train only
# - Transforms X_train for model training
# - Transforms X_test using train-fitted scaler for prediction
```

#### Real-World Impact of Leakage

| Dataset Size | Typical Metric Inflation from Leakage |
|-------------|--------------------------------------|
| Small (<10K rows) | 2-5% AUC overestimate |
| Medium (10K-100K) | 1-3% AUC overestimate |
| Large (>100K) | 0.5-1% AUC overestimate |

The smaller your dataset, the worse the leakage effect — because the test set has more influence on the global statistics.

#### Checklist: Am I Leaking?

Ask yourself these questions for every preprocessing step:

- [ ] Does this step compute any statistic from the data (mean, std, min, max, frequency)?
- [ ] If yes, am I computing it from training data only?
- [ ] Am I applying SMOTE/oversampling only to training data?
- [ ] Am I doing feature selection only on training data?
- [ ] Would this feature be available at prediction time in production?
- [ ] Am I using a Pipeline to enforce the correct order automatically?

If any answer is wrong, you have leakage.

---

## 8. Step 7: Model Selection

### Decision Framework: Which Algorithm?

```
Is your target variable...
│
├── Continuous (number)? → REGRESSION
│   ├── Linear relationship? → Linear Regression, Ridge, Lasso
│   ├── Non-linear? → Random Forest, XGBoost, Neural Network
│   └── Time-dependent? → ARIMA, Prophet, LSTM
│
├── Categorical (2 classes)? → BINARY CLASSIFICATION
│   ├── Need interpretability? → Logistic Regression, Decision Tree
│   ├── Need best accuracy? → XGBoost, LightGBM, Neural Network
│   └── Small dataset? → SVM, KNN
│
├── Categorical (3+ classes)? → MULTI-CLASS CLASSIFICATION
│   ├── Few classes? → Same as binary (most algorithms handle it)
│   └── Many classes? → Neural Network, One-vs-Rest
│
├── No target (unsupervised)? → CLUSTERING
│   ├── Know number of clusters? → K-Means
│   ├── Don't know? → DBSCAN, Hierarchical
│   └── Need soft assignments? → Gaussian Mixture
│
└── Sequence/text/image? → DEEP LEARNING
    ├── Text → Transformers (BERT, GPT)
    ├── Images → CNN (ResNet, EfficientNet)
    └── Sequences → LSTM, Transformer
```

### Algorithm Comparison (Tabular Data)

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Logistic Regression** | Fast, interpretable, baseline | Linear only | Baseline, interpretability |
| **Decision Tree** | Interpretable, handles non-linear | Overfits easily | Explanation, simple rules |
| **Random Forest** | Robust, handles non-linear | Slow for large data, less interpretable | General purpose |
| **XGBoost** | Best accuracy (tabular), fast | Less interpretable, needs tuning | Competitions, production |
| **LightGBM** | Faster than XGBoost, handles categorical | Similar to XGBoost | Large datasets |
| **CatBoost** | Handles categorical natively | Slower training | Lots of categorical features |
| **SVM** | Works in high dimensions | Slow for large data, needs scaling | Small-medium datasets |
| **KNN** | Simple, no training | Slow prediction, curse of dimensionality | Small datasets, baselines |
| **Neural Network** | Handles any pattern | Needs lots of data, black box | Large data, complex patterns |

### Start Simple, Then Iterate

```
1. Logistic Regression (baseline) → "Can a simple model solve this?"
2. Random Forest → "Does non-linearity help?"
3. XGBoost/LightGBM → "Can gradient boosting do better?"
4. Neural Network → "Is there a complex pattern to exploit?"
```

---

## 9. Step 8: Model Training

### Training Process

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Initialize model with hyperparameters
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 2. Fit on training data
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # Monitor validation performance
    early_stopping_rounds=10,    # Stop if no improvement for 10 rounds
    verbose=10                   # Print every 10 rounds
)

# 3. Predict
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities
y_pred = model.predict(X_test)                      # Binary predictions
```

### Handling Imbalanced Data

When one class is much rarer (e.g., 5% fraud, 95% legitimate):

| Strategy | How | When |
|----------|-----|------|
| **Class weights** | `class_weight='balanced'` | First try, simple |
| **SMOTE** | Generate synthetic minority samples | Moderate imbalance |
| **Undersampling** | Remove majority samples | Lots of data |
| **Oversampling** | Duplicate minority samples | Small dataset |
| **Threshold tuning** | Adjust decision threshold from 0.5 | After training |
| **Focal Loss** | Modified loss function | Deep learning |
| **Ensemble** | Balanced bagging | Severe imbalance |

```python
# Class weights
model = XGBClassifier(scale_pos_weight=19)  # ratio of neg/pos

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

---

## 10. Step 9: Model Evaluation (All Metrics)

### Classification Metrics

#### Confusion Matrix (Foundation of All Metrics)

```
                    Predicted
                 Positive  Negative
Actual Positive    TP        FN
Actual Negative    FP        TN

TP = True Positive  (correctly predicted positive)
FP = False Positive (incorrectly predicted positive) — "False alarm"
FN = False Negative (missed positive) — "Missed detection"
TN = True Negative  (correctly predicted negative)
```

#### Core Metrics

| Metric | Formula | Range | Use When |
|--------|---------|-------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-1 | Balanced classes ONLY |
| **Precision** | TP/(TP+FP) | 0-1 | False positives are costly |
| **Recall (Sensitivity)** | TP/(TP+FN) | 0-1 | False negatives are costly |
| **F1 Score** | 2×(P×R)/(P+R) | 0-1 | Balance precision & recall |
| **Specificity** | TN/(TN+FP) | 0-1 | True negative rate matters |
| **AUC-ROC** | Area under ROC curve | 0-1 | Overall ranking ability |
| **AUC-PR** | Area under Precision-Recall curve | 0-1 | Imbalanced datasets |
| **Log Loss** | -mean(y×log(p)+(1-y)×log(1-p)) | 0-∞ | Probability calibration |
| **MCC** | (TP×TN-FP×FN)/√(...) | -1 to 1 | Imbalanced, single metric |

#### When to Use Which Metric

| Scenario | Primary Metric | Why |
|----------|---------------|-----|
| Fraud detection | **Recall** + AUC-PR | Missing fraud is very costly |
| Spam filter | **Precision** | Don't want to block real emails |
| Medical diagnosis | **Recall** | Missing a disease is dangerous |
| Customer churn | **AUC-ROC** or **F1** | Balance both types of errors |
| Balanced dataset | **Accuracy** or **F1** | All errors equally bad |
| Ranking (recommendations) | **AUC-ROC** | Care about ordering |
| Probability estimates | **Log Loss** or **Brier Score** | Need calibrated probabilities |

#### Code for All Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, matthews_corrcoef
)

# All at once
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Individual metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"AUC-PR:    {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_pred_proba):.4f}")
print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")
```

### Regression Metrics

| Metric | Formula | Range | Use When |
|--------|---------|-------|----------|
| **MAE** | mean(\|y-ŷ\|) | 0-∞ | Robust to outliers, interpretable |
| **MSE** | mean((y-ŷ)²) | 0-∞ | Penalize large errors |
| **RMSE** | √MSE | 0-∞ | Same units as target |
| **R²** | 1 - SS_res/SS_tot | -∞ to 1 | Proportion of variance explained |
| **MAPE** | mean(\|y-ŷ\|/\|y\|)×100 | 0-∞% | Percentage error |
| **Adjusted R²** | R² adjusted for # features | -∞ to 1 | Compare models with different features |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

### Threshold Tuning (Classification)

The default threshold is 0.5, but it's rarely optimal:

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold that gives desired recall (e.g., 0.90)
target_recall = 0.90
idx = np.argmin(np.abs(recalls - target_recall))
optimal_threshold = thresholds[idx]
print(f"Threshold for {target_recall} recall: {optimal_threshold:.3f}")
print(f"Precision at this threshold: {precisions[idx]:.3f}")

# Apply custom threshold
y_pred_custom = (y_pred_proba >= optimal_threshold).astype(int)
```

---

## 11. Step 10: Hyperparameter Tuning

### What Are Hyperparameters?

Parameters set BEFORE training (not learned from data):

| Algorithm | Key Hyperparameters |
|-----------|-------------------|
| **XGBoost** | max_depth, learning_rate, n_estimators, subsample, colsample_bytree |
| **Random Forest** | n_estimators, max_depth, min_samples_split, max_features |
| **Logistic Regression** | C (regularization), penalty (L1/L2) |
| **SVM** | C, kernel, gamma |
| **Neural Network** | layers, neurons, learning_rate, batch_size, dropout |

### Tuning Methods

| Method | How | Pros | Cons |
|--------|-----|------|------|
| **Grid Search** | Try all combinations | Exhaustive | Slow, exponential |
| **Random Search** | Random combinations | Fast, good coverage | May miss optimal |
| **Bayesian (Optuna)** | Learn from previous trials | Most efficient | More complex |
| **Halving** | Start with many, eliminate bad ones | Fast | Approximate |

### Optuna (Recommended)

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = XGBClassifier(**params, random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=10, verbose=False)

    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 12. Step 11: Model Interpretation & Explainability

### Why Explainability Matters

- **Trust**: Stakeholders need to understand why predictions are made
- **Debugging**: Find if model is using spurious correlations
- **Compliance**: Regulations (GDPR, Fair Lending) require explanations
- **Improvement**: Understand what features to engineer next

### Methods

| Method | Scope | Type | Works With |
|--------|-------|------|-----------|
| **Feature Importance** | Global | Model-specific | Tree models |
| **Permutation Importance** | Global | Model-agnostic | Any model |
| **SHAP** | Global + Local | Model-agnostic | Any model |
| **LIME** | Local | Model-agnostic | Any model |
| **Partial Dependence Plots** | Global | Model-agnostic | Any model |
| **Coefficients** | Global | Model-specific | Linear models |

### SHAP (Best Practice)

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global: which features matter most overall?
shap.summary_plot(shap_values, X_test)

# Local: why did THIS customer get this prediction?
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Dependence: how does one feature affect predictions?
shap.dependence_plot('tenure_months', shap_values, X_test)
```

### Feature Importance: Why It Matters and How to Use It

#### Why Feature Importance Is Critical

Feature importance is not just a "nice to have" — it is one of the most valuable outputs of your entire modeling process. Here is why:

| Reason | Explanation |
|--------|-------------|
| **Validates your model** | If the top features make domain sense, the model is learning real patterns — not noise |
| **Detects data leakage** | If a feature is suspiciously dominant (>50% importance), it might be leaking the target |
| **Guides feature engineering** | Shows which features to invest more effort in (interactions, transformations) |
| **Enables feature pruning** | Remove low-importance features to simplify the model without losing performance |
| **Builds stakeholder trust** | Business users trust a model more when they understand what drives predictions |
| **Debugging poor performance** | If important features are unexpected, something is wrong with your data or pipeline |
| **Informs data collection** | Tells you which data sources are most valuable to maintain and improve |
| **Regulatory compliance** | Many industries require you to explain what factors influence decisions |

#### Types of Feature Importance

**1. Built-in (Model-Specific) Feature Importance**

Tree-based models compute importance during training. Two common methods:

```python
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Train model
model = XGBClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# Plot top 20 features
plt.figure(figsize=(10, 8))
importance.head(20).plot(kind='barh')
plt.title('XGBoost Feature Importance (Top 20)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Print importance values
print("Top 10 Most Important Features:")
print(importance.head(10))
print(f"\nBottom 10 (candidates for removal):")
print(importance.tail(10))
```

**XGBoost importance types:**

```python
from xgboost import plot_importance

# 'weight' — number of times a feature is used to split (default)
# 'gain' — average improvement in loss when the feature is used (RECOMMENDED)
# 'cover' — average number of samples affected by splits on this feature

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

plot_importance(model, ax=axes[0], importance_type='weight', max_num_features=15)
axes[0].set_title('Importance by Weight (split count)')

plot_importance(model, ax=axes[1], importance_type='gain', max_num_features=15)
axes[1].set_title('Importance by Gain (BEST)')

plot_importance(model, ax=axes[2], importance_type='cover', max_num_features=15)
axes[2].set_title('Importance by Cover (sample count)')

plt.tight_layout()
plt.show()
```

| Importance Type | What It Measures | Best For |
|----------------|-----------------|----------|
| **Weight** | How many times a feature is used in splits | Quick overview |
| **Gain** | Average loss reduction when feature is used | Best overall measure |
| **Cover** | Average number of samples in splits using this feature | Understanding reach |

**2. Permutation Importance (Model-Agnostic)**

Works with ANY model. Measures how much performance drops when a feature's values are randomly shuffled.

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance on TEST set (not train!)
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,           # Shuffle 10 times for stability
    random_state=42,
    scoring='roc_auc'       # Use your primary metric
)

# Create sorted dataframe
perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
}).sort_values('Importance Mean', ascending=False)

print("Permutation Importance (Top 15):")
print(perm_df.head(15).to_string(index=False))

# Plot with error bars
plt.figure(figsize=(10, 8))
top_n = 20
plt.barh(
    range(top_n),
    perm_df['Importance Mean'].head(top_n),
    xerr=perm_df['Importance Std'].head(top_n),
    align='center'
)
plt.yticks(range(top_n), perm_df['Feature'].head(top_n))
plt.xlabel('Mean Importance (AUC drop when shuffled)')
plt.title('Permutation Importance (Top 20)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Why permutation importance is better than built-in importance:**

| Aspect | Built-in (Gain/Weight) | Permutation Importance |
|--------|----------------------|----------------------|
| Works with any model | No (tree models only) | Yes |
| Affected by correlated features | Yes (splits importance) | Yes (but less so) |
| Computed on test data | No (uses training splits) | Yes (unbiased) |
| Accounts for feature interactions | Partially | Yes |
| Speed | Instant (already computed) | Slower (re-evaluates model) |

**3. SHAP Feature Importance (Gold Standard)**

SHAP gives you both global importance AND the direction of effect (positive/negative).

```python
import shap

# For tree models (fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance: mean absolute SHAP value per feature
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Detailed: shows direction of effect (red = high value, blue = low value)
shap.summary_plot(shap_values, X_test)

# Get numerical importance values
shap_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean |SHAP|', ascending=False)

print("SHAP Feature Importance:")
print(shap_importance.head(15).to_string(index=False))
```

**Why SHAP is the gold standard:**

- Shows **direction**: "Higher V14 values → more likely fraud" (not just "V14 is important")
- Shows **magnitude per prediction**: Explains individual decisions, not just averages
- **Mathematically grounded**: Based on Shapley values from game theory — fair attribution
- **Consistent**: If a feature contributes more to a prediction, it always gets a higher SHAP value
- Works with **any model** (TreeExplainer for trees, KernelExplainer for others)

#### Comparing All Three Methods Side by Side

```python
# Create a comparison dataframe
comparison = pd.DataFrame({
    'Feature': X_test.columns,
    'Built-in (Gain)': model.feature_importances_,
    'Permutation': perm_importance.importances_mean,
    'SHAP': np.abs(shap_values).mean(axis=0)
})

# Rank each method
for col in ['Built-in (Gain)', 'Permutation', 'SHAP']:
    comparison[f'{col}_Rank'] = comparison[col].rank(ascending=False).astype(int)

# Sort by SHAP (most reliable)
comparison = comparison.sort_values('SHAP', ascending=False)

print("Feature Importance Comparison (Top 15):")
print(comparison[['Feature', 'Built-in (Gain)_Rank', 'Permutation_Rank', 'SHAP_Rank']].head(15).to_string(index=False))

# If rankings disagree significantly, investigate why:
# - Correlated features split importance differently across methods
# - Built-in importance can be biased toward high-cardinality features
# - Permutation importance can underestimate correlated features
```

#### How to Interpret Feature Importance Results

**Scenario 1: Top feature has >50% importance**

```
Red flag! Possible data leakage.
Ask: "Would this feature be available at prediction time?"
Ask: "Is this feature derived from the target?"
Example: 'days_since_churn' being top feature for churn prediction = LEAKAGE
```

**Scenario 2: Many features have near-zero importance**

```
These features are noise. Remove them:
- Speeds up training and prediction
- Reduces overfitting
- Simplifies the model

threshold = 0.005  # Features below 0.5% importance
low_importance = importance[importance < threshold].index.tolist()
X_train_reduced = X_train.drop(columns=low_importance)
```

**Scenario 3: Importance rankings differ between methods**

```
This usually means features are correlated.
Example: Feature A and Feature B are 0.9 correlated.
- Built-in importance: splits between them randomly
- Permutation: shuffling one doesn't hurt much (the other compensates)
- SHAP: distributes fairly between correlated features

Solution: Check correlation, consider dropping one of the pair.
```

**Scenario 4: A feature you expected to be important is not**

```
Possible reasons:
1. The feature has low variance (almost constant)
2. The feature is redundant (another feature captures the same info)
3. The relationship is non-linear and the model can't capture it
4. The feature needs transformation (log, binning, interaction)
5. There's a data quality issue (too many nulls, wrong encoding)
```

#### Feature Importance for Linear Models

For linear models, coefficients serve as feature importance — but only after scaling:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# MUST scale first — otherwise coefficients are not comparable
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train logistic regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Coefficients = feature importance (after scaling)
coef_importance = pd.Series(
    np.abs(lr.coef_[0]),
    index=X_train.columns
).sort_values(ascending=False)

# Positive coefficient = increases probability of positive class
# Negative coefficient = decreases probability of positive class
coef_direction = pd.Series(
    lr.coef_[0],
    index=X_train.columns
).sort_values()

print("Logistic Regression Feature Importance:")
print(coef_importance.head(10))

print("\nFeature Direction (positive = increases fraud probability):")
print(coef_direction.tail(5))  # Top positive
print(coef_direction.head(5))  # Top negative
```

#### Practical Workflow: Feature Importance After Training

```python
# Complete workflow: train → importance → prune → retrain

# 1. Train initial model with all features
model = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 2. Get importance
importance = pd.Series(model.feature_importances_, index=X_train.columns)

# 3. Identify features to remove (below 1% importance)
low_importance_features = importance[importance < 0.01].index.tolist()
print(f"Features to remove ({len(low_importance_features)}): {low_importance_features}")

# 4. Retrain without low-importance features
X_train_reduced = X_train.drop(columns=low_importance_features)
X_test_reduced = X_test.drop(columns=low_importance_features)

model_reduced = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
model_reduced.fit(X_train_reduced, y_train)

# 5. Compare performance
from sklearn.metrics import roc_auc_score
auc_full = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
auc_reduced = roc_auc_score(y_test, model_reduced.predict_proba(X_test_reduced)[:, 1])

print(f"\nFull model ({X_train.shape[1]} features): AUC = {auc_full:.4f}")
print(f"Reduced model ({X_train_reduced.shape[1]} features): AUC = {auc_reduced:.4f}")
print(f"Performance change: {auc_reduced - auc_full:+.4f}")

# If AUC barely changes (< 0.005 drop), keep the simpler model!
```

#### Key Takeaways on Feature Importance

1. **Always check feature importance after training** — it is your primary debugging and validation tool
2. **Use SHAP for final analysis** — it is the most reliable and informative method
3. **Use permutation importance on the test set** — built-in importance can be misleading for correlated features
4. **Suspicious top features = investigate immediately** — could be leakage
5. **Low-importance features can usually be removed** — simpler models generalize better
6. **Feature importance informs your next iteration** — engineer more features from the important ones, drop the useless ones
7. **Compare multiple methods** — if they agree, you can be confident; if they disagree, dig deeper
8. **Domain knowledge is the final judge** — if the model says "customer_id" is important, something is wrong regardless of what the numbers say

---

## 13. Step 12: Model Validation & Testing

### Validation Checks Before Deployment

| Check | What | How |
|-------|------|-----|
| **Performance stability** | Consistent across data slices | Evaluate on subgroups |
| **Temporal stability** | Works on recent data | Test on latest time period |
| **Fairness** | No bias against protected groups | Check metrics by group |
| **Robustness** | Handles edge cases | Test with missing values, outliers |
| **Calibration** | Predicted probabilities are accurate | Calibration plot |
| **Overfitting check** | Train vs test gap is small | Compare train/test metrics |
| **Data leakage check** | No future info in features | Review feature engineering |

### Overfitting Detection

```python
# If train AUC >> test AUC, you're overfitting
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")
print(f"Gap:       {train_auc - test_auc:.4f}")

# Gap > 0.05 suggests overfitting
# Solutions: more data, regularization, simpler model, fewer features
```

### Calibration Check

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

# Plot: if well-calibrated, points should be on the diagonal
# If not, use CalibratedClassifierCV to fix
```

### Fairness Check

```python
# Check if model performs equally across groups
for group in ['male', 'female']:
    mask = df_test['gender'] == group
    group_auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
    print(f"AUC for {group}: {group_auc:.4f}")

# Large differences indicate bias
```

---

## 14. Step 13: Deployment

### Deployment Decision Matrix

| Question | If Yes → |
|----------|----------|
| Need instant predictions? | Real-time endpoint |
| Scoring millions of records? | Batch transform |
| Traffic is sporadic? | Serverless endpoint |
| Need to explain each prediction? | Add SHAP to inference |
| Multiple models to serve? | Multi-model endpoint |
| Need A/B testing? | Traffic splitting |

### Pre-Deployment Checklist

- [ ] Model artifact saved and versioned
- [ ] Inference code tested locally
- [ ] Input validation in inference code
- [ ] Error handling for edge cases
- [ ] Latency tested and acceptable
- [ ] Memory usage within instance limits
- [ ] Rollback plan documented
- [ ] Monitoring configured
- [ ] Alerts set up

---

## 15. Step 14: Monitoring & Maintenance

### What Can Go Wrong in Production

| Problem | Cause | Detection |
|---------|-------|-----------|
| **Data drift** | Input data distribution changes | Compare feature distributions |
| **Concept drift** | Relationship between features and target changes | Monitor prediction accuracy |
| **Feature unavailability** | Upstream data source breaks | Monitor for nulls/errors |
| **Model staleness** | Model trained on old data | Track time since last training |
| **Performance degradation** | Gradual accuracy decline | Monitor metrics over time |

### Monitoring Metrics

```python
# Track these over time:
monitoring_metrics = {
    'prediction_volume': count_predictions_per_hour,
    'prediction_distribution': mean_prediction_probability,
    'feature_drift': psi_score_per_feature,  # Population Stability Index
    'latency_p99': endpoint_latency_99th_percentile,
    'error_rate': failed_predictions / total_predictions,
    'actual_vs_predicted': accuracy_on_labeled_data  # When labels arrive
}
```

### Population Stability Index (PSI)

Measures how much a distribution has shifted:

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.1 | No significant change |
| 0.1 - 0.25 | Moderate change, investigate |
| > 0.25 | Significant change, retrain |

### Retraining Triggers

| Trigger | When to Use |
|---------|-------------|
| **Scheduled** (weekly/monthly) | Stable environments |
| **Performance-based** (AUC drops below threshold) | When you have ground truth |
| **Drift-based** (PSI > 0.25) | When ground truth is delayed |
| **Data volume** (N new records since last train) | Growing datasets |

---

## 16. Algorithm Reference (When to Use What)

### Quick Selection Guide

| Scenario | First Try | If Not Good Enough |
|----------|-----------|-------------------|
| Tabular, <100K rows | Random Forest | XGBoost |
| Tabular, >100K rows | LightGBM | XGBoost + tuning |
| Need interpretability | Logistic Regression | Decision Tree + SHAP |
| Time series | Prophet / ARIMA | LSTM / Transformer |
| Text classification | TF-IDF + LogReg | BERT / fine-tuned LLM |
| Image classification | Pre-trained CNN (transfer learning) | Fine-tune ResNet/EfficientNet |
| Anomaly detection | Isolation Forest | Autoencoder |
| Recommendation | Collaborative filtering | Neural CF / Two-tower |
| Clustering | K-Means | DBSCAN / Gaussian Mixture |

### XGBoost Hyperparameter Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `max_depth` | 3-10 | Higher = more complex, risk overfit |
| `learning_rate` (eta) | 0.01-0.3 | Lower = more robust, needs more trees |
| `n_estimators` | 50-1000 | More = better (with early stopping) |
| `subsample` | 0.5-1.0 | Lower = more regularization |
| `colsample_bytree` | 0.5-1.0 | Lower = more regularization |
| `min_child_weight` | 1-10 | Higher = more conservative |
| `gamma` | 0-5 | Higher = more pruning |
| `reg_alpha` (L1) | 0-1 | Feature selection |
| `reg_lambda` (L2) | 0-1 | Weight regularization |

---

## 17. Metrics Reference (Complete)

### Classification Metrics Summary

| Metric | Perfect Score | Random Score | Handles Imbalance? |
|--------|:---:|:---:|:---:|
| Accuracy | 1.0 | ~majority class % | ❌ |
| Precision | 1.0 | ~positive rate | ✓ |
| Recall | 1.0 | ~positive rate | ✓ |
| F1 | 1.0 | ~2×pos_rate/(1+pos_rate) | ✓ |
| AUC-ROC | 1.0 | 0.5 | Partially |
| AUC-PR | 1.0 | ~positive rate | ✓ |
| Log Loss | 0.0 | ~0.693 | ✓ |
| MCC | 1.0 | 0.0 | ✓ |

### Regression Metrics Summary

| Metric | Perfect Score | Interpretation |
|--------|:---:|-------------|
| MAE | 0 | Average absolute error in target units |
| RMSE | 0 | Penalizes large errors more |
| R² | 1.0 | % of variance explained (0.8 = 80%) |
| MAPE | 0% | Average % error |

---

## 18. Common Pitfalls & How to Avoid Them

| Pitfall | What Goes Wrong | How to Avoid |
|---------|----------------|-------------|
| **Data leakage** | Future info in features | Split BEFORE feature engineering |
| **Target leakage** | Feature derived from target | Review each feature's source |
| **Overfitting** | Great train, bad test | Regularization, cross-validation |
| **Underfitting** | Bad train AND test | More features, complex model |
| **Class imbalance ignored** | Model predicts majority class | Use appropriate metrics, resampling |
| **Wrong metric** | Optimizing wrong thing | Match metric to business cost |
| **Not enough data** | Model can't learn patterns | More data, simpler model, transfer learning |
| **Multicollinearity** | Unstable coefficients | VIF check, drop correlated features |
| **Survivorship bias** | Only see successful cases | Include all historical data |
| **Look-ahead bias** | Using future data in features | Strict temporal ordering |
| **Distribution shift** | Train ≠ production data | Monitor drift, retrain regularly |
| **Feature not available at inference** | Can't compute feature in real-time | Verify all features available at prediction time |

### The Golden Rules

1. **Start simple** — complex models aren't always better
2. **Trust your validation** — if test metrics are bad, the model is bad
3. **Features > algorithms** — better features beat better algorithms
4. **More data > more complexity** — get more data before trying fancier models
5. **Monitor everything** — models degrade, always
6. **Document decisions** — future you will thank present you
7. **Iterate fast** — quick experiments beat perfect first attempts

---

*End of Data Science Bible*
