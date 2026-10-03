# Module 8: Feature Engineering

---

## 1. CONCEPT

Feature engineering is where most real-world model performance gains actually come from — more consistently than algorithm choice. This module also covers **data leakage**, arguably the single most damaging and most common mistake in applied machine learning, in dedicated depth.

**Encoding categorical variables:**
- **One-hot encoding** — creates a binary column per category; safe default for nominal (unordered) categories, but can explode dimensionality with high-cardinality features
- **Label encoding** — assigns an arbitrary integer per category; appropriate for tree-based models (which don't assume numeric ordering means anything) but dangerous for linear models (which *will* assume the encoded numbers have ordinal meaning that doesn't exist)
- **Target/mean encoding** — replaces a category with the mean target value for that category; powerful for high-cardinality features, but carries a serious leakage risk if not computed carefully (see below)

**Scaling/normalization:**
- Matters for **distance-based models** (k-NN, SVM, k-means) and **regularized regression** (Ridge/Lasso, since they penalize coefficient magnitude)
- Largely irrelevant for **tree-based models** (decision trees, random forests, gradient boosting), which split on thresholds regardless of scale

**Feature creation:**
- **Interaction features** — capturing that the effect of one variable depends on the level of another (e.g., `discount_pct × is_holiday_season`)
- **Polynomial features** — capturing non-linear relationships within linear model frameworks
- **Domain-driven features** — often the highest-value feature engineering, requiring actual subject-matter understanding rather than mechanical transformation (this is where your stats/domain background becomes a real edge)

**Feature selection:**
- **Filter methods** — select features based on statistical properties alone (correlation, chi-square) before any model is trained
- **Wrapper methods** — iteratively add/remove features based on model performance (e.g., recursive feature elimination) — more accurate but computationally expensive
- **Embedded methods** — feature selection happens as part of model training itself (Lasso's coefficient shrinkage to zero, tree-based feature importances)

**Data leakage — the dedicated focus of this module:**
Leakage happens when information that wouldn't be available at prediction time sneaks into training, producing a model that looks excellent in testing and fails (or actively misleads) in production. Common forms:
- A feature that's a **direct proxy for the target** (a "cancellation_request_date" column when predicting churn)
- **Target encoding computed on the full dataset** before splitting, letting test-set target information leak into training-set feature values
- **Temporal leakage** — using future information to predict a past/present outcome (common in time series, covered further in Module 11)
- **Preprocessing fit on the full dataset** (e.g., scaling or imputing using statistics computed across train + test together) before splitting

---

## 2. WHY IT MATTERS

A leaky model is worse than a bad model, because a bad model fails visibly while a leaky model fails invisibly — it looks great in every offline evaluation and then either doesn't work in production or, worse, confidently produces wrong business decisions that nobody catches because the reported metrics all looked strong. Diagnosing and preventing leakage is one of the clearest signals of an experienced practitioner versus a beginner, and it's tested constantly in DS interviews via exactly this kind of "why is this model suspiciously good" scenario.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: scikit-learn (Pipeline enforces correct fit/transform order) ---
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# CRITICAL: split BEFORE any fitting of encoders/scalers/imputers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = ["region", "category"]
numeric_cols = ["price", "quantity"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# Pipeline ensures preprocessor is fit ONLY on training data,
# then applied (not re-fit) to test data — this structurally prevents leakage
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier())
])
pipeline.fit(X_train, y_train)          # fit_transform on train
pipeline.predict(X_test)                # transform only on test — never re-fit

# Target encoding done SAFELY (fit only on training fold)
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols=["high_cardinality_category"])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)  # transform only, using train-derived stats

# Feature importance (embedded feature selection)
model.fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False)
```

```r
# --- R: recipes (tidymodels) — same fit-on-train-only discipline ---
library(recipes)

rec <- recipe(target ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%      # one-hot / dummy encoding
  step_normalize(all_numeric_predictors()) %>%   # scaling
  step_impute_median(all_numeric_predictors())   # imputation

prepped <- prep(rec, training = train_data)   # statistics computed on TRAIN only
train_processed <- bake(prepped, new_data = train_data)
test_processed  <- bake(prepped, new_data = test_data)  # applies train-derived stats
```

---

## 4. WORKED EXAMPLE

**Scenario:** A churn-prediction model is getting suspiciously good results (98% accuracy). Diagnose the leakage.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

customers = pd.read_csv("customer_churn.csv")
print(customers.columns.tolist())
# ['customer_id', 'tenure_months', 'monthly_charges', 'support_tickets',
#  'cancellation_request_date', 'plan_type', 'churned']

X = customers.drop(columns=["customer_id", "churned"])
y = customers["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
# Accuracy: 0.98 — suspiciously high for a churn problem, which is usually
# a hard, noisy prediction task in reality. Time to investigate.

# --- Step 1: Check feature importances — leakage often shows up as one
#     feature dominating implausibly ---
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False))
# Finding: 'cancellation_request_date' has importance ~0.91 — one feature
# is doing almost all the work, which is a major red flag

# --- Step 2: Investigate that feature directly ---
print(customers[["cancellation_request_date", "churned"]].head(10))
# Finding: cancellation_request_date is populated ONLY for customers who
# churned — it's not a predictor of churn, it's a DIRECT RECORD of churn.
# The model isn't predicting the future; it's reading the answer key.

# --- Step 3: Remove the leaky feature and retrain ---
X_clean = X.drop(columns=["cancellation_request_date"])
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)
model_clean = RandomForestClassifier(random_state=42)
model_clean.fit(X_train, y_train)
y_pred_clean = model_clean.predict(X_test)
print(f"Accuracy after removing leaky feature: {accuracy_score(y_test, y_pred_clean):.3f}")
# Accuracy: 0.74 — a large drop, and a far more believable, trustworthy number
# for a genuinely hard prediction problem. This is the model that will
# actually work in production, unlike the 98% version.
```

The pattern to internalize: **suspiciously good results are a signal to investigate, not celebrate.** Feature importance is your first diagnostic tool for finding the culprit.

---

## 5. COMMON MISTAKES

- **Fitting a scaler, encoder, or imputer on the full dataset before splitting.** Even something as seemingly harmless as `StandardScaler().fit(X)` before `train_test_split` leaks test-set distributional information into the training process — always split first, fit preprocessing on train only, then transform (not re-fit) test.
- **Not investigating suspiciously good results.** As shown in the worked example — a beginner's instinct is often to be pleased by a 98% accuracy score; the correct instinct is suspicion, especially for problems known to be genuinely hard (churn, fraud, stock prediction).
- **Including a feature that's only populated after the outcome occurs.** `cancellation_request_date` is a clean example, but subtler versions exist constantly — a "days_since_last_login" feature computed using a snapshot date *after* the churn event has already happened is functionally the same leakage, just harder to spot.
- **Applying target encoding across the full dataset instead of per-fold.** Target encoding computed globally (rather than fit only on training data, or better, computed with cross-validation folds) leaks target information directly into feature values — a subtle but common leakage source even among practitioners who know to split before scaling.
- **Trusting cross-validation scores when preprocessing was done before the CV loop.** If scaling/encoding/imputation happens once on the whole dataset before cross-validation splits are made, every fold is contaminated identically — CV scores will look artificially stable and optimistic. Preprocessing must happen inside each fold, which is exactly what `Pipeline` (Python) and `recipes`/`workflows` (R) are designed to enforce automatically.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Leakage Spotting**
For each feature below, in a model predicting whether a loan will default, identify whether it's safe to use or a leakage risk, and explain why:
a) `credit_score` (recorded at loan application time)
b) `days_late_on_first_payment` (only populated after the loan has been active for a month)
c) `annual_income` (self-reported at application)
d) `collections_agency_assigned` (only populated for loans that have already defaulted)

<details>
<summary>Sample Solution</summary>

a) Safe — available at prediction time (loan origination).
b) Leakage risk — depends on when the prediction is meant to be made. If predicting default *at origination*, this feature wouldn't exist yet and is leakage. If predicting default risk *after* the first month has passed, it could legitimately be used, but the model's use case and timing must be explicitly scoped.
c) Safe — available at application time (though worth separately checking for data quality issues, since self-reported income is often noisy).
d) Leakage — this is a near-direct proxy for the target itself, populated only after default has already occurred, exactly like `cancellation_request_date` in the worked example.
</details>

---

**Exercise 2 — Pipeline Construction**
Given a dataset with 3 numeric columns (needing scaling) and 2 categorical columns (needing one-hot encoding), write a scikit-learn `Pipeline` + `ColumnTransformer` that correctly handles both, ensures no leakage between train and test, and fits a `LogisticRegression` model at the end.

<details>
<summary>Hint</summary>
Use `ColumnTransformer` to apply different preprocessing to different column subsets, wrap it in a `Pipeline` alongside the model, and call `.fit()` only on training data — the pipeline structure itself prevents you from accidentally fitting preprocessing on test data.
</details>

<details>
<summary>Solution</summary>

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

numeric_cols = ["col1", "col2", "col3"]
categorical_cols = ["cat1", "cat2"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```
</details>

---

**Exercise 3 — Full Leakage Audit (Applied)**
Take a model you've already built in a previous module's exercises (e.g., the `data_jobs` regression or classification work). Deliberately audit it for leakage: check every feature and ask "would this be known at the actual moment of prediction, in the real world?" Document your audit as a table (feature name, leakage risk: yes/no, reasoning), and if you find any leaky features, retrain without them and compare performance — the delta itself is informative and worth including in your portfolio write-up.
