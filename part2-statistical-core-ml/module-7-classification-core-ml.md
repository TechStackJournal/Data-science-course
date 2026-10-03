# Module 7: Classification & Core ML Algorithms

---

## 1. CONCEPT

This module covers the standard supervised classification toolkit and — more importantly — how to evaluate classifiers correctly, which is where most real-world classification work actually lives or dies.

**Core algorithms:**
- **Decision trees** — split data on feature thresholds to maximize class purity at each node; intuitive and interpretable, but prone to overfitting if left unconstrained (depth, min samples per leaf)
- **Random forests** — an ensemble of decision trees trained on bootstrapped samples with random feature subsets; trades some interpretability for much better generalization than a single tree
- **k-Nearest Neighbors (k-NN)** — classifies a point based on the majority class among its k closest neighbors; simple, but sensitive to feature scaling and struggles in high dimensions
- **Naive Bayes** — applies Bayes' theorem assuming feature independence (a strong, often-wrong assumption that works surprisingly well in practice, especially for text classification)
- **SVM (Support Vector Machines)** — finds the boundary that maximizes margin between classes; effective in high-dimensional spaces, less commonly the first choice now that gradient-boosted trees usually outperform it on tabular data

**Evaluation metrics — the actual crux of this module:**
- **Accuracy** — proportion correct; misleading on imbalanced data (a model predicting "not fraud" every time can be 99.7% accurate and useless)
- **Precision** — of everything predicted positive, what fraction actually is positive (relevant when false positives are costly)
- **Recall** — of everything actually positive, what fraction was caught (relevant when false negatives are costly)
- **F1 score** — harmonic mean of precision and recall, useful when you need a single balanced metric
- **ROC-AUC** — measures ranking quality across all thresholds; less sensitive to class imbalance than accuracy, but still worth pairing with precision-recall analysis for heavily imbalanced problems

**Confusion matrix** — the foundation all of the above metrics are computed from; always look at the raw matrix before trusting any single summary metric, since two very different error profiles can produce the same F1 score.

**Handling imbalanced classes:**
- **Class weights** — penalize misclassifying the minority class more heavily during training
- **SMOTE** (Synthetic Minority Oversampling) — generates synthetic minority-class examples rather than simply duplicating existing ones
- **Threshold tuning** — the default 0.5 probability threshold is rarely optimal for imbalanced problems; tuning it against the actual business cost of false positives vs. false negatives is usually more impactful than any algorithm change

---

## 2. WHY IT MATTERS

The single most common way beginners get burned in classification isn't picking the wrong algorithm — it's picking the wrong metric. A model can look excellent on accuracy and be operationally useless, and this exact failure mode (high accuracy, zero business value) is one of the most frequently asked interview scenarios in data science precisely because it's such a common real mistake.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: scikit-learn ---
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  # stratify preserves class balance
)

model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))  # precision, recall, F1 in one call
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")

# Threshold tuning
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# SMOTE (imbalanced-learn library)
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

```r
# --- R: caret / tidymodels ---
library(tidymodels)

set.seed(42)
split <- initial_split(data, prop = 0.8, strata = target)  # stratified split
train_data <- training(split)
test_data <- testing(split)

rf_spec <- rand_forest(trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_fit <- rf_spec %>% fit(target ~ ., data = train_data)

predictions <- predict(rf_fit, test_data, type = "prob") %>%
  bind_cols(predict(rf_fit, test_data)) %>%
  bind_cols(test_data)

# Metrics
predictions %>% conf_mat(truth = target, estimate = .pred_class)
predictions %>% f_meas(truth = target, estimate = .pred_class)
predictions %>% roc_auc(truth = target, .pred_1)
```

---

## 4. WORKED EXAMPLE

**Scenario:** Build a fraud detection classifier where fraud is 0.3% of transactions. Demonstrate explicitly why accuracy is the wrong metric here.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve, f1_score
)

transactions = pd.read_csv("transactions.csv")
print(transactions["is_fraud"].value_counts(normalize=True))
# is_fraud: 0.997 non-fraud, 0.003 fraud

X = transactions.drop(columns="is_fraud")
y = transactions["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Step 1: Demonstrate the accuracy trap with a naive baseline ---
naive_predictions = pd.Series([0] * len(y_test))  # predict "not fraud" every time
naive_accuracy = accuracy_score(y_test, naive_predictions)
print(f"Naive 'always predict not-fraud' accuracy: {naive_accuracy:.3f}")
# Naive accuracy: 0.997 — a completely useless model still scores 99.7%

# --- Step 2: Train a real model, evaluate with the RIGHT metrics ---
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Focus on precision, recall, and F1 for the fraud (minority) class specifically —
# not the overall accuracy, which will still look high and tell you almost nothing

# --- Step 3: Tune the decision threshold against business cost ---
# Assume: a missed fraud costs $500 on average; a false fraud alarm costs
# $20 in customer friction/support time. These costs justify prioritizing
# recall over precision — better to over-flag than under-flag.
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

cost_per_missed_fraud = 500
cost_per_false_alarm = 20

best_threshold, lowest_cost = 0.5, float("inf")
for i, threshold in enumerate(thresholds):
    preds_at_threshold = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_at_threshold).ravel()
    total_cost = fn * cost_per_missed_fraud + fp * cost_per_false_alarm
    if total_cost < lowest_cost:
        lowest_cost, best_threshold = total_cost, threshold

print(f"Cost-optimal threshold: {best_threshold:.3f} (default is 0.5)")
```

This walks through the exact reasoning chain a real fraud team goes through: expose why the obvious metric is a trap, use the right metrics for the minority class, then go one step further and tune the threshold against actual business costs rather than accepting the default.

---

## 5. COMMON MISTAKES

- **Reporting accuracy as the headline metric on imbalanced data**, as demonstrated directly in the worked example — always check class balance first, and if it's skewed, lead with precision/recall/F1 or a cost-weighted metric instead.
- **Splitting train/test without stratification on imbalanced data.** A random (non-stratified) split can leave the test set with almost no minority-class examples by chance, making evaluation metrics unreliable — always use `stratify=y` (or the equivalent) when classes are imbalanced.
- **Applying SMOTE before the train/test split.** This leaks synthetic information derived from test-set-adjacent data into training, inflating apparent performance. SMOTE (and any resampling) must be applied only to the training set, after splitting.
- **Leaving the classification threshold at the default 0.5 without considering business cost.** As shown in the worked example — the "correct" threshold depends entirely on the relative cost of false positives vs. false negatives for the specific problem, not on a universal default.
- **Comparing models using a single metric without looking at the confusion matrix.** Two models can have identical F1 scores with very different error profiles (one erring toward false positives, the other toward false negatives) — which one is preferable depends on the business context, and you can't tell from F1 alone.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Metric Selection**
For each scenario, state whether precision or recall should be prioritized, and why:
a) A spam filter (false positive = a real email gets marked as spam and missed)
b) A cancer screening test (false negative = a real case is missed)
c) A churn-prediction model used to trigger a low-cost retention email

<details>
<summary>Sample Solution</summary>

a) Prioritize **precision** — a missed spam email is a minor annoyance, but a real, important email getting buried in spam is a much costlier false positive.
b) Prioritize **recall** — missing a real cancer case (false negative) is far more costly than a false alarm that leads to further (safe) testing.
c) Prioritize **recall**, leaning toward a lower threshold — since the retention email is low-cost, it's usually worth over-flagging potential churners rather than under-flagging, unless the email carries real cost (e.g., a large discount) in which case precision matters more.
</details>

---

**Exercise 2 — Confusion Matrix Reasoning**
Two fraud models both achieve F1 = 0.75. Model A's confusion matrix shows high false positives, low false negatives. Model B shows the reverse. Given that missed fraud costs far more than a false alarm, which model would you deploy, and why does F1 alone fail to distinguish them?

<details>
<summary>Sample Solution</summary>

Deploy **Model A** (high false positives, low false negatives) — it catches more actual fraud cases, which matters more given the stated cost asymmetry. F1 is the harmonic mean of precision and recall and can be identical for two models with very different precision/recall balances — it collapses a 2-dimensional tradeoff into a single number, losing exactly the information needed to make this cost-sensitive decision. This is why the confusion matrix (or a cost-weighted metric) should always be checked directly rather than relying on F1 alone.
</details>

---

**Exercise 3 — Full Imbalanced Classification Pipeline (Applied)**
Using a real or simulated imbalanced dataset (fraud, churn, or a rare-event dataset of your choice), build a full pipeline: stratified train/test split, a baseline "predict majority class" comparison to demonstrate the accuracy trap, a trained classifier with appropriate class weighting, a full confusion matrix and classification report, and a cost-based threshold tuning step similar to the worked example (define your own plausible cost assumptions and justify them). Write up your threshold decision as if presenting it to a business stakeholder who will ask "why not just use 0.5?"
