# Module 9: Ensemble Methods & Model Tuning

---

## 1. CONCEPT

This module pushes from "a model that works" to "a model that's actually competitive" — the gap where most of the visible skill difference between an intermediate and strong practitioner shows up.

**Bagging vs. boosting:**
- **Bagging** (Bootstrap Aggregating) — trains many models independently and in parallel on bootstrapped (randomly resampled with replacement) subsets of data, then averages/votes their predictions. Reduces variance. Random Forest is the canonical example.
- **Boosting** — trains models sequentially, where each new model focuses on correcting the errors of the previous ones. Reduces bias, often achieves higher accuracy than bagging, but is more prone to overfitting if not carefully tuned. XGBoost, LightGBM, CatBoost are the modern standard.

**Why ensembles usually beat single models:** a single decision tree is high-variance (small data changes produce very different trees); averaging many such trees (bagging) or sequentially correcting errors (boosting) systematically outperforms any single tree on most real tabular data problems — this is why gradient-boosted trees are the default strong baseline for tabular ML in industry.

**Gradient boosting libraries:**
- **XGBoost** — the long-standing industry standard; extensive tuning options, strong documentation, wide adoption
- **LightGBM** — faster training on large datasets via a different tree-growth strategy (leaf-wise rather than level-wise), often the practical choice for large tabular data
- **CatBoost** — handles categorical features natively without manual encoding, and defaults tend to perform well with less tuning — a good choice when categorical features dominate the dataset

**Hyperparameter tuning:**
- **Grid search** — exhaustively tries every combination in a predefined parameter grid; thorough but computationally expensive, scales poorly with more parameters
- **Random search** — samples random combinations from parameter distributions; often finds comparably good results to grid search with far fewer trials
- **Bayesian optimization** — uses the results of previous trials to intelligently choose the next parameters to try, typically the most sample-efficient approach for expensive-to-train models

**Model interpretability for ensembles:**
- **Feature importance** — how much each feature contributed to reducing impurity/error across the ensemble; useful but can be biased toward high-cardinality features
- **SHAP (SHapley Additive exPlanations)** — grounded in game theory, attributes each prediction's outcome to individual feature contributions in a mathematically consistent way; works at both the global level (overall feature importance) and the individual-prediction level (why did the model predict *this specific* outcome for *this specific* row)
- **Partial dependence plots (PDPs)** — show how the model's average prediction changes as one feature varies, holding others constant

---

## 2. WHY IT MATTERS

Ensembles and their tuning are where measurable, defensible performance gains happen — the difference between a portfolio project that says "I built a model" and one that says "I built a model, then improved it from F1 0.68 to 0.81 through systematic tuning and feature work, and can explain every individual prediction it makes." That second framing is what hiring managers are actually screening for.

SHAP specifically has become close to a required skill in regulated industries (lending, insurance, healthcare) where "the model said so" is not a legally or ethically acceptable explanation — being able to produce a defensible, individual-level explanation for a specific prediction is often a hard requirement, not a nice-to-have.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: XGBoost + tuning + SHAP ---
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

# Grid search
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "n_estimators": [100, 300],
}
grid_search = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Randomized search (more efficient for larger grids)
from scipy.stats import uniform, randint
param_distributions = {
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.3),
    "n_estimators": randint(100, 500),
}
random_search = RandomizedSearchCV(
    model, param_distributions, n_iter=50, scoring="f1", cv=5, n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

# SHAP explanations
import shap
best_model = grid_search.best_estimator_
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)              # global feature importance
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])  # single prediction
```

```r
# --- R: tidymodels tuning + xgboost ---
library(tidymodels)

xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), learn_rate = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_grid <- grid_regular(
  trees(), tree_depth(), learn_rate(),
  levels = 3
)

xgb_workflow <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(target ~ .)

cv_folds <- vfold_cv(train_data, v = 5)

tuning_results <- tune_grid(
  xgb_workflow, resamples = cv_folds, grid = xgb_grid,
  metrics = metric_set(f_meas)
)

best_params <- select_best(tuning_results, metric = "f_meas")
final_model <- finalize_workflow(xgb_workflow, best_params) %>%
  fit(data = train_data)
```

---

## 4. WORKED EXAMPLE

**Scenario:** A bank needs a credit-risk model that's both high-performing and explainable to regulators — a legally-mandated constraint where "black box, trust me" isn't acceptable, especially for individual loan denials.

```python
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint

loans = pd.read_csv("loan_applications.csv")
X = loans.drop(columns=["loan_id", "defaulted"])
y = loans["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Step 1: Tune the model for strong performance ---
model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
param_distributions = {
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.01, 0.2),
    "n_estimators": randint(100, 400),
    "subsample": uniform(0.7, 0.3),
}
search = RandomizedSearchCV(
    model, param_distributions, n_iter=30, scoring="roc_auc",
    cv=5, n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

print(classification_report(y_test, best_model.predict(X_test)))

# --- Step 2: Build global explainability for the regulator ---
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
# Global finding: debt_to_income_ratio, credit_score, and payment_history
# are the top 3 drivers of predicted default risk across the whole portfolio
# — this satisfies the regulator's need to understand overall model behavior

# --- Step 3: Build individual-level explanations for specific denials ---
# A regulator (or an applicant, under certain lending laws) may ask:
# "why was THIS SPECIFIC application denied?"
denied_applications = X_test[best_model.predict(X_test) == 1].head(3)

for idx in denied_applications.index:
    row_position = X_test.index.get_loc(idx)
    print(f"\n--- Application {idx} ---")
    contributions = pd.Series(
        shap_values[row_position], index=X_test.columns
    ).sort_values(key=abs, ascending=False)
    print(contributions.head(5))
    # Output: a ranked list of exactly which features pushed this specific
    # application toward denial, and by how much — auditable, defensible,
    # and specific to this one applicant, not just a global statement
```

This is what "explainable to regulators" means concretely in practice: global SHAP summary for overall model behavior, plus individual SHAP explanations for any specific decision that gets challenged. Neither piece alone satisfies a real regulatory requirement — both are typically needed.

---

## 5. COMMON MISTAKES

- **Tuning hyperparameters using the test set instead of cross-validation.** This is a subtle form of leakage — repeatedly checking test performance while adjusting hyperparameters means you're indirectly fitting to the test set, inflating your final reported performance. Always tune using cross-validation on the training set only, and touch the test set exactly once, at the very end.
- **Using default hyperparameters and assuming that's "good enough."** Default settings are reasonable starting points, not optimized ones — the gap between default and tuned performance is often the single largest lever available after feature engineering.
- **Over-relying on global feature importance for individual decisions.** A feature that's globally unimportant on average can still be the dominant driver for a specific individual prediction — global importance and SHAP's per-prediction explanations answer genuinely different questions, and conflating them (especially in regulated contexts) is a real compliance risk, not just a technical nitpick.
- **Running an exhaustive grid search over a large parameter space without considering the computational cost.** A grid with 5 parameters × 5 values each is 3,125 combinations × cross-validation folds — often computationally infeasible. Random or Bayesian search scales far better and typically finds comparably good results with a fraction of the compute.
- **Treating SHAP summary plots as proof of causation.** SHAP explains what the *model* is doing, which reflects correlational patterns the model learned from historical data — not necessarily true causal drivers of the outcome. This distinction matters enormously in regulated/high-stakes contexts and is worth stating explicitly in any report using SHAP.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Bagging vs. Boosting**
Explain in your own words why Random Forest (bagging) tends to be more resistant to overfitting than a single deep decision tree, and why XGBoost (boosting) can achieve higher accuracy but requires more careful tuning to avoid overfitting than Random Forest does.

<details>
<summary>Sample Solution</summary>

Random Forest averages predictions across many trees trained on different bootstrapped samples with random feature subsets — the errors of individual trees, being somewhat independent, tend to cancel out in the average, reducing variance without much effort. Boosting instead trains trees sequentially, each one specifically correcting the previous ensemble's errors — this lets it fit the training data's patterns very closely (higher potential accuracy) but also means it can start fitting to noise in later boosting rounds if the number of trees, learning rate, and tree depth aren't carefully controlled, which is why boosting typically needs more deliberate tuning (early stopping, learning rate, regularization) than bagging to avoid overfitting.
</details>

---

**Exercise 2 — Tuning Strategy Selection**
You have an XGBoost model with 6 hyperparameters to tune, each with a reasonable range of values, and training takes 2 minutes per run. Would you choose grid search, random search, or Bayesian optimization? Justify your choice given the computational constraint.

<details>
<summary>Hint</summary>
Calculate roughly how many combinations a grid search over 6 parameters would require even with just 4 values each, multiply by the 2-minute training time and by cross-validation folds, and compare that to a fixed budget of, say, 50 random or Bayesian trials.
</details>

<details>
<summary>Sample Solution</summary>

Grid search over 6 parameters with even just 4 values each is 4^6 = 4,096 combinations — at 2 minutes each with 5-fold CV, that's computationally infeasible (over 68,000 minutes). Random search or Bayesian optimization with a fixed budget of, say, 50–100 trials is far more practical and typically finds results close to the grid search optimum. Bayesian optimization is the stronger choice here specifically because each training run is expensive (2 minutes) — its sample efficiency (using prior results to choose smarter next trials) pays off more when trials are costly, compared to random search's purely random sampling.
</details>

---

**Exercise 3 — Full Tuned Ensemble with Explainability (Applied)**
Using the `data_jobs` dataset (or a classification/regression problem from a previous module), train a baseline XGBoost/LightGBM model with default parameters, then tune it using randomized search with cross-validation, and report the performance improvement. Generate a global SHAP summary plot and pick 2 individual predictions to explain at the row level. Write a short paragraph distinguishing what the global plot tells you versus what the individual explanations tell you — this distinction is exactly what Module 9's worked example hinges on, and demonstrating you understand it clearly is a strong interview and portfolio signal.
