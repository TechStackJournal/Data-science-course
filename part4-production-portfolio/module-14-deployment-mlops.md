# Module 14: Model Deployment & MLOps Fundamentals

---

## 1. CONCEPT

This module covers the path from a notebook to a production system — the part of the lifecycle that most separates someone who can build a model from someone who can ship one that keeps working. Given your data engineering background, this module is where that experience becomes a direct, compounding advantage rather than a separate skill set.

**Serving models:**
- **Flask/FastAPI** — lightweight Python web frameworks used to wrap a trained model behind an HTTP endpoint; FastAPI is the more modern choice (built-in validation, automatic docs, async support) and increasingly the default for new ML services
- The core pattern: load the trained model once at startup, expose a `/predict` endpoint that accepts input, runs it through the same preprocessing pipeline used in training, and returns a prediction

**Containerization:**
- **Docker** packages the model, its dependencies, and the serving code into a reproducible image — solving the "works on my machine" problem that plagues ML deployment specifically, since ML environments have notoriously fragile dependency trees (specific library versions, CUDA versions for GPU models, etc.)

**Experiment tracking & model versioning:**
- **MLflow** logs parameters, metrics, and model artifacts across training runs, letting you compare experiments systematically and reproduce any specific run later — the ML equivalent of version control, applied to experiments and models rather than just code
- Model versioning matters because "which exact model is currently in production, trained on which data, with which hyperparameters" needs to be answerable with certainty, not reconstructed from memory during an incident

**Batch vs. real-time inference:**
- **Batch inference** — running predictions on a large set of records on a schedule (e.g., nightly churn scores for the whole customer base); simpler, higher throughput, appropriate when predictions don't need to be instantaneous
- **Real-time inference** — serving predictions on-demand via an API, typically with latency requirements (e.g., a fraud check that must return in under 100ms during checkout); more operationally complex, needed when the business process requires an immediate response

**Monitoring: drift and decay:**
- **Data drift** — the distribution of input features in production shifts away from what the model was trained on (e.g., a product redesign changes user behavior patterns)
- **Concept drift** — the actual relationship between features and the target changes over time, even if input distributions look stable (e.g., what predicts churn shifts after a pricing change)
- A model that isn't monitored will eventually degrade silently — nobody gets paged when accuracy quietly drops, unlike a crashed service, which is exactly why proactive monitoring is a deliberate practice, not something that happens automatically

**Cloud deployment basics:**
- Given your existing Azure focus, Azure ML provides managed infrastructure for training, registering, and deploying models (as real-time endpoints or batch jobs) without hand-rolling all the serving infrastructure yourself — the concepts here (endpoints, model registries, compute targets) map directly onto what you already know from the ADF/Databricks side of your DE track

---

## 2. WHY IT MATTERS

A model that never leaves a notebook has produced zero business value — deployment is where a data science project actually starts generating return. It's also where the DS/DE boundary gets genuinely blurry, and someone who can competently operate on both sides (build the model *and* the pipeline that feeds it *and* the service that serves it) is unusually hireable, because most organizations struggle specifically at the handoff between "the data scientist's model" and "a production system someone can operate and trust."

---

## 3. SYNTAX REFERENCE

```python
# --- Python: FastAPI model serving ---
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")  # loaded once at startup, not per-request

class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: str

@app.post("/predict")
def predict(request: PredictionRequest):
    input_df = preprocess(request.dict())  # reuse the SAME preprocessing as training
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    return {"prediction": int(prediction[0]), "probability": float(probability[0])}

# --- Dockerfile ---
"""
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# --- MLflow experiment tracking ---
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 200)
    model.fit(X_train, y_train)
    mlflow.log_metric("f1_score", f1_score(y_test, model.predict(X_test)))
    mlflow.sklearn.log_model(model, "model")

# --- Drift detection (simplified conceptual example) ---
from scipy.stats import ks_2samp

def check_drift(reference_data, production_data, feature, threshold=0.05):
    """Kolmogorov-Smirnov test: are these two distributions meaningfully different?"""
    stat, p_value = ks_2samp(reference_data[feature], production_data[feature])
    if p_value < threshold:
        print(f"DRIFT DETECTED in {feature}: p={p_value:.4f}")
    return p_value
```

```r
# --- R: plumber for API serving (R's equivalent to Flask/FastAPI) ---
library(plumber)

#* @post /predict
function(feature_1, feature_2, feature_3) {
  input_df <- data.frame(
    feature_1 = as.numeric(feature_1),
    feature_2 = as.numeric(feature_2),
    feature_3 = feature_3
  )
  prediction <- predict(model, input_df)
  list(prediction = prediction)
}

# Run with: plumb("api.R") %>% pr_run(port = 8000)
```

---

## 4. WORKED EXAMPLE

**Scenario:** The churn model from Module 8 has been in production for 6 months and performance has quietly degraded — the input data distribution shifted after a product redesign. Diagnose drift, decide whether to retrain or redesign features, and build a basic monitoring approach.

```python
import pandas as pd
from scipy.stats import ks_2samp
import mlflow

# --- Step 1: Compare current production data against the training reference ---
training_reference = pd.read_csv("training_data_snapshot.csv")
production_last_30_days = pd.read_csv("production_data_recent.csv")

numeric_features = ["tenure_months", "monthly_charges", "support_tickets"]

drift_report = {}
for feature in numeric_features:
    stat, p_value = ks_2samp(
        training_reference[feature], production_last_30_days[feature]
    )
    drift_report[feature] = {"ks_statistic": stat, "p_value": p_value}

drift_df = pd.DataFrame(drift_report).T
print(drift_df)
# Finding: 'support_tickets' shows p < 0.001 — a significant distributional
# shift. Investigation reveals the product redesign changed the in-app
# support flow, causing users to file tickets through a different channel
# not fully captured in this feature — the feature's MEANING has silently
# changed, not just its distribution

# --- Step 2: Check whether the model's actual performance has degraded,
#     not just the input distribution (drift doesn't always hurt performance,
#     but here we can check against known outcomes for a labeled holdout) ---
recent_labeled = pd.read_csv("recent_labeled_outcomes.csv")
model = mlflow.sklearn.load_model("models:/churn_model/production")

from sklearn.metrics import f1_score
current_f1 = f1_score(recent_labeled["churned"], model.predict(recent_labeled[numeric_features]))
original_f1 = 0.81  # logged in MLflow at original training time
print(f"Original F1: {original_f1}, Current F1: {current_f1:.3f}")
# Finding: F1 dropped from 0.81 to 0.64 — confirms the drift is materially
# hurting model performance, not just a benign distributional shift

# --- Step 3: Decide — retrain, or redesign the feature? ---
decision = """
DECISION: Redesign the feature, not just retrain.
Rationale: The drift isn't just a distribution shift that retraining on
recent data would naturally absorb — the underlying MEANING of
'support_tickets' changed because tickets now route through a different
channel not fully captured in this data source. Simply retraining on
recent data with the same broken feature would still miss the tickets
filed through the new channel. Need to first fix the data pipeline to
capture the new support channel, THEN retrain.
This is exactly the kind of data pipeline problem that data engineering
skills are needed to diagnose and fix — not purely a modeling problem.
"""
print(decision)

# --- Step 4: Build an ongoing monitoring check (to catch this automatically
#     next time, rather than discovering it 6 months later) ---
def monitoring_job():
    """Run on a schedule (e.g., weekly) as part of a pipeline."""
    for feature in numeric_features:
        _, p_value = ks_2samp(training_reference[feature], get_last_7_days_production_data()[feature])
        if p_value < 0.01:
            send_alert(f"Drift detected in {feature}, p={p_value:.4f}")
    # Also log current model performance against any newly labeled outcomes
    # that have accumulated, to catch performance decay directly, not just
    # input drift as a proxy for it
```

---

## 5. COMMON MISTAKES

- **Treating any detected drift as automatically requiring retraining.** As the worked example shows — sometimes drift reflects a genuine, silent data pipeline problem (a feature that no longer means what it used to) rather than a case where the world simply changed and the model needs updating on fresh data. Diagnosing *why* drift occurred matters as much as detecting that it occurred.
- **Deploying a model without logging which exact code, data, and hyperparameters produced it.** Without this (what MLflow and similar tools are for), an incident 6 months later becomes a forensic reconstruction project instead of a quick lookup — always log enough to reproduce any production model exactly.
- **Reusing different preprocessing logic between training and serving code.** If the training pipeline and the serving API implement feature engineering separately (even with the intent to keep them identical), they will eventually drift apart — a classic, hard-to-detect bug. Share the actual preprocessing code/pipeline object between training and serving rather than reimplementing it.
- **Monitoring only model performance metrics and not input data distributions.** Performance metrics require labeled outcomes, which are often delayed (you don't know if a churn prediction was "right" until months later) — monitoring input drift gives an earlier warning signal, before delayed ground truth confirms a problem.
- **Assuming a model deployed once stays relevant indefinitely.** As covered throughout this module — the world changes, and a model with no monitoring or retraining cadence is a model on a silent countdown to becoming wrong, with no alarm to notify anyone when it happens.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Batch vs. Real-Time**
For each use case, decide whether batch or real-time inference is appropriate, and justify:
a) Scoring all customers for churn risk once per week to prioritize retention outreach
b) Fraud detection during a live credit card transaction
c) Generating personalized product recommendations shown on a website homepage

<details>
<summary>Sample Solution</summary>

a) **Batch** — no need for instant results; a weekly scoring job is simpler to build and operate than a real-time service, and matches the weekly outreach cadence.
b) **Real-time** — the transaction must be approved or declined within the transaction flow itself, requiring low-latency, on-demand inference.
c) **Could go either way, context-dependent** — often implemented as batch-precomputed recommendations refreshed periodically (simpler, cheaper) unless genuinely real-time personalization (e.g., reacting to the current browsing session) is a specific product requirement, in which case real-time serving is needed.
</details>

---

**Exercise 2 — Drift Diagnosis Reasoning**
A fraud model's input feature `avg_transaction_amount` shows statistically significant drift (KS test p < 0.001) between training data and last month's production data, but the model's actual fraud-catch rate (recall) hasn't changed. Should you be concerned? What would you check next?

<details>
<summary>Sample Solution</summary>

Not necessarily alarming on its own — drift in an input feature doesn't automatically mean the model's predictive relationship has broken; it might just reflect a benign shift (e.g., general inflation raising average transaction amounts across the board) that the model handles fine because the *relationship* between transaction amount and fraud likelihood hasn't changed, even though the raw distribution has shifted. Next steps: confirm performance metrics (recall, precision) really are stable using recent labeled outcomes, not just assume it from this one signal; check whether OTHER features are also drifting in a correlated way that might suggest a deeper systemic shift; and keep monitoring, since detected-but-currently-harmless drift can still be an early warning of something that becomes harmful later.
</details>

---

**Exercise 3 — Full Deployment Pipeline (Applied)**
Take a model you've built in a previous module (the churn, fraud, or salary-prediction model from earlier exercises). Wrap it in a FastAPI (or plumber, in R) service with a `/predict` endpoint, containerize it with a Dockerfile, and log the training run that produced it using MLflow (parameters, metrics, and the model artifact). Then write a short monitoring plan document: which features you'd track for drift, what threshold would trigger an alert, and what your response process would be if drift is detected — this document itself is a strong portfolio artifact demonstrating production-mindedness, distinct from most beginner portfolios that stop at a Jupyter notebook.
