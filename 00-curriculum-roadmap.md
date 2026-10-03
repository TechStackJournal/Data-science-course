# Data Science Mastery Course
## A Complete Beginner-to-Advanced Curriculum with Real-World Analytics Focus

**Languages:** Python (primary) + R (parallel, for statistical modeling)
**Format:** Self-paced, project-driven. Every module ends with a real-world scenario, not a toy dataset.
**Structure:** 4 Parts → 15 Modules → each with Objectives, Topics, Real-World Case Study, Tools, and a Deliverable.

---

# PART 1: FOUNDATIONS (Beginner)
*Goal: Get comfortable with the tools and the mindset of a data scientist before touching a single model.*

## Module 0 — What Data Science Actually Is (Orientation)
**Objectives:** Understand the DS lifecycle; distinguish DS from data analytics, BI, and ML engineering; learn how to frame a business question as a data question.

**Topics:**
- The DS lifecycle: business question → data collection → cleaning → EDA → modeling → communication → deployment
- Data science vs. data analytics vs. ML engineering vs. data engineering — where each role starts and stops
- Types of data: structured, semi-structured, unstructured
- Types of analytics: descriptive, diagnostic, predictive, prescriptive

**Real-World Scenario:** A retail company sees a 15% revenue drop last quarter. Practice turning that vague problem into 3 specific, answerable data questions (e.g., "Did drop-off happen in a specific region, product category, or customer segment?").

**Deliverable:** A one-page "problem framing" document for a business scenario of your choice.

---

## Module 1 — Python for Data Science
**Objectives:** Working fluency in Python for data manipulation — not general software engineering Python, DS-specific Python.

**Topics:**
- Core syntax, data structures (lists, dicts, tuples, sets), control flow
- Functions, list comprehensions, lambda functions
- File I/O: CSV, JSON, Excel
- Virtual environments, pip, Jupyter notebooks
- NumPy: arrays, vectorized operations, broadcasting

**Real-World Scenario:** Parse a folder of inconsistent daily sales CSVs (different column orders, some missing headers) into one clean dataset using pure Python before pandas is even introduced — this is what messy real pipelines actually look like.

**Deliverable:** A script that ingests and standardizes 5+ inconsistent raw files.

---

## Module 2 — R for Statistical Computing
**Objectives:** Parallel fluency in R, focused on where R still beats Python: fast statistical exploration and publication-quality visualization.

**Topics:**
- R syntax, vectors, data frames, factors
- Base R vs. tidyverse philosophy
- RStudio / Quarto workflow
- Reading data, basic manipulation with `dplyr`

**Real-World Scenario:** Reproduce Module 1's cleaning task in R using `dplyr` — direct side-by-side comparison so the two languages stop feeling like separate worlds.

**Deliverable:** Same dataset, cleaned in both Python and R — a personal reference sheet of "how do I do X" in both languages.

---

## Module 3 — Data Wrangling with pandas & dplyr
**Objectives:** This is the actual day-to-day job. Most real-world DS time goes here, not modeling.

**Topics:**
- pandas: filtering, grouping, merging/joining, pivot tables, `apply`/`map`
- Handling missing data: deletion vs. imputation strategies, when each is appropriate
- Handling duplicates, outliers, inconsistent categorical encoding
- Date/time handling and resampling
- `dplyr`/`tidyr` equivalents: `filter`, `mutate`, `group_by`, `summarise`, `pivot_longer`/`pivot_wider`

**Real-World Scenario:** You're given a hospital readmissions dataset with inconsistent date formats, 12% missing values in a key field, and duplicate patient records from a merged system after a hospital acquisition. Clean it defensibly — document every decision, because in healthcare/finance you will be asked to justify every row you dropped.

**Deliverable:** A cleaning script + a "data cleaning decisions log" (this habit becomes a portfolio differentiator later).

---

## Module 4 — Exploratory Data Analysis (EDA) & Data Visualization
**Objectives:** Turn cleaned data into insight before any model exists.

**Topics:**
- Univariate, bivariate, multivariate EDA
- matplotlib/seaborn: distributions, correlations, categorical breakdowns
- ggplot2: grammar of graphics philosophy (different mental model from matplotlib — worth understanding explicitly)
- Choosing the right chart for the question (this is more subtle than it sounds — bar vs. line vs. box vs. violin)
- Identifying data quality red flags through visualization (e.g., a bimodal distribution that shouldn't be there)

**Real-World Scenario:** An e-commerce company wants to know why cart abandonment rose. Full EDA on a transactions + sessions dataset — find the story in the data before anyone builds a model. Present 5 concrete findings with charts, written for a non-technical stakeholder.

**Deliverable:** An EDA notebook + a 1-page executive summary (this dual deliverable — technical notebook + plain-English summary — should become a habit for every project going forward).

---

# PART 2: STATISTICAL FOUNDATIONS & CORE MACHINE LEARNING (Intermediate)
*Goal: Understand why models work, not just how to call `.fit()`.*

## Module 5 — Probability & Statistics for Data Science
**Objectives:** The statistical foundation that separates a data scientist from someone who just runs sklearn.

**Topics:**
- Probability distributions (normal, binomial, Poisson) and when each shows up in real data
- Descriptive stats: mean vs. median (and when median lies less), variance, std dev, skewness
- Hypothesis testing: t-tests, chi-square, ANOVA — and their real precondition checks (not just "p < 0.05")
- Confidence intervals and what they actually mean (widely misunderstood, worth doing carefully)
- A/B testing: design, sample size/power calculation, common pitfalls (peeking, multiple comparisons, Simpson's paradox)

**Real-World Scenario:** A product team ran an A/B test on a new checkout flow and got p = 0.04 — they want to ship it immediately. Evaluate whether the test was actually valid (sample size, test duration, novelty effects, multiple metrics tested) before signing off. This is one of the most common real interview and real job scenarios in DS.

**Deliverable:** A full A/B test analysis with a recommendation and explicit caveats — the caveats matter as much as the conclusion.

---

## Module 6 — Regression Modeling
**Objectives:** Understand regression as inference, not just prediction.

**Topics:**
- Simple and multiple linear regression, assumptions (linearity, homoscedasticity, independence, normality of residuals) and how to actually check them
- Interpreting coefficients, p-values, R², adjusted R²
- Logistic regression for classification, interpreting odds ratios
- Regularization: Ridge, Lasso, Elastic Net — and *why* each exists (bias-variance tradeoff made concrete)
- Multicollinearity: detection (VIF) and remedies

**Real-World Scenario:** A real estate company wants to know which home features actually drive price — not just predict price, but explain it to stakeholders who will ask "why does an extra bathroom add more value than an extra bedroom?" Requires interpretable modeling, not a black box.

**Deliverable:** A regression model + a written explanation of coefficients in business terms (this is the actual job — models nobody can explain don't get shipped).

---

## Module 7 — Classification & Core ML Algorithms
**Objectives:** The standard supervised learning toolkit.

**Topics:**
- Decision trees, random forests, k-NN, Naive Bayes, SVM
- Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC — and which metric actually matters for which business problem
- Confusion matrices and the real cost of false positives vs. false negatives
- Train/test/validation splits, k-fold cross-validation
- Handling imbalanced classes: class weights, SMOTE, threshold tuning

**Real-World Scenario:** Build a fraud detection classifier where fraud is 0.3% of transactions. A model with 99.7% "accuracy" that predicts nothing is fraud is useless — walk through why accuracy is the wrong metric here, and what to use instead (precision-recall curve, cost-weighted evaluation).

**Deliverable:** A classifier with a justified metric choice and a threshold selection defended against a stated business cost (e.g., cost of a missed fraud vs. cost of a false alarm).

---

## Module 8 — Feature Engineering
**Objectives:** This is where most real performance gains come from — more than algorithm choice.

**Topics:**
- Encoding categorical variables: one-hot, label, target/mean encoding, and leakage risks in each
- Scaling/normalization: when it matters (distance-based models) vs. when it doesn't (tree-based models)
- Creating interaction features, polynomial features, domain-driven features
- Feature selection: filter, wrapper, embedded methods
- Data leakage: the single most common way beginners silently break a model — dedicated deep dive with examples

**Real-World Scenario:** A churn-prediction model is getting suspiciously good results (98% accuracy). Diagnose the leakage — a feature in the dataset ("cancellation_request_date") is a proxy for the target itself. This exact mistake happens constantly in real DS work and is a classic interview scenario.

**Deliverable:** A feature engineering pipeline with a documented leakage audit.

---

## Module 9 — Ensemble Methods & Model Tuning
**Objectives:** Push from "good enough" to "actually competitive" model performance.

**Topics:**
- Bagging vs. boosting, why ensembles usually beat single models
- Random Forest deep dive, XGBoost, LightGBM, CatBoost
- Hyperparameter tuning: grid search, random search, Bayesian optimization
- Model interpretability: SHAP values, feature importance, partial dependence plots
- `caret`/`tidymodels` in R for parallel exposure

**Real-World Scenario:** A bank needs a credit-risk model that's both high-performing and *explainable* to regulators (this is a real, legally-mandated constraint in lending — "black box, trust me" isn't acceptable). Use SHAP to make an XGBoost model's decisions auditable feature-by-feature for individual loan denials.

**Deliverable:** A tuned ensemble model + SHAP-based explanation report for 3 individual predictions.

---

# PART 3: ADVANCED TOPICS
*Goal: Handle unstructured data, temporal data, and unsupervised problems — the topics that separate an intermediate DS from an advanced one.*

## Module 10 — Unsupervised Learning
**Objectives:** Find structure in data with no labels.

**Topics:**
- Clustering: k-means, hierarchical clustering, DBSCAN — and how to choose k without cheating
- Dimensionality reduction: PCA, t-SNE, UMAP — when each is appropriate and their real limitations
- Anomaly detection: isolation forests, statistical approaches
- Evaluating unsupervised results without ground truth (silhouette score, business validation)

**Real-World Scenario:** A marketing team wants "customer segments" for a campaign, with no predefined categories. Cluster customers on purchase behavior, then — critically — translate abstract clusters ("Cluster 3: high frequency, low basket size") into a business-usable segment name and actionable strategy. The translation step is where most unsupervised projects fail in practice.

**Deliverable:** A segmentation analysis with named, business-actionable clusters (not just "Cluster 0, 1, 2").

---

## Module 11 — Time Series Analysis & Forecasting
**Objectives:** Handle temporal data correctly — a common real-world blind spot for people trained only on i.i.d. datasets.

**Topics:**
- Trend, seasonality, stationarity, differencing
- Classical methods: moving averages, exponential smoothing, ARIMA/SARIMA
- Modern approaches: Prophet, feature-based ML for time series
- Why standard cross-validation is *wrong* for time series (temporal leakage) and how to do it correctly (walk-forward validation)
- Evaluation: MAPE, RMSE and their tradeoffs

**Real-World Scenario:** Forecast next quarter's demand for a retailer with strong holiday seasonality and a recent supply chain disruption that broke the historical pattern. Handle the very real problem of "the model trained on history that no longer applies."

**Deliverable:** A forecast with prediction intervals and an explicit discussion of what could break the forecast (regime changes, external shocks).

---

## Module 12 — Deep Learning Fundamentals
**Objectives:** Enough depth to build simple neural networks and understand what's happening inside modern AI systems, without needing a PhD.

**Topics:**
- Perceptrons, backpropagation intuition, activation functions
- Building an MLP from scratch (conceptually — no library) then in PyTorch
- CNNs for image data — convolutions, pooling, why they work for spatial data
- Regularization for deep nets: dropout, batch normalization, early stopping
- When to use deep learning vs. when classical ML is still the better (faster, more interpretable, less data-hungry) choice

**Real-World Scenario:** A manufacturing company wants automated defect detection from product images. Build a CNN classifier, but also explicitly justify *why* deep learning is the right tool here versus a classical approach — this justification is what separates someone who reaches for deep learning by default from someone who reasons about it.

**Deliverable:** A trained CNN + a written tool-selection rationale.

---

## Module 13 — NLP & Working with Text/Unstructured Data
**Objectives:** Handle the data type most DS courses under-cover, despite it being everywhere in real business (reviews, support tickets, contracts).

**Topics:**
- Text preprocessing: tokenization, stemming/lemmatization, stopwords — and when to skip each
- Bag-of-words, TF-IDF, and their limitations
- Word embeddings (Word2Vec conceptually), then modern transformer embeddings
- Using pretrained transformer models (Hugging Face) for classification, sentiment, and NER without training from scratch
- When fine-tuning is worth it vs. when a pretrained model + good prompting is sufficient

**Real-World Scenario:** A company has 50,000 unstructured customer support tickets and wants to know the top 5 recurring complaint categories — with no predefined label taxonomy. Combine embeddings + clustering to discover categories, then classify new tickets automatically.

**Deliverable:** A ticket-categorization pipeline + a summary of discovered complaint themes.

---

# PART 4: PRODUCTIONIZATION, PORTFOLIO & CAREER READINESS
*Goal: Convert skill into a job. This is the part most courses skip and most job-seekers underinvest in.*

## Module 14 — Model Deployment & MLOps Fundamentals
**Objectives:** Understand the full path from notebook to production — the thing that makes you hireable over someone who's only ever run `.fit()` in Jupyter.

**Topics:**
- Serving models: Flask/FastAPI basics
- Containerization: Docker fundamentals for reproducible environments
- Experiment tracking (MLflow) and model versioning
- Batch vs. real-time inference — when each is appropriate
- Monitoring: data drift, model decay, and why a shipped model isn't a finished model
- Cloud deployment basics (Azure ML, or your platform of choice)

**Real-World Scenario:** Your churn model from Module 8 has been in production for 6 months and performance has quietly degraded — the input data distribution shifted after a product redesign. Diagnose drift, decide whether to retrain or redesign features, and build a basic monitoring dashboard so this is caught automatically next time.

**Deliverable:** A deployed model behind a simple API + a monitoring plan document.

---

## Module 15 — Capstone Project & Portfolio Development
**Objectives:** Prove everything above works together on an ambiguous, realistic problem — and package it for hiring managers.

**Topics:**
- End-to-end capstone: pick a genuinely messy, real dataset (not a clean Kaggle competition dataset) and take it from raw data → business framing → EDA → modeling → deployment → written communication
- Portfolio structure: README standards, reproducibility (requirements.txt/environment files), architecture diagrams, "why I made this decision" documentation
- Case-study interview practice: framing ambiguous prompts ("how would you detect fake reviews?") into a structured DS approach
- Resume/story positioning: translating projects into concrete, quantified achievements

**Real-World Scenario:** This *is* the real-world scenario — treat the capstone like a 2-week take-home assignment from a real employer. Suggested options: healthcare readmission risk, retail demand forecasting with external shocks, fraud detection with class imbalance, or customer support ticket triage using NLP (pick whichever most closely simulates your target job's industry).

**Deliverable:** A complete, GitHub-ready capstone repository + a 10-minute presentation of it as if pitching to a hiring panel.

---

# Appendix A: Suggested Pacing (3–6 Month Intensive)

| Weeks | Modules | Focus |
|---|---|---|
| 1–3 | 0–4 | Foundations, wrangling, EDA |
| 4–7 | 5–7 | Stats, regression, classification |
| 8–10 | 8–9 | Feature engineering, ensembles |
| 11–14 | 10–13 | Unsupervised, time series, deep learning, NLP |
| 15–17 | 14 | Deployment/MLOps |
| 18–20+ | 15 | Capstone + portfolio + interview prep |

# Appendix B: Recurring Habits (apply in every module, not just once)
- **Data cleaning decisions log** — document and justify every transformation
- **Dual deliverable** — technical notebook + plain-English executive summary
- **Metric justification** — never report a metric without explaining why it's the right one for the business problem
- **Leakage audit** — actively check for it in every supervised project, not just Module 8

# Appendix C: Tooling Reference
- **Python:** pandas, NumPy, matplotlib/seaborn, scikit-learn, XGBoost/LightGBM, PyTorch, Hugging Face Transformers, statsmodels, Prophet
- **R:** tidyverse (dplyr, tidyr, ggplot2), tidymodels/caret, forecast
- **Deployment:** FastAPI/Flask, Docker, MLflow
- **Environment:** Jupyter, RStudio/Quarto, Git/GitHub
