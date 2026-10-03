# Module 6: Regression Modeling

---

## 1. CONCEPT

Regression is covered here as **inference first, prediction second** — understanding *why* a model says what it says is often more valuable to a business than the prediction itself, since most regression use cases in practice are "which factors drive this outcome" rather than pure forecasting.

**Linear regression:**
- Models a continuous outcome as a weighted sum of predictors
- Core assumptions worth checking, not assuming: **linearity** (the relationship is actually linear, checkable via residual plots), **homoscedasticity** (residual variance is constant across predicted values, not fanning out), **independence** (residuals aren't correlated with each other — a common violation in time-ordered data), **normality of residuals** (mainly matters for small-sample inference, less so for large samples or pure prediction)
- **Coefficients** represent the expected change in the outcome per one-unit change in a predictor, holding other predictors constant — the "holding constant" part is exactly what makes multiple regression different from, and more useful than, a series of separate simple correlations
- **R² / adjusted R²** — proportion of variance explained; adjusted R² penalizes adding predictors that don't actually help, making it the fairer comparison metric when comparing models with different numbers of predictors

**Logistic regression:**
- Used for binary classification, modeling the *log-odds* of the outcome as a linear function of predictors
- Coefficients are interpreted via **odds ratios** (exponentiate the coefficient) — a coefficient of 0.4 means `exp(0.4) ≈ 1.49`, i.e., each one-unit increase in that predictor is associated with roughly a 49% increase in the odds of the outcome

**Regularization (Ridge, Lasso, Elastic Net):**
- All three address the same core problem: an unregularized model with many/correlated predictors tends to overfit, producing unstable, hard-to-trust coefficients
- **Ridge** shrinks coefficients toward zero without eliminating any (L2 penalty) — good when you believe most predictors contribute a little
- **Lasso** can shrink coefficients exactly to zero (L1 penalty), performing implicit feature selection — good when you believe only a subset of predictors truly matter
- **Elastic Net** blends both — a practical default when you're unsure which situation you're in

**Multicollinearity:**
- When predictors are highly correlated with each other, individual coefficient estimates become unstable and hard to interpret (even though overall prediction may still be fine)
- **VIF (Variance Inflation Factor)** quantifies this — a common rule of thumb flags VIF > 5 or 10 as concerning, though this is a rule of thumb, not a hard law

---

## 2. WHY IT MATTERS

Stakeholders rarely just want a prediction — they want to know what to *do*. "Which home features drive price" is actionable; "here's a price prediction" alone is not. Regression's interpretability is exactly why it remains heavily used in industry even though more complex models often predict better: a coefficient a business analyst can understand and act on is frequently worth more than a marginal accuracy gain from a black-box model.

Multicollinearity specifically trips up beginners badly: a model can have great overall predictive accuracy while individual coefficients are meaningless or even flip sign — presenting those coefficients as "insights" without checking for multicollinearity first is a fast way to give a stakeholder a wrong, confidently-stated answer.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: statsmodels (inference-focused) & scikit-learn (prediction-focused) ---
import statsmodels.api as sm
import pandas as pd

X = df[["sqft", "bedrooms", "bathrooms", "age"]]
X = sm.add_constant(X)  # statsmodels requires an explicit intercept term
y = df["price"]

model = sm.OLS(y, X).fit()
print(model.summary())  # full inferential output: coefficients, p-values, R², CIs

# Multicollinearity check
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Logistic regression
logit_model = sm.Logit(df["converted"], X).fit()
odds_ratios = pd.Series(logit_model.params).apply(lambda x: 2.71828 ** x)

# Regularized regression (scikit-learn)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X.drop(columns="const"))
ridge = Ridge(alpha=1.0).fit(X_scaled, y)
lasso = Lasso(alpha=0.1).fit(X_scaled, y)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_scaled, y)
```

```r
# --- R: lm() / glm() — richer inferential output by default ---

model <- lm(price ~ sqft + bedrooms + bathrooms + age, data = homes)
summary(model)  # coefficients, p-values, R², adjusted R², all built in

# Multicollinearity check
library(car)
vif(model)

# Logistic regression
logit_model <- glm(converted ~ sqft + bedrooms, data = homes, family = "binomial")
exp(coef(logit_model))  # odds ratios directly

# Regularized regression
library(glmnet)
x_matrix <- as.matrix(homes[, c("sqft", "bedrooms", "bathrooms", "age")])
ridge_model <- glmnet(x_matrix, homes$price, alpha = 0)   # alpha=0 is Ridge
lasso_model <- glmnet(x_matrix, homes$price, alpha = 1)   # alpha=1 is Lasso
elastic_model <- glmnet(x_matrix, homes$price, alpha = 0.5)  # blend
```

---

## 4. WORKED EXAMPLE

**Scenario:** A real estate company wants to know which home features actually drive price, not just get a price prediction. Stakeholders will ask "why does an extra bathroom add more value than an extra bedroom?" — the model needs to be interpretable.

```python
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

homes = pd.read_csv("home_sales.csv")

# --- Step 1: Check for multicollinearity before trusting any coefficient ---
predictors = ["sqft", "bedrooms", "bathrooms", "lot_size", "age", "garage_spaces"]
X = sm.add_constant(homes[predictors])

vif_data = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif_data)
# Finding: sqft and bedrooms have VIF > 8 — they're highly correlated
# (larger homes tend to have more bedrooms), which will make their
# individual coefficients unstable and hard to interpret cleanly

# --- Step 2: Fit the model and inspect diagnostics ---
model = sm.OLS(homes["price"], X).fit()
print(model.summary())

# --- Step 3: Check residual assumptions ---
import matplotlib.pyplot as plt
residuals = model.resid
fitted = model.fittedvalues

plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residual Plot — checking homoscedasticity")
# Finding: residual spread widens for higher-priced homes — heteroscedasticity
# present, meaning standard errors (and therefore p-values) may be understated
# for the model as currently specified. A log-transform of price is a common fix.

# --- Step 4: Refit with a log-transformed target to address heteroscedasticity ---
import numpy as np
homes["log_price"] = np.log(homes["price"])
model_log = sm.OLS(homes["log_price"], X).fit()

# --- Step 5: Translate coefficients to business language ---
# For a log-linear model, a coefficient of 0.05 on "bathrooms" means:
# each additional bathroom is associated with roughly a 5% increase in price,
# holding other features constant.
coef_bathrooms = model_log.params["bathrooms"]
pct_increase = (np.exp(coef_bathrooms) - 1) * 100
print(f"Each additional bathroom is associated with a "
      f"{pct_increase:.1f}% increase in price, holding other features constant.")
```

Given the multicollinearity finding, the honest business answer isn't "bedrooms don't matter" (they might, but their individual effect can't be cleanly separated from square footage in this dataset) — it's "square footage and bedroom count move together, and their combined effect on price is strong, but we can't cleanly attribute the effect to one or the other without more data or a different design." This kind of nuanced, defensible answer is exactly what interpretable regression is for.

---

## 5. COMMON MISTAKES

- **Reporting coefficients without checking multicollinearity first.** As shown above — a coefficient can look small or even flip sign purely because of correlation with another predictor, not because the true effect is small. Always run a VIF check before interpreting coefficients as "insights."
- **Interpreting R² as "model quality" in isolation.** A high R² doesn't mean the model is correctly specified (assumptions could still be violated) or that the relationships are causal — it only means a large share of variance is explained by the predictors as included.
- **Forgetting logistic regression coefficients are in log-odds, not probability.** Reporting a raw logistic coefficient as if it were a probability change is a common and materially misleading error — always exponentiate to odds ratios (or compute marginal effects) before presenting to a non-technical audience.
- **Applying Ridge/Lasso without scaling predictors first.** Both penalize coefficient *magnitude*, so unscaled predictors on different scales (e.g., square footage in the thousands vs. bathroom count in single digits) get penalized unevenly and unfairly — always standardize predictors before regularized regression.
- **Treating "the model predicts well" and "the coefficients are interpretable" as the same claim.** A model with severe multicollinearity can still predict accurately (the combined effect of correlated predictors is captured fine) while individual coefficients are meaningless — know which claim you're actually making before you make it to a stakeholder.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Coefficient Interpretation**
A logistic regression predicting `converted` (0/1) from `email_opens` returns a coefficient of 0.22 on `email_opens`. Convert this to an odds ratio and write one sentence explaining what it means in plain business language.

<details>
<summary>Hint</summary>
Odds ratio = e^coefficient. Once you have the odds ratio, express it as a percentage increase in odds per one-unit increase in the predictor.
</details>

<details>
<summary>Solution</summary>

`exp(0.22) ≈ 1.246`. Interpretation: each additional email open is associated with approximately a 24.6% increase in the odds of conversion, holding other factors constant.
</details>

---

**Exercise 2 — Multicollinearity Diagnosis**
You fit a regression predicting employee salary from `years_experience`, `years_at_company`, and `education_level`. The VIF for `years_experience` and `years_at_company` is both above 12. Explain why this is happening and propose two different ways to address it.

<details>
<summary>Hint</summary>
Think about why these two predictors would naturally move together for most employees, and consider both a feature-engineering solution and a modeling solution (regularization).
</details>

<details>
<summary>Sample Solution</summary>

These two predictors are naturally highly correlated because most employees' tenure at the company is a large share of their total experience (especially for employees who haven't changed jobs recently) — they're capturing overlapping information.

Two approaches: (1) **Feature engineering** — create a single combined feature like "years experience elsewhere" (`years_experience - years_at_company`) to separate the two distinct signals rather than keeping two overlapping ones; (2) **Regularization** — use Ridge regression, which handles correlated predictors more gracefully than OLS by shrinking their combined effect rather than arbitrarily favoring one over the other.
</details>

---

**Exercise 3 — Full Interpretable Regression (Applied)**
Using the `data_jobs` dataset, build a regression model predicting salary from at least 3 relevant predictors (e.g., years of experience, a skill count, seniority level). Run a full VIF check, inspect residual plots for heteroscedasticity, and — given salary data's typical skew — check whether a log-transform of salary is warranted before fitting. Write a plain-English paragraph translating your final coefficients into a business-usable finding.

<details>
<summary>Hint</summary>
Salary data is very commonly right-skewed (a small number of very high salaries pull the mean up), which is exactly the situation a log-transform of the target variable is designed to address — check the skewness of `salary` before deciding.
</details>
