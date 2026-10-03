# Module 11: Time Series Analysis & Forecasting

---

## 1. CONCEPT

Time series data has a structural property that breaks many of the assumptions standard ML techniques rely on: **observations are ordered and dependent on each other**, not independent and identically distributed. This module covers how to handle that correctly.

**Core concepts:**
- **Trend** — the long-term direction of the series (upward, downward, flat)
- **Seasonality** — regular, predictable patterns tied to a fixed period (daily, weekly, yearly)
- **Stationarity** — a series whose statistical properties (mean, variance) don't change over time; many classical forecasting methods assume or require stationarity
- **Differencing** — subtracting each value from the previous one, a common technique to transform a non-stationary series into a stationary one

**Classical methods:**
- **Moving averages** — smooth out noise to reveal underlying trend; simple but purely descriptive, not great for forecasting complex patterns alone
- **Exponential smoothing** (including Holt-Winters) — weights recent observations more heavily than older ones, with variants that explicitly model trend and seasonality
- **ARIMA/SARIMA** — AutoRegressive Integrated Moving Average; models a series as a function of its own past values (AR), past forecast errors (MA), and differencing (I) to achieve stationarity; SARIMA extends this with an explicit seasonal component

**Modern approaches:**
- **Prophet** (Meta/Facebook) — designed for business time series with strong seasonality and holiday effects, relatively easy to use with sensible defaults, good for quick, decent forecasts without deep ARIMA tuning
- **Feature-based ML for time series** — reframe forecasting as a supervised learning problem by engineering lag features, rolling statistics, and date-derived features, then apply standard regression/ensemble models

**Why standard cross-validation is wrong for time series:**
Standard k-fold CV shuffles data randomly across folds, which for time series means training on future data to predict the past — a severe, easy-to-miss form of leakage. The correct approach is **walk-forward validation** (also called time series cross-validation): train on an expanding or sliding window of past data, always validate on data that comes strictly after the training window.

**Evaluation metrics:**
- **MAPE** (Mean Absolute Percentage Error) — intuitive, expressed as a percentage, but undefined/unstable when actual values are near zero
- **RMSE** (Root Mean Squared Error) — penalizes large errors more heavily than MAPE; in the original units of the target, which aids interpretability but makes cross-series comparison harder

---

## 2. WHY IT MATTERS

Treating time series data like any other tabular dataset — random train/test splits, standard cross-validation — produces forecasts that look great in evaluation and fail in production, because the evaluation itself was invalid (the model effectively "saw the future" during training). This is a genuinely common blind spot even among people with solid general ML skills, precisely because the mistake is structural rather than a bug you'd notice by inspecting code line-by-line.

Real-world forecasting also constantly runs into the problem of the future *not* looking like the past — a regime change, a supply shock, a pandemic — and being able to reason about when a forecast's underlying assumptions have broken down is as important as the forecasting technique itself.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: statsmodels (classical) + Prophet (modern) ---
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Check stationarity (Augmented Dickey-Fuller test)
result = adfuller(series)
print(f"ADF p-value: {result[1]:.4f}")  # p < 0.05 suggests stationary

# Differencing to achieve stationarity
series_diff = series.diff().dropna()

# ARIMA
model = ARIMA(series, order=(1, 1, 1))  # (p, d, q)
fitted = model.fit()
forecast = fitted.forecast(steps=12)

# SARIMA (with seasonal component)
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fitted = sarima_model.fit()

# Prophet
from prophet import Prophet
df_prophet = pd.DataFrame({"ds": dates, "y": series})  # Prophet requires these exact column names
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
prophet_model.fit(df_prophet)
future = prophet_model.make_future_dataframe(periods=90)
forecast = prophet_model.predict(future)

# Walk-forward validation (correct CV for time series)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # train always precedes test in time — no leakage

# Evaluation
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
mape = mean_absolute_percentage_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

```r
# --- R: forecast package (long-standing standard) ---
library(forecast)

ts_data <- ts(series, frequency = 12)  # e.g., monthly data

# Stationarity check
library(tseries)
adf.test(ts_data)

# Auto-ARIMA (automatically selects p, d, q)
auto_model <- auto.arima(ts_data, seasonal = TRUE)
summary(auto_model)
forecast_result <- forecast(auto_model, h = 12)
plot(forecast_result)

# Exponential smoothing (Holt-Winters)
hw_model <- HoltWinters(ts_data)
hw_forecast <- forecast(hw_model, h = 12)

# Walk-forward / time series CV
tsCV_errors <- tsCV(ts_data, forecastfunction = function(x, h) forecast(auto.arima(x), h = h), h = 12)
sqrt(mean(tsCV_errors^2, na.rm = TRUE))  # RMSE from walk-forward validation
```

---

## 4. WORKED EXAMPLE

**Scenario:** Forecast next quarter's demand for a retailer with strong holiday seasonality and a recent supply chain disruption that broke the historical pattern.

```python
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

sales = pd.read_csv("daily_sales.csv", parse_dates=["date"])

# --- Step 1: Visualize and identify the disruption ---
import matplotlib.pyplot as plt
plt.plot(sales["date"], sales["units_sold"])
plt.axvline(pd.Timestamp("2026-03-15"), color="red", linestyle="--",
            label="Supply disruption start")
plt.legend()
# Finding: a clear level shift downward starting mid-March, inconsistent
# with the prior seasonal pattern — the historical data BEFORE this point
# reflects a different regime than the current one

# --- Step 2: Decide how to handle the regime change ---
# Option A: Train only on post-disruption data (loses seasonal history,
#           but avoids contaminating the model with an outdated pattern)
# Option B: Train on full history but add a changepoint/regressor flagging
#           the disruption period (keeps seasonal learning, explicitly
#           models the disruption rather than ignoring it)
# Decision: Option B — Prophet supports custom changepoints and regressors
# explicitly for this kind of scenario

df_prophet = sales.rename(columns={"date": "ds", "units_sold": "y"})
df_prophet["supply_disrupted"] = (
    df_prophet["ds"] >= pd.Timestamp("2026-03-15")
).astype(int)

model = Prophet(
    yearly_seasonality=True,
    changepoints=["2026-03-15"]  # explicitly tell Prophet a regime change happened here
)
model.add_regressor("supply_disrupted")
model.fit(df_prophet)

# --- Step 3: Build the future dataframe, assuming disruption continues ---
future = model.make_future_dataframe(periods=90)
future["supply_disrupted"] = 1  # assume disruption conditions persist into forecast window
forecast = model.predict(future)

# --- Step 4: Validate with walk-forward validation, not a random split ---
from prophet.diagnostics import cross_validation, performance_metrics
cv_results = cross_validation(
    model, initial="180 days", period="30 days", horizon="90 days"
)
metrics = performance_metrics(cv_results)
print(metrics[["horizon", "mape", "rmse"]])

# --- Step 5: Communicate uncertainty, not just a point forecast ---
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))
# ALWAYS present the prediction interval alongside the point forecast —
# especially critical here, since the disruption introduces genuine
# additional uncertainty beyond normal seasonal variation
```

**Communicating the forecast:** "Our Q2 forecast assumes the current supply disruption continues at its present severity. If conditions improve, actual demand will likely exceed this forecast; if they worsen, it could fall further below. The 80% prediction interval reflects this added uncertainty." — this kind of explicit caveat is what separates a forecast that protects you from one that sets you up to be blamed when reality inevitably diverges somewhat from the point estimate.

---

## 5. COMMON MISTAKES

- **Using standard k-fold cross-validation on time series data.** As covered in Concept — this trains on future data to predict the past, producing artificially optimistic and invalid evaluation metrics. Always use walk-forward/time series-aware validation.
- **Ignoring a known regime change and forecasting as if history still applies.** As in the worked example — a model trained on pre-disruption data without any adjustment will confidently predict a demand level that's no longer realistic, and won't self-correct until enough new data accumulates to drag the trend estimate down, which may be too slow for a business decision that needs to be made now.
- **Reporting only a point forecast, without a prediction interval.** A single number invites false precision — especially damaging when the forecast is being used for inventory or staffing decisions where the cost of being wrong is asymmetric (e.g., stockouts costing more than modest overstock, or vice versa).
- **Differencing a series without checking whether it was actually necessary.** Over-differencing can introduce artificial patterns or discard genuine long-term trend information — always check stationarity with a formal test (ADF) rather than differencing reflexively.
- **Extrapolating a forecast far beyond the horizon the model was validated on.** A model validated on 30-day-ahead walk-forward forecasts shouldn't be trusted for a 180-day-ahead forecast without separately validating at that longer horizon — forecast accuracy typically degrades substantially further out, and that degradation should be measured, not assumed away.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Stationarity & Differencing**
You run an ADF test on a monthly revenue series and get p = 0.62. What does this suggest about the series, and what would you do next before fitting an ARIMA model?

<details>
<summary>Hint</summary>
The ADF test's null hypothesis is that the series is non-stationary. A p-value of 0.62 fails to reject that null.
</details>

<details>
<summary>Solution</summary>

A p-value of 0.62 means you fail to reject the null hypothesis of non-stationarity — the series is likely non-stationary (e.g., has a trend). Before fitting an ARIMA model, apply differencing (start with first-order differencing, `series.diff()`), then re-run the ADF test on the differenced series to confirm it's now stationary before proceeding to model fitting. This differencing order becomes the `d` parameter in ARIMA(p, d, q).
</details>

---

**Exercise 2 — Validation Strategy**
Explain, in your own words, why applying `train_test_split(X, y, test_size=0.2, random_state=42)` (a standard random split) to a time series forecasting problem is a form of data leakage, using a concrete example of what could go wrong.

<details>
<summary>Sample Solution</summary>

A random split can place, say, January and March data in the training set while February ends up in the test set — meaning the model is effectively trained on data both before AND after the test period, and could exploit that future information (e.g., a rolling average feature computed using March data indirectly informs a February prediction). This produces test performance that's artificially good and doesn't reflect how the model would actually perform in production, where it only ever has access to genuinely past data when forecasting the future. Walk-forward validation avoids this by strictly ensuring the test period always comes chronologically after the training period.
</details>

---

**Exercise 3 — Full Forecast with Regime-Change Handling (Applied)**
Using a real or simulated time series dataset with at least one identifiable regime change or anomaly (a COVID-era retail dataset works well if available, or simulate one by artificially shifting a portion of a series), build a Prophet or SARIMA forecast. Explicitly handle the regime change (via a changepoint, regressor, or a documented decision to exclude pre-disruption data), validate using walk-forward cross-validation, and report both the point forecast and prediction intervals. Write a short paragraph, addressed to a business stakeholder, explaining what assumptions the forecast depends on and when it should be revisited.
