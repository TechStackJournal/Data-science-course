# Module 5: Probability & Statistics for Data Science

---

## 1. CONCEPT

This module is the statistical foundation everything downstream depends on. Given your stats background, treat this as a fast, applied review — the goal is connecting theory you already know to the specific ways it shows up in DS work, not re-deriving it from scratch.

**Probability distributions in practice:**
- **Normal** — the default assumption for many statistical tests; real data (salaries, transaction amounts) is often right-skewed instead, which matters for which summary statistics and tests are valid
- **Binomial** — models counts of successes in fixed trials (conversion events, click-through)
- **Poisson** — models counts of rare events over a fixed interval (support tickets per hour, defects per batch)

**Descriptive statistics — choosing the right one:**
- **Mean vs. median** — the mean is pulled toward outliers/skew; the median is robust to them. For skewed data (salaries, transaction values), the median is usually the more honest "typical value," and `PERCENTILE_CONT(0.5)` (or `.median()` in pandas, `median()` in R) is more reliable than `AVG` for this reason
- **Variance / standard deviation** — spread around the mean; sensitive to the same skew issues
- **Skewness** — a formal measure of asymmetry; worth checking explicitly before choosing mean vs. median as your headline statistic

**Hypothesis testing:**
- **t-tests** — compare means between two groups (with variants for paired vs. independent samples, and equal vs. unequal variance)
- **Chi-square** — tests association between categorical variables
- **ANOVA** — compares means across 3+ groups
- Every test has **preconditions** (normality, independence, sample size, homogeneity of variance) — checking these before trusting a p-value is what separates rigorous testing from p-value fishing

**Confidence intervals** — a 95% CI does *not* mean "95% probability the true value is in this range." It means: if you repeated the sampling process many times, 95% of the intervals constructed this way would contain the true value. This distinction is subtle, frequently stated wrong even by practitioners, and worth internalizing precisely.

**A/B testing** — the applied home for all of the above: design (randomization, sample size/power calculation), execution, and the pitfalls that break validity (peeking at results early, testing multiple metrics without correction, novelty effects, Simpson's paradox from confounded segments).

---

## 2. WHY IT MATTERS

Every "is this difference real or just noise" question a business asks — did the new feature increase conversion, did the price change hurt revenue, is one marketing channel actually better than another — routes through this module's tools. Getting it wrong doesn't look like a crash; it looks like a confident, plausible, wrong recommendation that costs the business money and costs you credibility when it doesn't hold up.

A/B testing specifically is one of the most common real DS job functions (far more common day-to-day than deep learning) and one of the most common interview topics, because it's where statistical rigor meets business stakes directly.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: scipy.stats ---
from scipy import stats
import numpy as np

# Descriptive stats
data = np.array([45000, 52000, 48000, 51000, 250000])  # note the outlier
print(f"Mean: {data.mean():.0f}, Median: {np.median(data):.0f}")
print(f"Skewness: {stats.skew(data):.2f}")

# Independent t-test
group_a = np.array([23, 25, 21, 28, 24])
group_b = np.array([30, 32, 29, 35, 31])
t_stat, p_value = stats.ttest_ind(group_a, group_b)

# Chi-square test of independence
from scipy.stats import chi2_contingency
contingency_table = np.array([[50, 30], [20, 40]])  # e.g., converted x device_type
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# ANOVA
f_stat, p_value = stats.f_oneway(group_a, group_b, np.array([40, 38, 42, 41, 39]))

# Confidence interval for a mean
confidence_interval = stats.t.interval(
    confidence=0.95, df=len(data) - 1,
    loc=data.mean(), scale=stats.sem(data)
)

# Sample size / power calculation for A/B testing
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()
required_n = analysis.solve_power(
    effect_size=0.2, alpha=0.05, power=0.8, alternative="two-sided"
)
```

```r
# --- R: base + stats ---

# Descriptive stats
data <- c(45000, 52000, 48000, 51000, 250000)
mean(data); median(data)
library(moments)
skewness(data)

# Independent t-test
group_a <- c(23, 25, 21, 28, 24)
group_b <- c(30, 32, 29, 35, 31)
t.test(group_a, group_b)  # returns t-stat, p-value, CI, all in one call

# Chi-square test
contingency_table <- matrix(c(50, 30, 20, 40), nrow = 2)
chisq.test(contingency_table)

# ANOVA
group_c <- c(40, 38, 42, 41, 39)
values <- c(group_a, group_b, group_c)
groups <- factor(rep(c("A", "B", "C"), each = 5))
summary(aov(values ~ groups))

# Power calculation
library(pwr)
pwr.t.test(d = 0.2, sig.level = 0.05, power = 0.8, type = "two.sample")
```

Note: R's `t.test()` returns the confidence interval automatically as part of its output — one of the small conveniences that makes R faster for pure statistical exploration.

---

## 4. WORKED EXAMPLE

**Scenario:** A product team ran an A/B test on a new checkout flow. Variant B shows a higher conversion rate, and they got p = 0.04 after 3 days — they want to ship it immediately. Evaluate whether the test is actually valid before signing off.

```python
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

results = pd.read_csv("ab_test_results.csv")
# columns: variant, converted (0/1), session_date

# --- Check 1: Was the sample size sufficient (power check), or is this
#     an underpowered test that got lucky? ---
summary = results.groupby("variant")["converted"].agg(["mean", "count"])
print(summary)

effect_size = proportion_effectsize(
    summary.loc["A", "mean"], summary.loc["B", "mean"]
)
power_analysis = NormalIndPower()
required_n = power_analysis.solve_power(
    effect_size=effect_size, alpha=0.05, power=0.8
)
print(f"Required sample size per group for 80% power: {required_n:.0f}")
print(f"Actual sample size per group: {summary['count'].min()}")
# Finding: actual N is well below required N — the test was underpowered,
# meaning this "significant" result has an elevated chance of being noise

# --- Check 2: Was the test run for a full business cycle, or does it
#     capture a novelty effect / day-of-week bias? ---
results["session_date"] = pd.to_datetime(results["session_date"])
daily_conversion = (
    results.groupby(["session_date", "variant"])["converted"]
    .mean().unstack()
)
print(daily_conversion)
# Finding: 3 days doesn't span a full weekly cycle — weekend vs. weekday
# behavior isn't represented, which risks a confounded result

# --- Check 3: Was only one metric tested, or were multiple metrics checked
#     without correction (multiple comparisons problem)? ---
# (This requires knowing the test plan — ask directly: "how many metrics
#  did you check before landing on this one?")

# --- Conclusion ---
conclusion = """
RECOMMENDATION: Do not ship yet.
1. The test is underpowered — actual sample size is below what's required
   for reliable detection at this effect size. The p=0.04 result has a
   real chance of being noise, not a durable effect.
2. The 3-day window doesn't cover a full weekly cycle, risking a
   confounded comparison (e.g., weekend behavior skewing results).
3. Recommend extending the test to reach the required sample size AND
   at least one full week, with the primary metric locked in advance.
"""
print(conclusion)
```

---

## 5. COMMON MISTAKES

- **Treating p < 0.05 as proof, full stop.** A p-value tells you how surprising the observed data would be under the null hypothesis — it does not tell you the effect is large, business-relevant, or even real given other context (underpowered tests, multiple comparisons). Always pair p-values with effect size and confidence intervals.
- **"Peeking" at A/B test results and stopping early once significance appears.** This inflates the false positive rate substantially — the test's statistical validity assumes a pre-committed sample size/duration, and stopping early the moment p dips below 0.05 breaks that assumption.
- **Testing many metrics and reporting only the significant one.** Checking 20 metrics at α=0.05 gives you roughly a 64% chance of at least one false positive by pure chance — this is the multiple comparisons problem, and it's a common, often unintentional way "significant" results get reported that aren't real.
- **Using the mean as the headline statistic for skewed data without checking.** As in the worked example's descriptive stats — a single outlier can move a mean dramatically while barely moving a median. Always check skewness before defaulting to mean.
- **Confusing statistical significance with practical significance.** With enough sample size, even a trivially small, business-irrelevant effect will become "statistically significant." Always ask: is this effect size big enough to matter, separate from whether it's detectable at all.
- **Ignoring Simpson's paradox risk.** An aggregate result can reverse when broken down by a confounding subgroup (e.g., variant B looks better overall but is actually worse in every individual segment, because segment mix differs between variants). Always check whether the aggregate result holds within key subgroups before trusting it.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Test Selection**
For each scenario, identify which statistical test is appropriate and why:
a) Comparing average time-on-page between two website designs (2 groups, continuous outcome)
b) Testing whether purchase rate differs by marketing channel (categorical x categorical)
c) Comparing average order value across 4 different regions (4 groups, continuous outcome)

<details>
<summary>Solution</summary>

a) Independent samples t-test — comparing means between 2 groups on a continuous outcome.
b) Chi-square test of independence — testing association between two categorical variables.
c) One-way ANOVA — comparing means across more than 2 groups. (Note: a significant ANOVA tells you *some* group differs, not *which* — a post-hoc test like Tukey's HSD is needed to identify which pairs differ.)
</details>

---

**Exercise 2 — Power & Sample Size**
A company wants to detect a 2 percentage-point lift in conversion rate (from a baseline of 10% to 12%), with 80% power and α = 0.05. Using the `statsmodels`/`pwr` code from the syntax reference, calculate the required sample size per group, and explain what happens to the required sample size if they instead wanted to detect a 5 percentage-point lift.

<details>
<summary>Hint</summary>
Smaller effect sizes require larger sample sizes to detect reliably — the relationship is inverse and non-linear (roughly, required N scales with the inverse square of the effect size). Calculate `proportion_effectsize(0.10, 0.12)` vs. `proportion_effectsize(0.10, 0.15)` and compare the resulting required N.
</details>

<details>
<summary>Sample Solution</summary>

```python
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower

power_analysis = NormalIndPower()

effect_small = proportion_effectsize(0.10, 0.12)
n_small = power_analysis.solve_power(effect_size=effect_small, alpha=0.05, power=0.8)

effect_large = proportion_effectsize(0.10, 0.15)
n_large = power_analysis.solve_power(effect_size=effect_large, alpha=0.05, power=0.8)

print(f"N required for 2pp lift: {n_small:.0f}")
print(f"N required for 5pp lift: {n_large:.0f}")
```

The 2pp lift requires a substantially larger sample size than the 5pp lift — smaller effects are harder to detect and demand more data, which is why many "quick" A/B tests are underpowered for the subtle effects they're actually trying to measure.
</details>

---

**Exercise 3 — Full A/B Test Audit (Applied)**
You're handed the results of a real or simulated A/B test (use the `data_jobs` dataset or a simulated dataset of your own construction: two groups, a binary outcome, and timestamps). Run the full audit from the worked example: check power/sample size adequacy, check for a full-cycle time window, and check whether the aggregate result holds when broken down by at least one plausible confounding subgroup. Write a recommendation memo (ship / don't ship / extend the test) with your reasoning made explicit — this is a strong portfolio artifact and a near-exact rehearsal of a real DS interview case study.
