# Module 3: Data Wrangling with pandas & dplyr

---

## 1. CONCEPT

This is the module where most of an actual data scientist's week lives. Wrangling covers everything between "I have raw data" and "I have a dataset I can trust enough to analyze."

Core operations:

- **Filtering** — selecting rows meeting a condition
- **Grouping & aggregation** — collapsing rows into summary statistics per category (`groupby` / `group_by`)
- **Merging/joining** — combining datasets from different sources (inner, left, right, outer joins — the same concepts as SQL joins, applied in-memory)
- **Reshaping** — pivoting between "wide" and "long" formats (`pivot_table`/`melt` in pandas; `pivot_wider`/`pivot_longer` in tidyr)
- **Missing data handling** — deciding, per-column and per-context, whether to drop, impute, or flag missing values
- **Duplicate handling** — detecting and resolving duplicate records, especially after merges
- **Date/time handling** — parsing, resampling, and extracting components (day of week, month, etc.) from timestamps

The throughline across all of these: **every cleaning decision is a judgment call that changes your results**, and defensible data science means being able to explain and justify each one — not just execute it.

---

## 2. WHY IT MATTERS

A model built on badly wrangled data will produce confident, precise, wrong answers — and wrongness born from data issues is far more common in real work than wrongness born from bad algorithm choice. Interviewers and hiring managers weight wrangling skill heavily precisely because it's unglamorous and commonly under-practiced by people who jumped straight to modeling tutorials.

The habit introduced here — a **data cleaning decisions log** — is what turns "I cleaned the data" into "I can defend every row I touched," which matters enormously in regulated industries (healthcare, finance) and in any interview where you're asked to walk through your process.

---

## 3. SYNTAX REFERENCE

```python
# --- pandas ---
import pandas as pd

df = pd.read_csv("orders.csv")

# Filtering
completed = df[df["status"] == "completed"]

# Grouping & aggregation
revenue_by_region = df.groupby("region")["amount"].sum()
summary = df.groupby("region").agg(
    total_revenue=("amount", "sum"),
    order_count=("order_id", "count"),
    avg_order=("amount", "mean")
)

# Merging
orders_with_customers = df.merge(
    customers_df, on="customer_id", how="left"
)

# Reshaping
wide = df.pivot_table(index="region", columns="category", values="amount", aggfunc="sum")
long = wide.reset_index().melt(id_vars="region", var_name="category", value_name="amount")

# Missing data
df["amount"].isna().sum()                      # count missing
df["amount"].fillna(df["amount"].median())      # impute with median
df.dropna(subset=["customer_id"])                # drop rows missing a critical field

# Duplicates
df.duplicated(subset=["order_id"]).sum()         # count duplicates
df.drop_duplicates(subset=["order_id"], keep="first")

# Date/time
df["order_date"] = pd.to_datetime(df["order_date"])
df["order_month"] = df["order_date"].dt.to_period("M")
df["day_of_week"] = df["order_date"].dt.day_name()
```

```r
# --- dplyr / tidyr ---
library(dplyr)
library(tidyr)
library(lubridate)

# Filtering
completed <- df %>% filter(status == "completed")

# Grouping & aggregation
summary <- df %>%
  group_by(region) %>%
  summarise(
    total_revenue = sum(amount),
    order_count   = n(),
    avg_order     = mean(amount)
  )

# Merging (join)
orders_with_customers <- df %>%
  left_join(customers_df, by = "customer_id")

# Reshaping
wide <- df %>% pivot_wider(names_from = category, values_from = amount)
long <- wide %>% pivot_longer(-region, names_to = "category", values_to = "amount")

# Missing data
sum(is.na(df$amount))
df <- df %>% mutate(amount = ifelse(is.na(amount), median(amount, na.rm = TRUE), amount))
df <- df %>% drop_na(customer_id)

# Duplicates
sum(duplicated(df$order_id))
df <- df %>% distinct(order_id, .keep_all = TRUE)

# Date/time
df$order_date <- ymd(df$order_date)
df$order_month <- floor_date(df$order_date, "month")
df$day_of_week <- wday(df$order_date, label = TRUE)
```

---

## 4. WORKED EXAMPLE

**Scenario:** A hospital readmissions dataset with inconsistent date formats, 12% missing values in `discharge_reason`, and duplicate patient records from a recent hospital-system merger. Clean it defensibly.

```python
import pandas as pd

df = pd.read_csv("readmissions_raw.csv")

# --- Cleaning decisions log ---
log = []

# 1. Inspect and standardize inconsistent date formats
#    Some rows use MM/DD/YYYY, others YYYY-MM-DD (common after a system merger)
df["admit_date"] = pd.to_datetime(df["admit_date"], format="mixed", errors="coerce")
n_unparseable = df["admit_date"].isna().sum()
log.append(f"Coerced admit_date to datetime; {n_unparseable} rows had unparseable "
           f"dates and are now NaT — flagged for manual review, not silently dropped.")

# 2. Handle missing discharge_reason (12% missing)
#    Decision: do NOT impute a clinical field with a guessed value — that could
#    distort readmission analysis in a way that's actively misleading.
#    Instead, create an explicit "Unknown" category so the missingness itself
#    remains visible and analyzable.
df["discharge_reason"] = df["discharge_reason"].fillna("Unknown")
log.append("Filled missing discharge_reason with explicit 'Unknown' category "
           "rather than imputing — clinical fields should not be guessed.")

# 3. Resolve duplicate patient records from the merger
#    Decision: two systems may have assigned different patient_id values to the
#    same person. Use name + date_of_birth as a composite key to detect true
#    duplicates, keeping the record with the more complete data.
df["completeness_score"] = df.notna().sum(axis=1)
df = (
    df.sort_values("completeness_score", ascending=False)
      .drop_duplicates(subset=["patient_name", "date_of_birth"], keep="first")
)
log.append("Deduplicated on (patient_name, date_of_birth) composite key, keeping "
           "the more complete record per duplicate pair — patient_id was not "
           "trustworthy alone due to the recent system merger.")

for entry in log:
    print("-", entry)
```

This is what "clean it defensibly" looks like in practice: every non-obvious decision gets a one-line justification, written at the time you make it — not reconstructed later from memory when someone asks.

---

## 5. COMMON MISTAKES

- **Imputing before checking why data is missing.** Missing-completely-at-random, missing-at-random, and missing-not-at-random are genuinely different situations that call for different handling — filling every missing value with the mean regardless of cause is a common shortcut that silently distorts results, especially when missingness itself is informative (e.g., patients who didn't report a field because of the very condition being studied).
- **Using `COUNT(*)`-equivalent logic (`.count()` in pandas, `n()` in dplyr) when you actually need `.count()` on a specific column.** `df.groupby("region").count()` counts non-null values per column, which can differ across columns if some have missing data — a frequent source of subtly wrong aggregate tables.
- **Dropping duplicates before understanding why they exist.** As in the worked example — blindly deduplicating on the wrong key (e.g., `patient_id` alone, when the real issue is two different IDs for the same person) can either fail to remove true duplicates or incorrectly merge two different people.
- **Merging without checking row counts before and after.** A join that unexpectedly multiplies row count (a classic many-to-many join gone wrong) or unexpectedly shrinks it (an inner join silently dropping unmatched rows) is one of the most common silent bugs in wrangling code — always sanity-check `len(df)` before and after a merge.
- **Not qualifying which "date" a date/time operation is anchored to.** Time zone handling and daylight saving transitions are a frequent, underestimated source of off-by-one-day errors in date-based aggregation.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Grouped Aggregation**
Given an `orders` DataFrame with columns `region`, `category`, `amount`, and `order_date`, write pandas code to find the total revenue per region per month, sorted by revenue descending within each region.

<details>
<summary>Hint</summary>
Extract a month period from `order_date`, then `groupby(["region", "month"])`, aggregate `amount` with `sum`, and sort. Consider whether you need `.reset_index()` after grouping to work with the result as a flat table.
</details>

<details>
<summary>Solution</summary>

```python
df["order_month"] = pd.to_datetime(df["order_date"]).dt.to_period("M")

result = (
    df.groupby(["region", "order_month"])["amount"]
      .sum()
      .reset_index()
      .sort_values(["region", "amount"], ascending=[True, False])
)
```
</details>

---

**Exercise 2 — Join Diagnosis**
You merge an `orders` table (10,000 rows) with a `customers` table (8,000 rows) on `customer_id` using a left join, and the result has 10,300 rows. Explain what likely happened and how you'd investigate it.

<details>
<summary>Hint</summary>
A left join should never produce more rows than the left table unless the right table has duplicate keys — meaning multiple rows per `customer_id` in the `customers` table.
</details>

<details>
<summary>Sample Solution</summary>

The row count increased, which means the `customers` table has duplicate `customer_id` values — each `orders` row matching more than one `customers` row gets duplicated in the output. Investigate with:

```python
customers_df["customer_id"].duplicated().sum()
customers_df[customers_df["customer_id"].duplicated(keep=False)].sort_values("customer_id")
```

Then decide: is this a data quality bug in the customers table (needs deduplication before merging), or are there legitimately multiple valid records per customer (e.g., address history) that require a different join strategy?
</details>

---

**Exercise 3 — Full Cleaning Log (Applied)**
Using the `data_jobs` dataset from your EDA project (or another real dataset of your choice), write a cleaning script that: handles at least one missing-data case with an explicit, justified decision (not blind imputation), checks for and resolves duplicates, and produces a written cleaning decisions log like the worked example. This log becomes part of your portfolio documentation.

<details>
<summary>Hint</summary>
Start by profiling missingness with `df.isna().sum()` and duplicates with `df.duplicated().sum()` before deciding on a strategy — don't clean blind. For each decision, write the log entry *as you make the decision*, not afterward from memory.
</details>
