# Module 4: Exploratory Data Analysis (EDA) & Data Visualization

---

## 1. CONCEPT

EDA is the process of understanding a dataset's shape, quality, and patterns *before* any modeling happens. It answers questions like: What does a "typical" row look like? Where are the outliers, and are they errors or real signal? Which variables actually vary meaningfully, and which are nearly constant? Are there relationships between variables worth investigating further?

**Levels of EDA:**
- **Univariate** — examining one variable at a time (distributions, central tendency, spread)
- **Bivariate** — examining relationships between two variables (correlation, grouped comparisons)
- **Multivariate** — examining relationships among three or more variables simultaneously

**Two visualization philosophies you'll use in parallel:**
- **matplotlib/seaborn (Python)** — imperative-ish, layer-by-layer construction; seaborn adds statistical plotting conveniences on top of matplotlib
- **ggplot2 (R)** — built on the **grammar of graphics**, a genuinely different mental model: you declare *data*, *aesthetic mappings* (which variable maps to which visual property), and *geometric objects* (how to draw it), and ggplot2 composes them. This is worth understanding explicitly rather than treating as "just another plotting library" — the mental model transfers to how you think about visualization even back in Python.

**Choosing the right chart** is a skill in itself:
- Distribution of one numeric variable → histogram or density plot
- Comparing a numeric variable across categories → box plot or violin plot (not a bar chart of means, which hides spread)
- Relationship between two numeric variables → scatter plot
- Change over time → line chart
- Part-to-whole composition → stacked bar (use sparingly; pie charts are usually a worse choice than people expect)

---

## 2. WHY IT MATTERS

EDA is where you catch problems before they become expensive. A bimodal distribution that "shouldn't" be there might reveal two distinct customer populations being incorrectly analyzed as one. A scatter plot might reveal a data entry error (a $50,000 coffee order) that would otherwise silently distort a regression model's coefficients. Skipping EDA and going straight to modeling means these issues surface later — as confusing model results — where they're far harder to diagnose.

EDA is also where most real business insight actually gets found. Plenty of valuable DS engagements end at EDA, with no model built at all, because the visualized patterns directly answered the business question.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: matplotlib / seaborn ---
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate
sns.histplot(df["order_amount"], bins=30, kde=True)
sns.boxplot(x=df["order_amount"])

# Bivariate
sns.boxplot(x="region", y="order_amount", data=df)
sns.scatterplot(x="session_duration", y="order_amount", data=df, alpha=0.5)

# Multivariate
sns.scatterplot(x="session_duration", y="order_amount", hue="region", data=df)
sns.heatmap(df[["amount", "quantity", "discount"]].corr(), annot=True, cmap="coolwarm")

# Time series
df.groupby(df["order_date"].dt.to_period("M"))["amount"].sum().plot(kind="line")

plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.show()
```

```r
# --- R: ggplot2 ---
library(ggplot2)

# Univariate
ggplot(df, aes(x = order_amount)) +
  geom_histogram(bins = 30)

# Bivariate
ggplot(df, aes(x = region, y = order_amount)) +
  geom_boxplot()

ggplot(df, aes(x = session_duration, y = order_amount)) +
  geom_point(alpha = 0.5)

# Multivariate (color as a third dimension)
ggplot(df, aes(x = session_duration, y = order_amount, color = region)) +
  geom_point()

# Time series
df %>%
  mutate(month = floor_date(order_date, "month")) %>%
  group_by(month) %>%
  summarise(revenue = sum(amount)) %>%
  ggplot(aes(x = month, y = revenue)) +
  geom_line() +
  labs(title = "Monthly Revenue", x = "Month", y = "Revenue ($)")
```

---

## 4. WORKED EXAMPLE

**Scenario:** An e-commerce company wants to know why cart abandonment rose. You have a `sessions` dataset (session duration, device type, cart value, whether checkout was completed) and a `transactions` dataset.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sessions = pd.read_csv("sessions.csv")

# --- Step 1: Univariate — understand abandonment rate overall ---
abandonment_rate = 1 - sessions["completed_checkout"].mean()
print(f"Overall abandonment rate: {abandonment_rate:.1%}")

# --- Step 2: Bivariate — does abandonment differ by device? ---
by_device = sessions.groupby("device_type")["completed_checkout"].agg(
    abandonment_rate=lambda x: 1 - x.mean(),
    session_count="count"
)
print(by_device)
# Finding: mobile abandonment (38%) is nearly double desktop (21%)

# --- Step 3: Bivariate — does session duration matter? ---
sns.boxplot(x="completed_checkout", y="session_duration_sec", data=sessions)
plt.title("Session Duration: Completed vs. Abandoned Checkouts")
# Finding: abandoned sessions cluster at BOTH very short durations (bounced fast)
#          AND very long durations (got stuck) — a bimodal pattern worth a second look

# --- Step 4: Multivariate — combine device + duration ---
sns.scatterplot(
    x="session_duration_sec", y="cart_value",
    hue="completed_checkout", style="device_type",
    data=sessions, alpha=0.6
)
# Finding: high cart-value, long-duration mobile sessions abandon at the highest rate —
#          suggests a mobile checkout friction point specifically for larger orders

# --- Step 5: Translate to a business finding ---
findings = """
1. Mobile abandonment (38%) is nearly 2x desktop (21%).
2. Abandoned sessions show a bimodal duration pattern: fast bounces (likely
   pricing/trust issues) and slow, stuck sessions (likely a usability issue).
3. The slow-and-stuck pattern concentrates in mobile sessions with high cart
   values — consistent with a mobile checkout friction point on larger orders.
RECOMMENDATION: Prioritize a UX audit of the mobile checkout flow, specifically
for high-cart-value sessions, before broader abandonment-reduction initiatives.
"""
print(findings)
```

Notice the structure: each visualization step is chosen to answer a specific question raised by the previous one, and the process ends with a plain-English translation — the dual deliverable habit from Module 0.

---

## 5. COMMON MISTAKES

- **Using a bar chart of means to compare groups.** This hides the spread entirely — two groups can have identical means and wildly different variance, and a bar chart makes them look the same. Use a box or violin plot instead whenever comparing distributions across categories.
- **Not checking axis scales before interpreting a chart.** A truncated y-axis can make a 2% difference look dramatic; an auto-scaled axis across subplots can make genuinely different magnitudes look comparable. Always sanity-check the axis range against the actual data.
- **Overplotting on scatter plots with large datasets.** Thousands of overlapping points render as a solid blob, hiding density patterns — use `alpha` (transparency), hexbin plots, or sampling to reveal the actual density.
- **Treating correlation heatmaps as causal evidence.** A strong correlation in EDA is a *lead to investigate*, not a conclusion — confusing the two is one of the most common errors carried from EDA into premature business recommendations.
- **Stopping EDA the moment you find one plausible story.** The bimodal duration finding in the worked example could easily have been missed if the analysis stopped at "mobile abandons more." Real EDA means continuing to probe even after finding a first plausible explanation, to check whether it's the *whole* story or just part of it.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Chart Selection**
For each scenario, state which chart type you'd use and why:
a) Comparing the distribution of order values across 4 customer segments
b) Showing how daily active users changed over the past 12 months
c) Showing the relationship between marketing spend and revenue across 50 campaigns
d) Showing what percentage of total revenue came from each of 5 product categories

<details>
<summary>Sample Solution</summary>

a) Box or violin plot — shows the full distribution (median, spread, outliers) per segment, not just a single summary number.
b) Line chart — time series data is inherently ordered and continuous; a line shows trend clearly.
c) Scatter plot — examining the relationship between two continuous numeric variables.
d) Stacked bar or a simple horizontal bar chart of percentages — generally clearer than a pie chart once you have more than 3-4 categories.
</details>

---

**Exercise 2 — Bimodal Detection**
You plot a histogram of customer session durations and notice it's clearly bimodal (two humps) rather than the single-peaked distribution you expected. List at least 3 plausible real-world explanations, and describe how you'd investigate which one is correct.

<details>
<summary>Hint</summary>
Think about what could cause two distinct populations within "session duration": different user intents, different device types, bot traffic mixed with human traffic, or a UX issue causing some users to get stuck.
</details>

<details>
<summary>Sample Solution</summary>

Plausible explanations: (1) two distinct user intents — quick single-item purchases vs. extended browsing sessions; (2) bot/automated traffic mixed in with real human sessions, producing an artificially short cluster; (3) a checkout usability issue causing a subset of users to get stuck for unusually long sessions; (4) a mix of mobile and desktop sessions with genuinely different typical durations.

Investigation approach: break the histogram down by a plausible splitting variable (device type, user-agent/bot flag, whether checkout completed) using `hue=`/color in the plot — if one of these variables cleanly separates the two humps, that's very likely your explanation. If none of them do, dig further before concluding.
</details>

---

**Exercise 3 — Full EDA with Executive Summary (Applied)**
Using the `data_jobs` dataset from your EDA project, perform a full EDA following the univariate → bivariate → multivariate progression shown in the worked example. Produce at least 5 concrete findings, each backed by a specific chart, and write a 1-page executive summary translating the findings into plain English for a non-technical stakeholder. This builds directly toward your capstone deliverable in Module 15.

<details>
<summary>Hint</summary>
Start univariate (what does the distribution of median salary look like? of job postings by role?), move to bivariate (does salary differ meaningfully by skill or region?), then multivariate (does the salary-by-skill relationship change depending on seniority level?) — following the same escalating structure as the worked example.
</details>
