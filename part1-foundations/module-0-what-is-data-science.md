# Module 0: What Data Science Actually Is (Orientation)

---

## 1. CONCEPT

Data science is the practice of turning raw, messy data into decisions. It sits at the intersection of three things: **programming** (to handle data at scale), **statistics** (to draw valid conclusions from it), and **domain knowledge** (to know which questions are worth asking in the first place).

The **data science lifecycle** has six stages:

1. **Business question** — What decision does this analysis need to inform?
2. **Data collection** — Where does the data live, and how do you get it?
3. **Cleaning** — Raw data is never analysis-ready. This stage usually takes the most time.
4. **Exploratory Data Analysis (EDA)** — Understand the data's shape, quality, and patterns before modeling.
5. **Modeling** — Build a statistical or ML model to answer the question (not every project needs this stage).
6. **Communication & deployment** — Present findings or ship a model into a system people actually use.

Most beginners assume the job is mostly step 5. In practice, steps 1–4 consume 70–80% of real project time.

### Data Science vs. Adjacent Roles

| Role | Primary Focus | Typical Output |
|---|---|---|
| **Data Analyst** | Descriptive analytics on existing data | Dashboards, reports |
| **Data Scientist** | Predictive/prescriptive analytics, experimentation | Models, statistical findings, recommendations |
| **Data Engineer** | Building pipelines that move and store data reliably | ETL/ELT pipelines, data warehouses |
| **ML Engineer** | Productionizing and scaling models | Deployed, monitored ML systems |

These roles overlap heavily in smaller companies and are more distinct in larger ones. A data scientist who can also engineer a pipeline and deploy a model (your trajectory, given your DE background) is unusually valuable — most DS practitioners can't do both ends.

### Types of Analytics

- **Descriptive** — What happened? ("Revenue dropped 15% last quarter.")
- **Diagnostic** — Why did it happen? ("Revenue dropped because of a 30% drop in the Northeast region.")
- **Predictive** — What will happen? ("Revenue will likely drop another 10% next quarter if the trend continues.")
- **Prescriptive** — What should we do about it? ("Reallocating marketing spend to the Northeast is projected to recover 60% of the loss.")

Most real DS work moves through all four in sequence.

---

## 2. WHY IT MATTERS

Framing determines everything downstream. A poorly framed question leads to a technically correct analysis that answers the wrong thing — a far more common failure mode than "the model was wrong." Companies rarely hand you a clean, well-formed data science problem. They hand you a business complaint ("sales are down," "users are churning," "support tickets are piling up") and expect you to translate it into something answerable with data.

Getting this translation right — before writing a single line of code — is what separates a data scientist from someone who runs models. It's also the single most tested skill in DS interviews, disguised as "how would you approach X?" questions.

---

## 3. SYNTAX REFERENCE (Framework Reference)

There's no code in this module. Instead, here's a repeatable framework for turning a vague business problem into a data question:

```
STEP 1: State the business problem in one sentence.
        (e.g., "Revenue dropped 15% last quarter.")

STEP 2: List every plausible driver, without judging them yet.
        (e.g., region, product category, customer segment,
         marketing spend, seasonality, competitor activity,
         pricing changes, supply issues)

STEP 3: For each driver, ask: "Do we have data that could
        confirm or rule this out?"
        (Cross off drivers with no available data — note them
        as a limitation, don't ignore them.)

STEP 4: Convert the remaining drivers into specific,
        answerable sub-questions.
        (e.g., "Did the revenue drop concentrate in one region?"
         "Did it concentrate in one product category?"
         "Did it happen gradually or in a step-change?")

STEP 5: Rank sub-questions by (a) how likely they are to be
        the real driver and (b) how cheap they are to check.
        Check cheap + likely first.
```

---

## 4. WORKED EXAMPLE

**Business problem:** "Our e-commerce revenue dropped 15% last quarter, and leadership wants to know why."

**Step 2 — Plausible drivers (unfiltered):**
Regional slowdown, category-specific slowdown, fewer new customers, lower repeat purchase rate, price increases, a site outage, seasonality, a competitor's promotion, ad spend cuts, inventory stockouts.

**Step 3 — Data availability check:**
- Region, category, customer segment, order dates → available in the orders database. Keep.
- Ad spend → available in marketing reporting. Keep.
- Competitor promotions → not in any internal dataset. Flag as an unmeasurable limitation, don't drop the hypothesis entirely.
- Site outage → check incident logs. Keep.

**Step 4 — Specific sub-questions:**
1. Is the revenue drop concentrated in specific regions or spread evenly?
2. Is it concentrated in specific product categories?
3. Did new customer acquisition drop, or did existing customers buy less?
4. Does the drop correlate with any known outages or ad spend cuts?
5. Is this quarter-over-quarter drop unusual relative to the same quarter last year (i.e., is it seasonal)?

**Step 5 — Prioritization:**
Question 5 (seasonality) is cheapest to check — one query against last year's data — and could immediately rule out a false alarm. Check it first. Questions 1–3 are next cheapest (a GROUP BY away) and highly likely to be informative. Question 4 requires joining two different data sources and is checked last.

This is the actual first hour of a real DS engagement — before any modeling, before any dashboard. Most of the "hard thinking" in data science happens right here.

---

## 5. COMMON MISTAKES

- **Jumping straight to modeling.** Building a churn/revenue prediction model before confirming what's actually driving the metric wastes time and often answers a question nobody asked.
- **Treating "no data available" as "hypothesis eliminated."** Missing data means a blind spot, not proof the driver doesn't matter — always state it as a limitation, not a conclusion.
- **Answering the literal question instead of the underlying decision.** If leadership asks "why did revenue drop," they usually actually want to know "what should we do about it" — keep the prescriptive end goal in view even while doing descriptive/diagnostic work.
- **Skipping the seasonality/baseline check.** A huge fraction of "urgent" business problems turn out to be normal seasonal variation. Always check this first — it's the cheapest possible check and often resolves the whole question.
- **Over-scoping the first pass.** Trying to check all 10 drivers simultaneously in one sprawling analysis instead of ranking and checking the cheap, likely ones first.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Problem Framing**
A SaaS company's leadership says: "Our customer churn rate went up last month, and we need to fix it." Apply the 5-step framework above. Write out at least 6 plausible drivers, note which ones you'd realistically have data for, and produce at least 4 specific sub-questions ranked by priority.

<details>
<summary>Hint</summary>
Think about churn drivers across the customer lifecycle: onboarding experience, pricing changes, product outages/bugs, support ticket volume/resolution time, competitor activity, contract renewal timing, and usage patterns before cancellation.
</details>

<details>
<summary>Sample Solution</summary>

**Plausible drivers:** pricing change, a recent product outage, a competitor's new feature launch, worsening support response times, a cohort of customers hitting contract renewal simultaneously, a recent onboarding flow change, declining product usage before cancellation, a specific customer segment (e.g., SMB vs. enterprise) driving the increase.

**Data availability:**
- Pricing changes, outages, support tickets, renewal dates, usage logs, customer segment → likely available internally.
- Competitor feature launches → not in internal data; flag as unmeasurable but worth a qualitative check (news, sales team feedback).

**Prioritized sub-questions:**
1. Is this month's churn rate unusual relative to the same month last year (seasonality check — cheapest, do first)?
2. Is the churn concentrated among customers who hit contract renewal this month?
3. Did average product usage decline for churned customers in the weeks before cancellation?
4. Is churn concentrated in one customer segment or plan tier?
</details>

---

**Exercise 2 — Role Boundaries**
For each of the following tasks, identify whether it's primarily a Data Analyst, Data Scientist, Data Engineer, or ML Engineer task (some may span two roles — note that too):

a) Building a nightly pipeline that loads sales data from 5 regional databases into a central warehouse
b) Designing and analyzing an A/B test for a new checkout flow
c) Building a dashboard showing weekly active users by region
d) Deploying a fraud-detection model behind an API with monitoring for performance drift
e) Investigating whether customer lifetime value differs meaningfully by acquisition channel

<details>
<summary>Sample Solution</summary>

a) Data Engineer
b) Data Scientist
c) Data Analyst
d) ML Engineer (with the model itself likely built by a Data Scientist)
e) Data Scientist
</details>

---

**Exercise 3 — Your Own Framing (Portfolio Prep)**
Pick a real dataset you plan to use later in this course (e.g., the `data_jobs` dataset from your EDA project). Write a one-page problem-framing document using the 5-step framework, treating the dataset as if a stakeholder handed you a vague business question about it. Save this document — you'll build directly on it in Module 4 (EDA).
</details>
