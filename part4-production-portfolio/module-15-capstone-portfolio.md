# Module 15: Capstone Project & Portfolio Development

---

## 1. CONCEPT

This module has no new technical material — it's where everything from Modules 0–14 gets integrated into one project and packaged for a hiring manager to evaluate in the few minutes they'll actually spend on your portfolio. Three components:

**The capstone project itself:**
A single end-to-end project, run on a genuinely messy real dataset (not a pre-cleaned Kaggle competition dataset), taken through the full lifecycle from Module 0: business framing → EDA → modeling → deployment → written communication. The goal is demonstrating breadth (you can do every stage) and depth (each stage is done well, not just checked off) simultaneously.

**Portfolio structure standards:**
- **README quality** — states the business problem before the technical approach; a reader should understand *why* the project exists before *how* it works
- **Reproducibility** — a `requirements.txt`/environment file, clear setup instructions, and ideally a one-command way to reproduce key results; an unreproducible project signals sloppiness regardless of how good the underlying work is
- **Architecture/pipeline documentation** — for any project with multiple stages (data ingestion → cleaning → modeling → serving), a simple diagram or clear written description of how pieces connect
- **Decision documentation** — the "why did you do X instead of Y" reasoning, not just the final code; this is what actually differentiates a portfolio piece from a tutorial completion

**Case-study interview practice:**
DS interviews frequently present an ambiguous, open-ended prompt ("how would you detect fake reviews?" "how would you decide whether to launch this feature?") and evaluate the *structured thinking process*, not a single correct answer. This directly reuses the Module 0 framework (business problem → plausible drivers → data availability → specific sub-questions → prioritization) as a live, verbal skill, not just a written one.

**Resume/story positioning:**
Translating projects into concrete, quantified achievements ("built a churn model" is weak; "built a churn prediction pipeline that identified a data leakage issue inflating apparent accuracy by 24 points, then shipped a corrected model with documented production monitoring" is a story an interviewer remembers) — and, specifically for your situation, explicitly positioning the data engineering background as a differentiator rather than a separate, unrelated track.

---

## 2. WHY IT MATTERS

Hiring managers and technical interviewers see enormous numbers of near-identical portfolio projects — another Titanic dataset, another Iris classifier, another basic Kaggle housing-price notebook with no real framing. What differentiates a candidate isn't finding a slightly more obscure dataset; it's demonstrating the full lifecycle thinking (business framing, defensible cleaning decisions, correct metric choice, leakage awareness, deployment mindset, honest communication of limitations) that this entire course has built toward. A capstone that visibly demonstrates that thinking, end to end, on one coherent project is worth more than five disconnected notebooks that each show a single skill in isolation.

---

## 3. FRAMEWORK: Building the Capstone

**Step 1 — Choose a dataset that simulates your target job's industry.**
Suggested options, matched to different target roles:
- **Healthcare readmission risk** — if targeting healthcare/insurance-adjacent DS roles; emphasizes defensible cleaning (Module 3) and interpretability (Module 6)
- **Retail demand forecasting with external shocks** — if targeting retail/supply-chain roles; emphasizes Module 11's regime-change handling
- **Fraud detection with class imbalance** — if targeting fintech/banking roles; emphasizes Module 7/8's imbalanced classification and leakage rigor
- **Customer support ticket triage using NLP** — if targeting a general SaaS/product-adjacent role; emphasizes Module 13's unstructured data handling

Given your Azure data engineering trajectory specifically, consider a project where the "data pipeline" stage is deliberately non-trivial (e.g., simulating data arriving from multiple inconsistent sources, requiring real ingestion/standardization work like Module 1's worked example) — this is exactly where you can visibly demonstrate the DS+DE combination that's your stated differentiator.

**Step 2 — Structure the project repository:**
```
capstone-project/
├── README.md                  # business framing FIRST, then technical summary
├── requirements.txt           # or environment.yml
├── data/
│   ├── raw/                   # original, untouched data (or a note on where to get it)
│   └── processed/             # cleaned outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_modeling.ipynb
├── src/                       # production-style code, not just notebook cells
│   ├── preprocessing.py
│   ├── train.py
│   └── serve.py                # FastAPI/plumber app from Module 14
├── docs/
│   ├── cleaning_decisions_log.md
│   ├── model_card.md           # what the model does, its limitations, intended use
│   └── monitoring_plan.md
└── Dockerfile
```

**Step 3 — Write the README as a hiring manager would actually read it:**
```
1. Business Problem (2-3 sentences, non-technical)
2. Approach Summary (what you did, at a high level)
3. Key Findings (2-4 bullet points with concrete numbers)
4. Technical Details (link to notebooks/docs for depth)
5. Limitations & What I'd Do With More Time (shows maturity, not weakness)
6. How to Run This (reproducibility)
```

**Step 4 — Prepare the case-study interview walkthrough.**
Practice presenting the project verbally in under 5 minutes, structured as: problem → approach → one interesting complication you hit and how you resolved it → result → limitation you're aware of. The "interesting complication" is the part that actually demonstrates skill — anyone can present a clean success story; walking through a leakage bug you caught, or a metric choice you had to defend, is what's memorable.

---

## 4. WORKED EXAMPLE: A Complete Capstone Walkthrough

**Chosen scenario:** Fraud detection with class imbalance (fintech-oriented, reusing and extending Module 7/8/9/14 work into one coherent narrative).

**1. Business Problem:**
"A payments company needs to flag potentially fraudulent transactions in real time, minimizing both missed fraud (costly) and false alarms (costly in customer friction), using a defensible, monitored model."

**2. Data pipeline (DE-flavored, per Step 1's suggestion):**
Simulate transaction data arriving from 3 inconsistent regional processing systems (reusing the Module 1/2 ingestion pattern) — different column names, different date formats — standardized into one clean schema before any modeling begins. This section of the project explicitly showcases the DE+DS combination.

**3. EDA (Module 4):**
Univariate: confirm severe class imbalance (0.3% fraud). Bivariate: fraud rate by transaction amount, time of day, merchant category. Multivariate: interaction between amount and merchant category. Produces the initial hypothesis that fraud concentrates in specific merchant categories at unusual hours.

**4. Feature engineering & leakage audit (Module 8):**
Build features (transaction velocity, merchant risk score, time-since-last-transaction). Explicitly audit every feature for leakage — document which were excluded and why (e.g., a "chargeback_filed" field that's only populated after the fraud has already been confirmed, structurally identical to the `cancellation_request_date` leakage example from Module 8).

**5. Modeling (Modules 7 & 9):**
Baseline Random Forest with class weighting. Tuned XGBoost via randomized search. Cost-based threshold tuning against explicit assumed costs (missed fraud vs. false alarm), exactly as in Module 7's worked example. SHAP explanations for both global feature importance and individual flagged transactions (Module 9).

**6. Deployment (Module 14):**
FastAPI serving endpoint, Dockerized, with MLflow experiment tracking logging the final model's parameters and metrics. A monitoring plan document specifying which features get drift-checked weekly and what triggers a retraining review.

**7. Documented limitations (the honesty that builds credibility):**
"This model was trained on synthetic/public fraud data and would need revalidation on real transaction patterns before production use. The cost assumptions used for threshold tuning ($500 per missed fraud, $20 per false alarm) were illustrative — real deployment would require actual business input on these figures. With more time, I would incorporate a graph-based feature capturing transaction network patterns between accounts, which is a known strong signal in fraud detection that this version doesn't yet include."

This last section is what most beginner portfolios skip — and it's specifically what signals seniority-adjacent judgment to an experienced reviewer.

---

## 5. COMMON MISTAKES

- **Presenting the capstone as a clean success story with no complications.** Real projects hit real problems (leakage, drift, ambiguous metric tradeoffs) — a portfolio that presents zero friction either means the project was too simple to be interesting, or the friction was hidden rather than resolved. Show the complication and the resolution; it's more credible and more memorable than a frictionless narrative.
- **Burying the business problem under technical detail.** A README that opens with library imports and model architecture, with the actual business problem mentioned in passing three paragraphs in, loses a reviewer who's skimming — lead with the problem, every time.
- **Treating the capstone as a checklist of techniques to demonstrate rather than a coherent project.** Cramming in unsupervised learning, deep learning, AND time series into one narrative just to "show all the skills" usually produces a worse project than a focused one that goes deep on the 3-4 techniques the actual problem calls for. Depth on the relevant techniques beats breadth for its own sake.
- **Skipping the "limitations and what I'd do with more time" section.** This is one of the highest-signal parts of a strong portfolio project precisely because it's rare — it demonstrates the self-awareness and honesty that separates someone who understands a project's real boundaries from someone who's just executed a tutorial.
- **Not being able to verbally walk through the project without re-reading your own code.** If you can't explain your leakage audit or your threshold-tuning reasoning from memory in an interview, it signals the decision was copied from a pattern rather than genuinely reasoned through — practice the 5-minute verbal walkthrough until it's natural, not memorized-sounding.

---

## 6. PRACTICE EXERCISES (Final Capstone Build)

**Exercise 1 — Project Selection & Framing**
Choose your capstone scenario from Step 1's options (or a variation suited to your specific target roles). Write the full "Business Problem" and "Approach Summary" sections of your README *before* writing any code — this forces the framing discipline from Module 0 to happen first, not retrofitted after the fact.

<details>
<summary>Hint</summary>
If you're unsure which scenario to pick, weight it toward whichever industry appears most often in the job postings you're actually targeting — a portfolio project that mirrors your target employer's domain is disproportionately more persuasive than a generically "impressive" one in an unrelated industry.
</details>

---

**Exercise 2 — Full Build**
Execute the complete capstone following the repository structure and workflow in Section 3, incorporating techniques from at minimum: Module 3 (defensible cleaning with a decisions log), Module 4 (EDA with a dual technical/executive deliverable), Module 7 or 6 (an appropriately evaluated supervised model), Module 8 (an explicit leakage audit), and Module 14 (a deployed, monitored endpoint). Choose 1-2 additional modules (9, 10, 11, 12, or 13) that genuinely fit your chosen scenario — resist the urge to force in every technique from the course.

---

**Exercise 3 — The 5-Minute Verbal Walkthrough**
Record yourself (audio is enough, video not required) presenting your capstone in under 5 minutes using the structure from Step 4: problem → approach → one interesting complication and its resolution → result → limitation. Listen back and check: did you lead with the business problem? Did you name a specific number (a metric, a percentage, a dollar-cost assumption) rather than staying purely qualitative? Did you mention a limitation without being asked? Refine and re-record until all three are true — this is the exact skill a real DS interview screen is evaluating.

---

## Course Complete

That's all 15 modules across 4 parts — from "what is a data question" through a fully deployed, monitored, portfolio-ready capstone. The through-lines that were seeded in Module 0 (problem framing before code, dual technical/executive deliverables, defensible decision logs, leakage vigilance, and honest limitation-reporting) were designed to compound across every module, so the capstone isn't really module 15's work alone — it's the accumulated discipline from all 14 modules before it, applied to one coherent project.
