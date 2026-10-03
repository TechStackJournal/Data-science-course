# Module 13: NLP & Working with Text/Unstructured Data

---

## 1. CONCEPT

Text is everywhere in real business data — reviews, support tickets, contracts, emails — and is chronically under-covered in beginner DS curricula despite being one of the most common real-world data types encountered on the job. This module covers preprocessing, classical text representations, embeddings, and practical use of modern pretrained transformer models.

**Text preprocessing:**
- **Tokenization** — splitting text into words/subwords/tokens, the basic unit models operate on
- **Stemming/lemmatization** — reducing words to a root form (stemming is a crude rule-based cut, e.g., "running" → "run"; lemmatization uses actual linguistic knowledge to find the true dictionary root, e.g., "better" → "good"); lemmatization is more accurate but slower
- **Stopwords** — common words ("the," "is," "and") often removed for classical bag-of-words approaches, since they carry little topical signal; **note:** stopword removal is often skipped or reconsidered for modern transformer-based approaches, which use context in ways that can make "unimportant" words informative

**Classical representations:**
- **Bag-of-words** — represents a document as word counts, ignoring order and grammar entirely; simple, interpretable, but loses all sequential/contextual meaning
- **TF-IDF** (Term Frequency–Inverse Document Frequency) — weights words by how frequent they are in a document relative to how common they are across all documents, down-weighting generically common words and up-weighting distinctive ones; a solid classical baseline, still genuinely useful and fast

**Word embeddings:**
- **Word2Vec** (conceptual understanding) — learns dense vector representations of words such that semantically similar words end up close together in vector space, capturing meaning in a way raw bag-of-words cannot
- **Transformer embeddings** (modern standard) — contextual, meaning the same word gets a different embedding depending on surrounding context (unlike Word2Vec's fixed, context-independent embedding per word) — this is the representation underlying BERT, GPT, and most current NLP work

**Using pretrained transformer models (Hugging Face):**
- For most practical business NLP tasks, you don't need to train a transformer from scratch — using a pretrained model (optionally fine-tuned on your specific data) for classification, sentiment analysis, or Named Entity Recognition (NER) is standard practice and dramatically more efficient than building from scratch
- **When fine-tuning is worth it** — when your domain language differs significantly from general text (medical, legal, highly technical jargon) or you need meaningfully higher accuracy than a pretrained model + good prompting/zero-shot approach provides
- **When a pretrained model alone suffices** — for many general business use cases (sentiment, general topic classification), a pretrained model used directly or with light prompting gets you most of the way there with far less effort

---

## 2. WHY IT MATTERS

Businesses generate enormous volumes of unstructured text that traditional structured-data skills simply can't touch — support tickets, reviews, survey free-text responses, contracts. A data scientist who can only handle structured tabular data is locked out of a huge and growing share of real analytical opportunities. NLP skills specifically — even at a practical, pretrained-model-usage level rather than research depth — are increasingly a baseline expectation rather than a specialization, precisely because so much valuable business data lives in unstructured text.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: preprocessing, TF-IDF, and Hugging Face transformers ---
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(cleaned_texts)

# Using a pretrained transformer for classification (zero-shot / off-the-shelf)
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")
result = sentiment_analyzer("The product broke after two days, very disappointed.")
# [{'label': 'NEGATIVE', 'score': 0.998}]

zero_shot_classifier = pipeline("zero-shot-classification")
result = zero_shot_classifier(
    "The app keeps crashing every time I try to check out.",
    candidate_labels=["billing issue", "technical bug", "feature request", "praise"]
)

ner_pipeline = pipeline("ner", grouped_entities=True)
entities = ner_pipeline("Contact support at Acme Corp regarding order #4521.")

# Getting embeddings directly for downstream tasks (e.g., clustering support tickets)
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(list_of_ticket_texts)
# embeddings can now feed into k-means (Module 10) for topic discovery
```

```r
# --- R: tidytext (classical) + reticulate (bridge to Python transformers) ---
library(tidytext)
library(dplyr)

tickets <- tibble(id = 1:3, text = c("...", "...", "..."))

tidy_tickets <- tickets %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word")  # remove stopwords

tfidf_result <- tidy_tickets %>%
  count(id, word) %>%
  bind_tf_idf(word, id, n)

# R has no mature native transformer ecosystem comparable to Hugging Face —
# the standard practice is calling Python transformer models via reticulate,
# or handling the NLP-heavy step in Python and continuing analysis in R
library(reticulate)
transformers <- import("transformers")
classifier <- transformers$pipeline("sentiment-analysis")
result <- classifier("This service was excellent.")
```

---

## 4. WORKED EXAMPLE

**Scenario:** A company has 50,000 unstructured customer support tickets and wants to know the top 5 recurring complaint categories, with no predefined label taxonomy.

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

tickets = pd.read_csv("support_tickets.csv")

# --- Step 1: Generate embeddings (no manual feature engineering needed —
#     the pretrained model already captures semantic meaning) ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(tickets["ticket_text"].tolist(), show_progress_bar=True)

# --- Step 2: Cluster the embeddings to discover categories
#     (directly reusing the Module 10 clustering workflow) ---
best_k, best_score = 2, -1
for k in range(4, 12):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    if score > best_score:
        best_k, best_score = k, score

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
tickets["cluster"] = kmeans.fit_predict(embeddings)
print(f"Discovered {best_k} clusters (silhouette: {best_score:.3f})")

# --- Step 3: Understand each cluster by sampling representative tickets ---
for cluster_id in sorted(tickets["cluster"].unique()):
    sample = tickets[tickets["cluster"] == cluster_id]["ticket_text"].sample(5, random_state=42)
    print(f"\n--- Cluster {cluster_id} sample tickets ---")
    for text in sample:
        print(f"  - {text[:100]}")
    # Manually reading samples reveals, e.g.:
    # Cluster 0: billing/refund complaints
    # Cluster 1: login/authentication issues
    # Cluster 2: shipping delays
    # Cluster 3: product defects
    # Cluster 4: positive feedback / praise (not a complaint at all — a useful finding itself)

# --- Step 4: Label clusters and build an automatic classifier for NEW tickets ---
# Once clusters are manually labeled based on sampling, train a lightweight
# classifier so future tickets can be auto-categorized without re-clustering
cluster_labels = {
    0: "Billing/Refund", 1: "Login/Auth", 2: "Shipping Delay",
    3: "Product Defect", 4: "Positive Feedback"
}
tickets["category"] = tickets["cluster"].map(cluster_labels)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, tickets["category"], test_size=0.2, random_state=42
)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
print(f"New-ticket auto-categorization accuracy: {classifier.score(X_test, y_test):.3f}")
```

This pattern — embed, cluster to discover structure with no labels, manually label the discovered clusters by sampling, then train a lightweight supervised classifier on the now-labeled data for ongoing use — is a genuinely common and powerful real-world workflow for exactly this "we have text but no taxonomy" situation.

---

## 5. COMMON MISTAKES

- **Applying heavy stopword removal and stemming before using a transformer model.** This preprocessing pipeline was designed for bag-of-words/TF-IDF approaches; transformer models are trained on natural, unprocessed text and generally perform *worse* if you strip context away before feeding them input — match your preprocessing to the technique you're actually using.
- **Assuming TF-IDF or bag-of-words captures meaning the way embeddings do.** "The service was not good" and "The service was good" share almost identical bag-of-words representations (same words, one negation) but opposite meanings — this is a classic failure case of classical representations that contextual embeddings handle much better.
- **Skipping the "read a sample of each cluster" step in unsupervised text categorization.** As shown in the worked example — an algorithm-generated cluster number means nothing without a human actually reading representative examples to understand and name what the cluster represents; skipping this step is the text-data version of Module 10's "stopping at cluster labels" mistake.
- **Using a general-purpose sentiment model on domain-specific text without validation.** A sentiment model trained on general text can misfire on domain jargon (e.g., medical or legal text where "positive" and "negative" carry different connotations) — always validate a pretrained model's outputs against a manually labeled sample from your actual domain before trusting it at scale.
- **Not handling class imbalance in ticket/complaint categorization.** Just as in Module 7, if one category (e.g., "general inquiry") vastly outnumbers others, a naive classifier can achieve high accuracy while being useless for the rarer, often more actionable categories (e.g., "safety complaint") — apply the same imbalance-handling principles from Module 7 here.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Representation Selection**
For each scenario, choose bag-of-words/TF-IDF or transformer embeddings, and justify:
a) A quick baseline spam classifier for a small dataset (500 emails), where training speed and simplicity matter most
b) Detecting sarcasm or nuanced sentiment in product reviews
c) Finding documents similar in *meaning* even when they use different specific wording

<details>
<summary>Sample Solution</summary>

a) **TF-IDF** — fast, simple, interpretable, and sufficient for a small dataset where a quick baseline is the goal; the complexity of transformer embeddings isn't justified here.
b) **Transformer embeddings** — nuance and sarcasm require contextual understanding that bag-of-words/TF-IDF fundamentally cannot capture, since they ignore word order and context.
c) **Transformer embeddings** — this is exactly the semantic similarity use case embeddings excel at; TF-IDF would only match on shared vocabulary, missing documents that express the same meaning with different words.
</details>

---

**Exercise 2 — Interpreting a Zero-Shot Classification**
Using the `zero_shot_classifier` pipeline shown in the syntax reference, classify the text "I love how fast the delivery was, but the packaging was damaged" against the labels `["shipping speed praise", "packaging complaint", "product quality issue"]`. Predict which label(s) you'd expect the highest confidence scores on, and explain why zero-shot classification can handle a text with mixed sentiment more gracefully than a single-label bag-of-words classifier might.

<details>
<summary>Sample Solution</summary>

Expect meaningfully high scores on both "shipping speed praise" and "packaging complaint," since the text genuinely contains both sentiments — zero-shot classification with multi-label scoring (rather than forcing a single label) naturally reflects that a single piece of text can express multiple, even conflicting, sentiments simultaneously. A traditional single-label bag-of-words classifier forced to pick one category would lose this nuance, potentially miscategorizing the ticket entirely depending on which sentiment's vocabulary happened to dominate the word counts.
</details>

---

**Exercise 3 — Full Unsupervised Text Categorization (Applied)**
Using a real or simulated set of unstructured text (product reviews, support tickets, or survey free-text — a public dataset like Amazon or Yelp reviews works well), follow the full worked example workflow: generate embeddings, cluster to discover categories, manually label clusters by sampling representative texts, and train a lightweight classifier for auto-categorizing new text going forward. Summarize the discovered categories and their relative frequency in a short business-facing write-up — this is a strong, distinctive portfolio piece since unstructured text projects are less common than yet another tabular Kaggle dataset.
