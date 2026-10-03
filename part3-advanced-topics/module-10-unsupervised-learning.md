# Module 10: Unsupervised Learning

---

## 1. CONCEPT

Unsupervised learning finds structure in data with no labeled outcome to predict against. This module covers clustering, dimensionality reduction, and anomaly detection — and, critically, how to evaluate results when there's no ground truth to check against.

**Clustering:**
- **k-means** — partitions data into k clusters by minimizing within-cluster variance; fast and scalable, but assumes roughly spherical, similarly-sized clusters and requires choosing k in advance
- **Hierarchical clustering** — builds a tree (dendrogram) of nested clusters; doesn't require pre-specifying k, and the dendrogram itself is useful for exploring structure at different granularities, but scales poorly to large datasets
- **DBSCAN** — density-based clustering that finds arbitrarily-shaped clusters and naturally identifies outliers as noise points, without requiring k in advance; sensitive to its two parameters (`eps`, `min_samples`) and struggles with clusters of varying density

**Choosing k (for k-means):**
- **Elbow method** — plot within-cluster variance against k, look for the point where additional clusters stop meaningfully reducing variance
- **Silhouette score** — measures how similar each point is to its own cluster versus other clusters, ranges from -1 to 1; higher is better, and it can be computed for a range of k values to help choose

**Dimensionality reduction:**
- **PCA (Principal Component Analysis)** — finds orthogonal directions of maximum variance in the data; linear, interpretable in terms of variance explained, commonly used both for visualization and as a preprocessing step before modeling
- **t-SNE** — non-linear, excellent for visualizing high-dimensional cluster structure in 2D/3D, but distances between distant clusters in a t-SNE plot are not meaningful (a common misinterpretation) and results can vary between runs
- **UMAP** — similar goals to t-SNE, generally faster and better at preserving some global structure, increasingly the default choice over t-SNE for exploratory visualization

**Anomaly detection:**
- **Isolation Forest** — isolates anomalies by randomly partitioning data; anomalies require fewer splits to isolate than normal points, which the algorithm exploits directly
- **Statistical approaches** — simple z-score or IQR-based thresholds; less powerful than Isolation Forest for multivariate anomalies but fast and interpretable for univariate cases

**Evaluating unsupervised results without ground truth:**
- **Silhouette score** and other internal metrics give a mathematical sense of cluster quality
- But the metric that actually matters most in practice is **business validation** — do the discovered clusters/anomalies make sense to someone with domain knowledge, and can they be acted on? A clustering with a great silhouette score that produces business-meaningless groups has failed at its actual job.

---

## 2. WHY IT MATTERS

Unsupervised learning is where a large share of real business analytics work happens — customer segmentation, anomaly/fraud flagging, topic discovery in text — precisely because labeled data is often unavailable or expensive to create. The skill that separates a good unsupervised analysis from a mediocre one isn't the algorithm; it's the translation step from abstract mathematical clusters into named, actionable business groups. Most unsupervised projects that fail, fail at exactly this translation step, not at the clustering math itself.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: scikit-learn ---
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

# Always scale before distance-based clustering
X_scaled = StandardScaler().fit_transform(X)

# k-means with elbow + silhouette to choose k
inertias, silhouettes = [], []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Fit final model at chosen k
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)  # -1 indicates noise/outlier points

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained: {pca.explained_variance_ratio_}")

# UMAP (separate library)
import umap
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
```

```r
# --- R: base + cluster packages ---
library(cluster)
library(factoextra)  # visualization helpers for clustering

X_scaled <- scale(X)

# k-means with elbow/silhouette
fviz_nbclust(X_scaled, kmeans, method = "wss")        # elbow plot
fviz_nbclust(X_scaled, kmeans, method = "silhouette")  # silhouette plot

km_result <- kmeans(X_scaled, centers = 4, nstart = 25)

# Hierarchical clustering
dist_matrix <- dist(X_scaled)
hc <- hclust(dist_matrix, method = "ward.D2")
plot(hc)  # dendrogram
clusters <- cutree(hc, k = 4)

# DBSCAN
library(dbscan)
db_result <- dbscan(X_scaled, eps = 0.5, minPts = 5)

# PCA
pca_result <- prcomp(X_scaled, scale. = TRUE)
summary(pca_result)  # variance explained per component
```

---

## 4. WORKED EXAMPLE

**Scenario:** A marketing team wants "customer segments" for a campaign, with no predefined categories. Cluster customers on purchase behavior, then translate abstract clusters into business-usable segment names.

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

customers = pd.read_csv("customer_purchase_behavior.csv")
# columns: customer_id, purchase_frequency, avg_order_value,
#          days_since_last_purchase, total_categories_purchased

features = ["purchase_frequency", "avg_order_value",
            "days_since_last_purchase", "total_categories_purchased"]
X = customers[features]
X_scaled = StandardScaler().fit_transform(X)

# --- Step 1: Choose k using silhouette score ---
best_k, best_score = 2, -1
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_k, best_score = k, score
print(f"Best k: {best_k} (silhouette: {best_score:.3f})")

# --- Step 2: Fit final clustering ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
customers["cluster"] = kmeans.fit_predict(X_scaled)

# --- Step 3: Profile each cluster — THIS is the translation step ---
cluster_profile = customers.groupby("cluster")[features].mean()
print(cluster_profile)

# Example output interpretation:
#            purchase_frequency  avg_order_value  days_since_last  categories
# cluster 0         8.2              25.10             12              5.1
# cluster 1         1.1              180.50            95              1.2
# cluster 2         5.5              45.00             8               3.0
# cluster 3         0.5              15.00             210             1.0

# --- Step 4: Translate abstract clusters into named, actionable segments ---
segment_names = {
    0: "Frequent Small-Basket Shoppers — high engagement, low order value. "
       "Target with cross-sell campaigns to increase basket size.",
    1: "High-Value Infrequent Buyers — large orders, long gaps between purchases. "
       "Target with a replenishment reminder campaign timed to their typical cycle.",
    2: "Steady Mainstream Customers — moderate on all dimensions. "
       "Good candidates for loyalty program enrollment.",
    3: "At-Risk / Churning — low frequency, long time since last purchase. "
       "Target with a win-back campaign before they're fully lost.",
}
for cluster_id, description in segment_names.items():
    print(f"Cluster {cluster_id}: {description}")
```

The clustering math (Steps 1–2) is the easy part. Steps 3–4 — profiling each cluster's actual characteristics and translating them into a name and an action a marketing team can use — is where the real value gets created, and where most unsupervised projects either succeed or quietly fail.

---

## 5. COMMON MISTAKES

- **Skipping feature scaling before distance-based clustering.** k-means and DBSCAN both rely on distance calculations — an unscaled feature with a large numeric range (e.g., `avg_order_value` in dollars) will dominate the clustering purely due to scale, not because it's actually more important. Always scale before clustering.
- **Choosing k arbitrarily without an elbow or silhouette check.** Picking "4 segments because that's a nice round number for a marketing presentation" instead of letting the data suggest a natural k undermines the whole exercise — validate k quantitatively first, then reconcile with business needs.
- **Stopping at the cluster labels without profiling them.** Cluster 0, 1, 2, 3 mean nothing to a stakeholder — as shown in the worked example, profiling each cluster's actual feature averages and translating that into a named, actionable segment is not optional polish, it's the actual deliverable.
- **Over-interpreting t-SNE plot distances.** A common and specific error: reading the distance *between* well-separated clusters in a t-SNE visualization as meaningful (e.g., "cluster A is more similar to cluster B than to cluster C because it's closer in the plot") — t-SNE only reliably preserves local neighborhood structure, not global distances between separated groups.
- **Treating a high silhouette score as sufficient validation on its own.** A mathematically clean clustering can still be business-meaningless if the features used don't reflect what the business actually cares about — always pair the internal metric with a sanity check against domain knowledge before presenting results.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Algorithm Selection**
For each scenario, choose k-means, hierarchical clustering, or DBSCAN, and justify the choice:
a) Segmenting 500,000 customers into a predefined number of tiers for a loyalty program
b) Detecting anomalous network traffic patterns where anomalies form irregular, non-spherical shapes
c) Exploring product categories at multiple levels of granularity (broad categories down to specific sub-types) for a catalog with only 200 products

<details>
<summary>Sample Solution</summary>

a) **k-means** — large dataset (scales well), and a predefined number of tiers means k is already known or easily justified.
b) **DBSCAN** — explicitly designed for irregular cluster shapes and naturally flags outliers/anomalies as noise points, unlike k-means which assumes spherical clusters.
c) **Hierarchical clustering** — the dendrogram naturally supports exploring structure at multiple granularities (cut the tree at different heights), and 200 products is small enough that hierarchical clustering's poorer scalability isn't a concern.
</details>

---

**Exercise 2 — Cluster Translation**
You run k-means on employee data and get 3 clusters with these profiles: Cluster A (high tenure, high salary, low recent promotions), Cluster B (low tenure, moderate salary, high recent promotions), Cluster C (moderate tenure, low salary, zero recent promotions). Write a one-sentence business-usable name and a suggested HR action for each cluster.

<details>
<summary>Sample Solution</summary>

- **Cluster A — "Plateaued Veterans":** High tenure and pay but stalled advancement; risk of disengagement or attrition. Action: targeted career-development conversations or lateral growth opportunities.
- **Cluster B — "Rising Talent":** New but advancing quickly. Action: ensure retention investment (competitive comp review, mentorship) given their high growth trajectory and likely external market demand.
- **Cluster C — "Overlooked / At-Risk":** Moderate tenure, low pay, no promotion — a classic attrition-risk profile. Action: prioritize for compensation/promotion review before they leave for a competitor.
</details>

---

**Exercise 3 — Full Segmentation with Named Clusters (Applied)**
Using the `data_jobs` dataset, cluster job postings or companies on relevant numeric features (e.g., salary, required years of experience, number of listed skills). Choose k using silhouette score, profile each cluster, and produce named, business-usable segments with a one-sentence action for each — following the exact structure of the worked example. This is a strong, distinct portfolio piece separate from your earlier supervised-learning projects on the same dataset.
