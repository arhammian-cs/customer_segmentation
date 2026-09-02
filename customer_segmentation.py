"""
Mall Customer Segmentation with K-Means Clustering
---------------------------------------------------
Goal: Group mall customers into distinct segments based on annual
income and spending score, so a business could target each group
with a different marketing strategy.

Skills demonstrated: EDA with Pandas/Matplotlib, feature scaling,
unsupervised learning (K-Means), choosing K with the elbow method,
cluster visualization and interpretation.

Dataset: Mall Customer Segmentation dataset (200 customers: age,
annual income, spending score).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")
df.columns = ["CustomerID", "Genre", "Age", "AnnualIncome", "SpendingScore"]
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------------
# 2. Feature selection & scaling
# ---------------------------------------------------------------
X = df[["AnnualIncome", "SpendingScore"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------
# 3. Choose K with the elbow method
# ---------------------------------------------------------------
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(5.5, 4))
plt.plot(list(k_range), inertias, marker="o", color="#4C72B0")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150)
print("\nSaved chart: elbow_method.png")
print("Inertia by k:", dict(zip(k_range, [round(i, 1) for i in inertias])))

# The elbow is clearly at k=5 for this dataset
K_FINAL = 5

# ---------------------------------------------------------------
# 4. Fit final model
# ---------------------------------------------------------------
kmeans = KMeans(n_clusters=K_FINAL, n_init=10, random_state=RANDOM_STATE)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------------------
# 5. Visualize clusters
# ---------------------------------------------------------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(6.5, 5))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
for cluster_id in range(K_FINAL):
    cluster_points = df[df["Cluster"] == cluster_id]
    plt.scatter(cluster_points["AnnualIncome"], cluster_points["SpendingScore"],
                s=35, color=colors[cluster_id], label=f"Segment {cluster_id}")
plt.scatter(centers[:, 0], centers[:, 1], s=220, c="black", marker="X", label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"Customer Segments (K-Means, k={K_FINAL})")
plt.legend()
plt.tight_layout()
plt.savefig("customer_segments.png", dpi=150)
print("Saved chart: customer_segments.png")

# ---------------------------------------------------------------
# 6. Interpret segments
# ---------------------------------------------------------------
summary = df.groupby("Cluster")[["Age", "AnnualIncome", "SpendingScore"]].mean().round(1)
summary["Count"] = df.groupby("Cluster").size()
print("\nSegment profiles:\n", summary)
