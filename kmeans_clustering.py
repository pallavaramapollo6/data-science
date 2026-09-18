# Unsupervised Learning + Feature Engineering + Model Evaluation
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. Create sample customer data
# -----------------------------
data = {
    "Age": [22, 25, 27, 45, 48, 50, 52, 23, 26, 47],
    "Annual_Income": [25, 28, 30, 70, 75, 80, 85, 27, 32, 72],
    "Spending_Score": [80, 75, 85, 30, 25, 20, 15, 78, 82, 28]
}

df = pd.DataFrame(data)

print(df)

# -----------------------------
# 2. Feature Engineering
# -----------------------------
# Create a new feature
df["Income_per_Age"] = df["Annual_Income"] / df["Age"]

features = ["Age", "Annual_Income", "Spending_Score", "Income_per_Age"]

X = df[features]

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Unsupervised Learning
#    K-Means Clustering
# -----------------------------
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster assignments:")
print(df)

# -----------------------------
# 5. Model Evaluation
# -----------------------------
silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("\nSilhouette Score:", silhouette)

# -----------------------------
# 6. Visualization
# -----------------------------
plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    cmap="viridis"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
