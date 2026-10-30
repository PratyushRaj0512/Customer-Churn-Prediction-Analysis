"""
segmentation.py
KMeans customer segmentation on tenure and charges, then churn rate per segment.
Run with: python segmentation.py   (outputs: models/segment_summary.csv, models/segments.png)
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["churn"] = (df["Churn"] == "Yes").astype(int)

features = ["tenure", "MonthlyCharges", "TotalCharges"]
Z = StandardScaler().fit_transform(df[features])  # KMeans is distance-based, so scaling is essential

# Choose k: compare silhouette scores
print("Silhouette score by k:")
for k in range(2, 7):
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
    print(f"  k={k}: {silhouette_score(Z, labels):.3f}")

K = 3  # chosen for interpretability (k=2 has a slightly higher silhouette)
df["segment"] = KMeans(K, n_init=10, random_state=42).fit_predict(Z)

summary = df.groupby("segment").agg(
    customers=("churn", "size"),
    avg_tenure_months=("tenure", "mean"),
    avg_monthly_charges=("MonthlyCharges", "mean"),
    churn_rate=("churn", "mean"),
).round(2)
print("\nSegment profile:\n", summary)
summary.to_csv("models/segment_summary.csv")

plt.figure(figsize=(7, 5))
sc = plt.scatter(df["tenure"], df["MonthlyCharges"], c=df["segment"], cmap="viridis", s=8, alpha=0.6)
plt.xlabel("Tenure (months)"); plt.ylabel("Monthly charges")
plt.title("Customer segments (KMeans, k=3)")
plt.colorbar(sc, label="Segment")
plt.tight_layout(); plt.savefig("models/segments.png", dpi=150)
print("\nSaved models/segment_summary.csv and models/segments.png")
