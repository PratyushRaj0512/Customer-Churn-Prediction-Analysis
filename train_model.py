"""
train_model.py
Trains and compares two ML models to predict customer churn.
Run with: python train_model.py
Saves the best model to models/churn_model.pkl
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---- 1. Load data ----
df = pd.read_csv("data/telco_churn.csv")
print("Shape:", df.shape)

# ---- 2. Clean data ----
# TotalCharges has some blank strings that need converting to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop customer ID, not a predictive feature
df.drop("customerID", axis=1, inplace=True)

# Target variable: Yes/No -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---- 3. Encode categorical columns ----
categorical_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # save each encoder, needed later for the app

# ---- 4. Train / test split ----
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 5. Scale numeric features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 6. Train Model A: Logistic Regression ----
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)

# ---- 7. Train Model B: Random Forest ----
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)  # tree models don't need scaling
rf_preds = rf.predict(X_test)

# ---- 8. Compare models ----
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall   :", round(recall_score(y_true, y_pred), 3))
    print("F1 Score :", round(f1_score(y_true, y_pred), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_pred), 3))
    print(classification_report(y_true, y_pred))

evaluate("Logistic Regression", y_test, log_preds)
evaluate("Random Forest", y_test, rf_preds)

# ---- 9. Feature importance (Random Forest) ----
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features driving churn:\n", importances.head(10))

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.head(10).values, y=importances.head(10).index)
plt.title("Top 10 Features Driving Customer Churn")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.close()

# ---- 10. Confusion matrix for the better model ----
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

# ---- 11. Save the model, scaler, and encoders for the app ----
joblib.dump(rf, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(list(X.columns), "models/columns.pkl")

print("\nModel and preprocessing objects saved in the 'models' folder.")