"""
train_model.py
Trains and compares two ML models to predict customer churn.
Run with: python train_model.py
Saves the best model to models/churn_model.pkl
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
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

# ---- 7b. Train Model C: XGBoost (boosting) ----
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()  # handles class imbalance
xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# ---- 8. Compare models ----
def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall   :", round(recall_score(y_true, y_pred), 3))
    print("F1 Score :", round(f1_score(y_true, y_pred), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_prob), 3))
    print(classification_report(y_true, y_pred))

evaluate("Logistic Regression", y_test, log_preds, log_reg.predict_proba(X_test_scaled)[:, 1])
evaluate("Random Forest", y_test, rf_preds, rf.predict_proba(X_test)[:, 1])
evaluate("XGBoost", y_test, xgb_preds, xgb.predict_proba(X_test)[:, 1])

# ---- 8b. 5-fold stratified cross-validation (more reliable than one split) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_models = {
    # scaler inside a pipeline so it is fit only on each training fold (no leakage)
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42),
}
cv_rows = []
for name, m in cv_models.items():
    r = cross_validate(m, X, y, cv=skf, scoring=["roc_auc", "recall", "precision", "f1"])
    cv_rows.append({"Model": name,
                    "ROC-AUC": f"{r['test_roc_auc'].mean():.3f} +/- {r['test_roc_auc'].std():.3f}",
                    "Recall": f"{r['test_recall'].mean():.3f} +/- {r['test_recall'].std():.3f}",
                    "Precision": f"{r['test_precision'].mean():.3f} +/- {r['test_precision'].std():.3f}",
                    "F1": f"{r['test_f1'].mean():.3f} +/- {r['test_f1'].std():.3f}"})
cv_df = pd.DataFrame(cv_rows)
print("\n5-fold cross-validation results:\n", cv_df.to_string(index=False))
cv_df.to_csv("models/cv_results.csv", index=False)

# ---- 8c. Business view: how many churners do we catch by contacting the top 20% riskiest customers? ----
probs = rf.predict_proba(X_test)[:, 1]
ranked = pd.DataFrame({"prob": probs, "churn": y_test.values}).sort_values("prob", ascending=False)
top = ranked.head(int(0.2 * len(ranked)))
capture = top["churn"].sum() / ranked["churn"].sum()
lift = top["churn"].mean() / ranked["churn"].mean()
print(f"\nTop 20% riskiest customers capture {capture:.1%} of all churners (lift = {lift:.2f}x vs random)")

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