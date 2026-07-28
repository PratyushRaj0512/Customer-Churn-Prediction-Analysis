# 📊 Customer Churn Prediction & Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end Data Analytics and Machine Learning pipeline built to identify high-risk telecom customers, uncover underlying churn drivers, and deliver actionable retention strategies through an interactive decision-support dashboard.

---

## 📌 Executive Summary

Customer acquisition costs (CAC) in the telecommunications industry are **5x to 7x higher** than customer retention costs. This project addresses subscription decay by combining **Exploratory Data Analysis (EDA)**, **Advanced Feature Engineering**, and **Cost-Sensitive Machine Learning Ensembles** to predict customer churn probability with high precision.

### Key Outcomes
- **89.4% ROC-AUC Score** achieved with tuned XGBoost and LightGBM models.
- **Identified Top Churn Predictors**: Contract type (Month-to-month), Fiber optic internet service, Tenure (<12 months), and Total Monthly Charges.
- **Estimated ROI Impact**: Preserving high-risk accounts via targeted incentive strategies yields a projected **18.5% reduction in annual revenue churn**.

---

## 🛠️ Tech Stack & Architecture

### **Core Stack**
* **Language**: Python 3.10+
* **Data Processing & Manipulation**: `pandas`, `numpy`
* **Exploratory Data Analysis & Visualization**: `seaborn`, `matplotlib`, `plotly`
* **Machine Learning & Modeling**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn` (SMOTE)
* **Hyperparameter Tuning**: `Optuna`, `GridSearchCV`
* **Model Explainability**: `SHAP` (SHapley Additive exPlanations)
* **Web Dashboard & Deployment**: `Streamlit`, `FastAPI`, `Docker`
* **Environment & Version Control**: `Conda`, `Git`

---

## 📁 Project Directory Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/                  # Original unprocessed dataset (e.g., Telco-Customer-Churn.csv)
│   └── processed/            # Cleaned, feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb      # EDA, distribution checks, missing value handling
│   ├── 02_feature_engineering.ipynb   # Customer lifetime, interaction ratios, encoding
│   └── 03_model_training_eval.ipynb   # Cross-validation, model comparison, SHAP analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and ingestion module
│   ├── preprocessor.py       # Pipelines for scaling, encoding, and imputation
│   ├── feature_engineering.py# Domain-specific feature creation
│   ├── train.py              # Model training scripts and hyperparameter tuning
│   └── evaluate.py           # Evaluation metrics, confusion matrix, ROC-AUC calculations
│
├── models/
│   ├── xgboost_churn_v1.pkl  # Serialized production model
│   └── preprocessor.pkl      # Saved scaling & encoding pipeline
│
├── app/
│   ├── dashboard.py          # Streamlit UI application
│   └── api.py                # FastAPI REST endpoint for real-time inference
│
├── tests/
│   ├── test_preprocessing.py # Unit tests for data pipeline
│   └── test_api.py           # Endpoint integration tests
│
├── Dockerfile                # Containerization setup
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md                 # Project documentation
```

---

## 📊 Dataset Architecture & Features

The baseline analysis utilizes the Telco Customer Churn dataset containing **7,043 customer records** and **21 features**.

| Feature Category | Description | Key Variables |
| :--- | :--- | :--- |
| **Demographics** | Customer baseline attributes | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Services** | Subscribed features & add-ons | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV` |
| **Account Info** | Financial & contract metadata | `Tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Target Variable** | Binary churn indicator | `Churn` (Yes / No) |

---

## 💡 Key Data Insights (EDA)

1. **Contract Type Impact**: Customers on **Month-to-month contracts** exhibit a churn rate of **42.7%**, compared to only **11%** for one-year and **3%** for two-year contract holders.
2. **Service Type Disparity**: Fiber Optic internet subscribers experience significantly higher churn rates (**41.9%**) than DSL subscribers (**19%**), primarily due to uncompetitive pricing and service reliability complaints.
3. **Tenure Velocity**: Customer churn peaks during the first **1–6 months** of onboarding. If a customer remains active past month 24, churn risk drops by over 65%.

---

## ⚙️ Machine Learning Pipeline

```
Raw Data ➔ Cleaning & Imputation ➔ Feature Engineering ➔ SMOTE Resampling ➔ Model Training ➔ SHAP Interpretability ➔ Streamlit Dashboard
```

### 1. Preprocessing & Feature Engineering
* **Missing Value Treatment**: Imputed missing `TotalCharges` using median values grouped by `Tenure` and `Contract`.
* **Feature Creation**:
  * `Tenure_Group`: Segmented into quarterly brackets (0–6 mos, 6–12 mos, 12–24 mos, etc.).
  * `Service_Density`: Count of active add-on subscriptions (Tech Support, Online Security, Backup, etc.).
  * `Cost_per_Service`: Ratio of `MonthlyCharges` to total active services.
* **Encoding & Scaling**: Target encoding for high-cardinality categoricals; RobustScaler applied to continuous variables (`MonthlyCharges`, `TotalCharges`).

### 2. Imbalance Handling
Given class imbalance (~26.5% positive churn rate), **SMOTE (Synthetic Minority Over-sampling Technique)** combined with Tomek Links was applied to the training set to prevent model bias toward majority retention classes.

---

## 📈 Model Performance & Evaluation

Models were evaluated using **Stratified 5-Fold Cross-Validation** optimized for **Recall** and **ROC-AUC** (minimizing False Negatives to capture maximum churners).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 79.2% | 0.63 | 0.71 | 0.67 | 0.841 |
| Random Forest Classifier | 81.5% | 0.68 | 0.62 | 0.65 | 0.856 |
| CatBoost Classifier | 82.1% | 0.69 | 0.74 | 0.71 | 0.882 |
| **XGBoost Classifier (Tuned)** | **83.6%** | **0.72** | **0.78** | **0.75** | **0.894** |
| LightGBM Classifier | 83.1% | 0.71 | 0.77 | 0.74 | 0.889 |

> **Selected Production Model**: **XGBoost Classifier** tuned via Optuna.

---

## 🔎 Model Interpretability (SHAP)

To explain predictions to business stakeholders, SHAP values were generated:
* **Positive Churn Drivers**: Month-to-month contracts, High Monthly Charges, Electronic Check payment method.
* **Retention Anchor Features**: Long tenure, TechSupport add-on, OnlineSecurity subscription, 2-Year Contract commitments.

---

## 🚀 Interactive Streamlit Dashboard

The project includes an interactive web dashboard for business managers to:
* Input individual customer parameters to receive real-time churn probabilities.
* View aggregate KPI metrics and customer segmentation clusters.
* Simulate custom retention offers (e.g., dynamic discount calculator) to evaluate churn probability reduction.

---

## 💻 Getting Started

### Prerequisites
* Python 3.10 or higher
* Git
* Docker (optional)

### Local Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run Model Training Pipeline**
   ```bash
   python src/train.py
   ```

5. **Launch the Dashboard**
   ```bash
   streamlit run app/dashboard.py
   ```
   *Navigate to `http://localhost:8501` in your browser.*

---

## 🐳 Docker Container Deployment

To run the application using Docker:

```bash
# Build the Docker image
docker build -t churn-predictor:v1 .

# Run the container
docker run -p 8501:8501 churn-predictor:v1
```

---

## 🗺️ Future Roadmap

- [ ] Incorporate time-series transaction records for temporal retention modeling (Survival Analysis).
- [ ] Integrate MLflow for experiment tracking and automated model registry.
- [ ] Develop automated CI/CD retraining pipelines via GitHub Actions.
- [ ] Implement A/B testing engine for retention campaign monitoring.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a Pull Request for improvements.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
