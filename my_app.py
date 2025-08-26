import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

# =========================
# Utility functions
# =========================
def save_model(model, name):
    joblib.dump(model, f"{name}.joblib")

def load_model(name):
    file_path = f"{name}.joblib"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

# =========================
# Load dataset
# =========================
CSV_PATH = "loan_approval_dataset.csv"
df = pd.read_csv(CSV_PATH)

# Normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Encode target
df["loan_status"] = df["loan_status"].astype(str).str.strip().map({"Approved": 1, "Rejected": 0})

# Replace 0s in critical columns
critical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score']
for col in critical_cols:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

# Fill missing values
for col in df.columns:
    if df[col].dtype.kind in "biufc":
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Navigation")
menu = ["Dataset Overview", "Model Training", "Visualization", "Loan Prediction"]
choice = st.sidebar.radio("Go to", menu)

# =========================
# Dataset Overview
# =========================
if choice == "Dataset Overview":
    st.title("Loan Approval Dataset Overview")

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Class Distribution")
    st.bar_chart(df['loan_status'].value_counts())

# =========================
# Model Training & Evaluation
# =========================
elif choice == "Model Training":
    st.title("Train & Evaluate Models")

    # One-hot encoding
    X = pd.get_dummies(df.drop('loan_status', axis=1), drop_first=True)
    y = df['loan_status']
    feature_columns = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_columns, "feature_columns.joblib")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=150),
        "Support Vector Machine": SVC(kernel='rbf', probability=True)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        save_model(model, name)

        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1})

    results_df = pd.DataFrame(results)
    st.subheader("Model Comparison")
    st.dataframe(results_df)

# =========================
# Visualization
# =========================
elif choice == "Visualization":
    st.title("Visualization & Insights")

    # Correlation Heatmap
    st.subheader("Correlation Matrix")
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=["loan_status"], errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Model Performance
    st.subheader("Model Performance Comparison")
    if os.path.exists("feature_columns.joblib") and os.path.exists("scaler.joblib"):
        feature_columns = joblib.load("feature_columns.joblib")
        scaler = joblib.load("scaler.joblib")

        X = pd.get_dummies(df.drop('loan_status', axis=1), drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(X)
        y = df['loan_status']

        results = []
        for name in ["Logistic Regression", "Random Forest", "Support Vector Machine"]:
            model = load_model(name)
            if model:
                y_pred = model.predict(X_scaled)
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y, y_pred),
                    "Precision": precision_score(y, y_pred, zero_division=0),
                    "Recall": recall_score(y, y_pred, zero_division=0),
                    "F1 Score": f1_score(y, y_pred, zero_division=0)
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Horizontal bar plots per metric
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            plot_df = results_df.sort_values(by=metric, ascending=True)
            ax.barh(plot_df["Model"], plot_df[metric])
            ax.set_title(metric)
            ax.set_xlim(0, 1)
            for y_pos, v in enumerate(plot_df[metric].values):
                ax.text(v + 0.01, y_pos, f"{v:.2f}", va="center")

        st.pyplot(fig2)

# =========================
# Loan Prediction
# =========================
elif choice == "Loan Prediction":
    st.title("Loan Approval Prediction")

    feature_columns = joblib.load("feature_columns.joblib")
    scaler = joblib.load("scaler.joblib")
    model = load_model("Random Forest")

    # Numeric inputs
    st.subheader("Enter Loan Applicant Information")
    raw = {}
    input_columns_num = [
        'no_of_dependents','income_annum','loan_amount','loan_term','cibil_score',
        'residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'
    ]
    for col in input_columns_num:
        if col in df.columns:
            raw[col] = st.number_input(col, min_value=0.0, step=1.0)

    # Categorical inputs
    if 'education' in df.columns:
        raw['education'] = st.selectbox("Education", df['education'].unique())
    if 'self_employed' in df.columns:
        raw['self_employed'] = st.selectbox("Self Employed", df['self_employed'].unique())

    if st.button("Predict"):
        new_df = pd.DataFrame([raw])
        new_df_dum = pd.get_dummies(new_df, drop_first=True).reindex(columns=feature_columns, fill_value=0)
        new_df_scaled = scaler.transform(new_df_dum)

        prediction = model.predict(new_df_scaled)[0]
        probability = model.predict_proba(new_df_scaled)[0][1]

        if prediction == 1:
            st.success(f"✅ Loan Approved with probability {probability:.2f}")
        else:
            st.error(f"❌ Loan Rejected with probability {1-probability:.2f}")
