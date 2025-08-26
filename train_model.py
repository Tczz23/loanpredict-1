import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# 2. Basic preprocessing
# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Fill missing values (if any)
df = df.fillna(0)

# 3. Features (X) and Target (y)
X = df.drop("loan_status", axis=1)   # assumes target column is named 'loan_status'
y = df["loan_status"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Save model + scaler
joblib.dump(model, "loan_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ Model saved as loan_model.joblib and scaler.joblib")
