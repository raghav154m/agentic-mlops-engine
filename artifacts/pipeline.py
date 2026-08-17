import os
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Ingestion & Target
df = pd.read_csv(r"data/customer_churn_messy.csv")
target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Drop identifiers
drop_cols = [c for c in ["id", "user_id", "Student_ID", "Name", "Email", "Date"] if c in X.columns]
X = X.drop(columns=drop_cols, errors="ignore")

# 3. Clean string numbers (handle '%' and convert to numeric)
for col in X.columns:
    if X[col].dtype == object:
        # Check if column contains any digit characters
        if X[col].astype(str).str.contains(r"\d").any():
            # Remove '%' and ',' before conversion
            X[col] = pd.to_numeric(
                X[col].str.replace("%", "").str.replace(",", ""), errors="coerce"
            )

# 4. Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# 5. Preprocessing pipelines
numeric_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ]
)

# 6. Model pipeline
model = RandomForestClassifier(random_state=42)
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ]
)

# 7. Cross‑validation evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(
    model_pipeline, X, y, cv=skf, method="predict", n_jobs=-1
)

accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)

# 8. Fit on full data and serialize
model_pipeline.fit(X, y)

os.makedirs("artifacts", exist_ok=True)
dump(model_pipeline, "artifacts/model.joblib")
print("Model pipeline saved to artifacts/model.joblib")