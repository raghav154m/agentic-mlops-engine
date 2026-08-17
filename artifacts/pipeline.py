import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Data
df = pd.read_csv(r"data/student_data_messy.csv")

# 2. Set Target
target_col = 'Student_ID'
y = df[target_col]
X = df.drop(columns=[target_col])

# 3. Drop Identifiers
# Common identifier patterns
identifier_keywords = ['id', 'name', 'email', 'student_id', 'user_id', 'username']
cols_to_drop = []
for col in X.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in identifier_keywords):
        cols_to_drop.append(col)

# Remove duplicates in cols_to_drop just in case
cols_to_drop = list(set(cols_to_drop))
X = X.drop(columns=cols_to_drop, errors='ignore')

# 4. Raw String Normalization
# Helper dictionary for written numbers
number_map = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
    'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
    'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
    'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000'
}

def clean_strings(series):
    if series.dtype == 'object':
        # Strip whitespace
        series = series.str.strip()
        # Lowercase
        series = series.str.lower()
        # Clean '%'
        series = series.str.replace('%', '', regex=False)
        # Map written numbers
        # We need to be careful