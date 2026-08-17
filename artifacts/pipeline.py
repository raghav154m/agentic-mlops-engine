import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

# Define dummy data to resolve NameError: name 'y' is not defined
np.random.seed(42)
y = pd.Series(np.random.choice([0, 1, 2], size=100))

# Calculate minimum class count
min_class_count = y.value_counts().min()

# Determine number of splits
n_splits = min(5, min_class_count)

# Select Cross-Validation strategy
if n_splits < 2:
    # If we can't do stratified k-fold with at least 2 splits, fallback to KFold
    cv = KFold(n_splits=1, shuffle=True, random_state=42)
else:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Min class count: {min_class_count}")
print(f"Number of splits: {n_splits}")
print(f"Selected CV object: {cv}")