import pandas as pd
import numpy as np
import json
import re


class DatasetProfiler:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def generate_profile(self) -> dict:
        total_rows = len(self.df)
        profile = {
            "dataset_shape": {
                "rows": total_rows,
                "columns": len(self.df.columns)
            },
            "columns": {}
        }

        for col in self.df.columns:
            series = self.df[col]
            unique_count = int(series.nunique(dropna=True))
            null_count = int(series.isnull().sum())
            null_percentage = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0

            # Identifier detection heuristic
            col_lower = str(col).lower()
            is_id = bool(
                col_lower.endswith("_id") or 
                col_lower == "id" or 
                col_lower in ["name", "email", "student_id", "user_id"] or 
                (unique_count == total_rows and total_rows > 5)
            )

            col_info = {
                "inferred_type": str(series.dtype),
                "unique_values": unique_count,
                "null_count": null_count,
                "null_percentage": null_percentage,
                "is_identifier": is_id
            }

            if pd.api.types.is_numeric_dtype(series):
                col_info["mean"] = float(series.mean()) if not series.dropna().empty else None
                col_info["std"] = float(series.std()) if not series.dropna().empty else None
                col_info["min"] = float(series.min()) if not series.dropna().empty else None
                col_info["max"] = float(series.max()) if not series.dropna().empty else None
            else:
                top_vals = series.astype(str).value_counts().head(5).to_dict()
                col_info["sample_values"] = top_vals

            profile["columns"][col] = col_info

        return profile