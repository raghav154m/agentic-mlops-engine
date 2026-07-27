import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


class DatasetProfiler:
    """Extracts lightweight dataset statistics for LLM context processing."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset standard file not found at {file_path}")
        self.df = pd.read_csv(self.file_path)

    def generate_profile(self) -> Dict[str, Any]:
        """Calculates key statistical dimensions of the dataframe.

        Returns:
            Dict[str, Any]: Compact metadata payload for LLM prompts.
        """
        num_rows, num_cols = self.df.shape
        missing_counts = self.df.isnull().sum().to_dict()

        column_metadata = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_vals = int(self.df[col].nunique())
            missing_ratio = float(self.df[col].isnull().mean())

            column_metadata[col] = {
                "dtype": dtype,
                "unique_values": unique_vals,
                "missing_ratio": round(missing_ratio, 2),
            }

            # Add basic numeric stats if applicable
            if pd.api.types.is_numeric_dtype(self.df[col]):
                column_metadata[col]["mean"] = round(float(self.df[col].mean()), 2)
                column_metadata[col]["std"] = round(float(self.df[col].std()), 2)

        profile = {
            "dataset_shape": {"rows": num_rows, "columns": num_cols},
            "columns": column_metadata,
        }
        return profile


if __name__ == "__main__":
    profiler = DatasetProfiler("data/sample_dataset.csv")
    report = profiler.generate_profile()
    print("Dataset Profile Output:")
    print(json.dumps(report, indent=2))