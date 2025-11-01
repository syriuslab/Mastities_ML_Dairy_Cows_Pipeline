"""Utility functions to load and preprocess the tabular (sensor/clinical) data.

This is a simplified version extracted from the original Colab notebook.
Keep comments in English (UK) as requested.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_tabular_data(csv_path: str, target_column: str = "class1", test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Tabular CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in CSV. Available: {list(df.columns)}")

    y = df[target_column].astype(int)
    X = df.drop(columns=[target_column])

    # First split: train + temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Second split: val + test
    val_ratio_relative = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_ratio_relative, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
