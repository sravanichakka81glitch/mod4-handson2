"""Module for loading news headlines data."""

import pandas as pd
import os


def load_data(filepath):
    """Load headlines dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} headlines from {filepath}")
    return df


def preview_data(df, n=5):
    """Preview the first n headlines."""
    print("\n--- Dataset Preview ---")
    print(df.head(n).to_string(index=False))
    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {list(df.columns)}")