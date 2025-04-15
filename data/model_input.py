import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_model_input(
    df: pd.DataFrame,
    target_col: str,
    n_steps_ahead: int = 1,
    test_size: float = 0.2
):
    df = df.copy().dropna()

    # Shift target column forward
    df['target'] = df[target_col].shift(-n_steps_ahead)

    # Drop rows with NaNs after shifting
    df = df.dropna()

    # Drop non-numeric (like regime if categorical)
    numeric_df = df.select_dtypes(include=[np.number])

    X = numeric_df.drop(columns=['target'])
    y = numeric_df['target']

    # Train/test split
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, y_train, X_test, y_test
