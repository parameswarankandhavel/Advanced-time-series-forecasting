import numpy as np

def rolling_window_split(X, y, window_size=500, horizon=24):
    """
    Perform rolling-window cross-validation for time series data.
    """
    splits = []
    for i in range(window_size, len(X) - horizon):
        X_train = X[:i]
        y_train = y[:i]
        X_val = X[i:i + horizon]
        y_val = y[i:i + horizon]
        splits.append((X_train, y_train, X_val, y_val))
    return splits
