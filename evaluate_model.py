import torch
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    """
    MAPE in percentage.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0


# Wrapper that returns several metrics at once
def evaluate_model(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """
    Compute common regression metrics.

    Returns a dict with the metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R2: Coefficient of Determination
    - MAPE: Mean Absolute Percentage Error
    - MedianAE: Median Absolute Error

    Accepts NumPy arrays *or* PyTorch tensors and ignores rows that contain NaNs.
    """
    # Convert tensors to NumPy and move to CPU if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Drop rows with any NaNs
    valid = ~np.isnan(y_true).any(axis=1) & ~np.isnan(y_pred).any(axis=1)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    # Compute metrics
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAE":      mean_absolute_error(y_true, y_pred),
        "RMSE":     rmse,
        "R2":       r2_score(y_true, y_pred),
        "MAPE":     mean_absolute_percentage_error(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
    }
