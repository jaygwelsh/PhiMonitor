# phi_monitor/utils.py

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List
from scipy.stats import entropy  # Added import for entropy


def process_in_batches(
    data: pd.DataFrame, batch_size: int = 1000
) -> Generator[pd.DataFrame, None, None]:
    """
    Generator to yield batches of a DataFrame.

    Parameters:
    - data: The pandas DataFrame to split into batches.
    - batch_size: Number of rows per batch.

    Yields:
    - A batch of the DataFrame.
    """
    num_rows = data.shape[0]
    for start in range(0, num_rows, batch_size):
        yield data.iloc[start:start + batch_size]


def aggregate_metrics(metrics: List[float]) -> float:
    """
    Aggregate a list of similarity metrics into a single score.

    Parameters:
    - metrics: List of similarity scores.

    Returns:
    - aggregated_score: The mean of the similarity scores.
    """
    if not metrics:
        return 1.0  # Assume identical if no metrics
    return np.mean(metrics)


def compute_jensenshannon(col1: np.ndarray, col2: np.ndarray) -> float:
    """
    Compute the Jensen-Shannon divergence between two numerical distributions.

    Parameters:
    - col1: First numerical column as a numpy array.
    - col2: Second numerical column as a numpy array.

    Returns:
    - js_divergence: The Jensen-Shannon divergence.
    """
    # Compute histograms
    hist1, _ = np.histogram(col1, bins=30, density=True)
    hist2, _ = np.histogram(col2, bins=30, density=True)
    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    # Compute Jensen-Shannon divergence
    m = 0.5 * (hist1 + hist2)
    js_divergence = 0.5 * (entropy(hist1, m) + entropy(hist2, m))
    return js_divergence
