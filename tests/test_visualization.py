# tests/test_visualization.py

import pytest
import pandas as pd
from phi_monitor.visualization import plot_metric_over_time, plot_feature_similarity

def test_plot_metric_over_time():
    history = {
        'drift_scores': [0.1, 0.2, 0.15, 0.3],
        'overfit_scores': [0.05, 0.07, 0.06, 0.08]
    }
    try:
        plot_metric_over_time(history, "drift_scores", "Drift Scores Over Time", "Drift Score")
        plot_metric_over_time(history, "overfit_scores", "Overfit Scores Over Time", "Overfit Score")
    except Exception as e:
        pytest.fail(f"plot_metric_over_time raised an exception {e}")

def test_plot_feature_similarity():
    feature_similarities = {
        'feature_0': 0.95,
        'feature_1': 0.85,
        'feature_2': 0.90,
        'feature_3': 0.80
    }
    try:
        plot_feature_similarity(feature_similarities, "Feature Similarity Scores")
    except Exception as e:
        pytest.fail(f"plot_feature_similarity raised an exception {e}")
