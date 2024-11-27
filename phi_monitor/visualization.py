# phi_monitor/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Dict, Any
import streamlit as st

def plot_metric_over_time(history: Dict[str, Any], metric_key: str, title: str, ylabel: str):
    """
    Plot a specific metric over time.

    Parameters:
    - history: Dictionary containing historical metrics.
    - metric_key: Key of the metric to plot.
    - title: Plot title.
    - ylabel: Y-axis label.

    Returns:
    - None
    """
    if metric_key not in history or not history[metric_key]:
        logging.error(f"Metric '{metric_key}' not found or empty in history.")
        raise ValueError(f"Metric '{metric_key}' not found or empty in history.")
    
    data = history[metric_key]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data)+1), data, marker='o', label=metric_key)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_similarity(feature_similarities: Dict[str, float], title: str = "Feature Similarity"):
    """
    Plot similarity scores for specific features.

    Parameters:
    - feature_similarities: Dictionary of feature names and their similarity scores.
    - title: Plot title.

    Returns:
    - None
    """
    if not feature_similarities:
        logging.error("No feature similarities provided.")
        raise ValueError("Feature similarities cannot be empty.")

    features = list(feature_similarities.keys())
    similarities = list(feature_similarities.values())
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=features, y=similarities)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def display_dashboard(history: Dict[str, Any]):
    """
    Create an interactive dashboard using Streamlit.

    Parameters:
    - history: Dictionary containing historical metrics.

    Returns:
    - None
    """
    st.title("PhiMonitor Dashboard")

    # Drift Scores Over Time
    if 'drift_scores' in history and history['drift_scores']:
        st.header("Drift Scores Over Time")
        df_drift = pd.DataFrame({
            'Time': range(1, len(history['drift_scores']) + 1),
            'Drift Score': history['drift_scores']
        })
        st.line_chart(df_drift.set_index('Time'))
    
    # Overfit Scores Over Time
    if 'overfit_scores' in history and history['overfit_scores']:
        st.header("Overfit Scores Over Time")
        df_overfit = pd.DataFrame({
            'Time': range(1, len(history['overfit_scores']) + 1),
            'Overfit Score': history['overfit_scores']
        })
        st.line_chart(df_overfit.set_index('Time'))

    # Model Performance Metrics
    if 'phi_scores' in history and history['phi_scores']:
        st.header("Model Performance Metrics")
        df_perf = pd.DataFrame(history['phi_scores'])
        st.write(df_perf)

    # Optionally, add more visualizations as needed
