# examples/example.py

from phi_monitor.core import PhiMonitor
from phi_monitor.visualization import plot_metric_over_time
from phi_monitor.alerts import AlertManager
import numpy as np
import pandas as pd

# Configure logging (optional)
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    # Initialize AlertManager
    alert_manager = AlertManager(verbose=True)

    # Generate synthetic datasets with realistic overlap
    # Including a case with missing values and zero variance features
    X_val = pd.DataFrame(np.random.normal(size=(100, 20)),
                         columns=[f'feature_{i}' for i in range(20)])
    X_test = X_val + pd.DataFrame(np.random.normal(scale=0.05, size=(100, 20)),
                                  columns=[f'feature_{i}' for i in range(20)])

    # Introduce zero variance features
    X_val['feature_0'] = 5.0  # First feature has zero variance
    X_val['feature_1'] = 5.0  # Second feature has zero variance
    X_test['feature_0'] = 5.0
    X_test['feature_1'] = 5.0

    # Introduce missing values
    X_val.loc[0, 'feature_2'] = np.nan
    X_test.loc[1, 'feature_3'] = np.nan

    # Generate synthetic predictions with slight variations
    train_preds = pd.DataFrame(np.linspace(0, 1, 100).reshape(-1, 1), columns=['pred'])
    val_preds = train_preds + pd.DataFrame(np.random.normal(scale=0.02, size=(100, 1)), columns=['pred'])

    # Configuration for PhiMonitor
    config = {
        "drift_threshold": 0.05,
        "overfit_threshold": 0.05,
        "categorical_features": ['feature_0', 'feature_1', 'feature_2', 'feature_3'],
        "imputation_strategy": 'mean'
    }

    # Initialize PhiMonitor
    monitor = PhiMonitor(
        config=config,
        verbose=True
    )

    # Track drift
    drift_score = monitor.track_drift(X_test, X_val)
    print(f"Drift Score: {drift_score:.4f}")

    # Check overfitting
    overfit_score = monitor.check_overfitting(train_preds, val_preds)
    print(f"Overfit Score: {overfit_score:.4f}")

    # Example of accessing history for visualization or further analysis
    print("\n--- Historical Scores ---")
    print(f"Drift Scores: {monitor.history['drift_scores']}")
    print(f"Overfit Scores: {monitor.history['overfit_scores']}")

    # Visualization
    plot_metric_over_time(monitor.history, "drift_scores", "Drift Scores Over Time", "Drift Score")
    plot_metric_over_time(monitor.history, "overfit_scores", "Overfit Scores Over Time", "Overfit Score")

if __name__ == "__main__":
    main()
