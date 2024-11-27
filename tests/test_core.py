# tests/test_core.py

import pytest
import pandas as pd
import numpy as np
from phi_monitor.core import PhiMonitor

def test_overfitting_detection():
    monitor = PhiMonitor(verbose=False)
    train_preds = pd.DataFrame({'pred': np.linspace(0, 1, 100)})
    val_preds = train_preds + pd.DataFrame({'pred': np.random.normal(scale=0.01, size=100)})
    
    overfit_score = monitor.check_overfitting(train_preds, val_preds)
    assert 0 <= overfit_score <= 1

def test_drift_detection_identical():
    monitor = PhiMonitor(verbose=False)
    data = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f'feature_{i}' for i in range(5)])
    drift_score = monitor.track_drift(data, data)
    assert drift_score == 0.0

def test_drift_detection_completely_dissimilar():
    monitor = PhiMonitor(verbose=False)
    data_val = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f'feature_{i}' for i in range(5)])
    data_test = pd.DataFrame(np.random.normal(loc=10, size=(100, 5)), columns=[f'feature_{i}' for i in range(5)])
    drift_score = monitor.track_drift(data_test, data_val)
    assert drift_score > 0.9

def test_empty_datasets():
    monitor = PhiMonitor(verbose=False)
    data_val = pd.DataFrame()
    data_test = pd.DataFrame()
    with pytest.raises(ValueError):
        monitor.track_drift(data_test, data_val)

def test_mismatched_columns():
    monitor = PhiMonitor(verbose=False)
    data_val = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f'feature_{i}' for i in range(5)])
    data_test = pd.DataFrame(np.random.normal(size=(100, 4)), columns=[f'feature_{i}' for i in range(4)])
    with pytest.raises(ValueError):
        monitor.track_drift(data_test, data_val)