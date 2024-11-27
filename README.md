
# PhiMonitor

**PhiMonitor** is a Python library for monitoring machine learning models to detect **data drift** and **overfitting**. It combines advanced similarity metrics, such as Jensen-Shannon divergence and Wasserstein distance, to quantify changes in datasets and predictions, enabling proactive retraining and maintaining model performance.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Quick Start](#quick-start)
  - [Example Usage](#example-usage)
- [Benchmarks](#benchmarks)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Data Drift Detection**: Quantifies changes in input distributions using combined similarity metrics.
- **Overfitting Prevention**: Tracks prediction similarity between training and validation datasets to detect overfitting.
- **Customizable Metrics**: Combines cosine similarity, Wasserstein distance, and Jensen-Shannon divergence for robust detection.
- **Scalable Batch Processing**: Handles datasets of any size efficiently with batch-based computation.
- **Visualization**: Optional tools to visualize drift and overfit scores over time.
- **Open Source**: Free and easy to integrate into Python-based ML pipelines.

---

## Installation

### Prerequisites

- Python 3.7 or higher.

### Installation Steps

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-repo/phi_monitor.git
    cd phi_monitor
    ```

2. **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Required Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install PhiMonitor:**
    ```bash
    pip install -e .
    ```

---

## Quick Start

### Example Usage

```python
from phi_monitor.core import PhiMonitor
import numpy as np

# Initialize PhiMonitor
monitor = PhiMonitor(verbose=True)

# Generate synthetic datasets
X_val = np.random.normal(size=(100, 20))
X_test = X_val + np.random.normal(scale=0.1, size=(100, 20))

# Detect Drift
drift_score = monitor.track_drift(X_test, X_val)
print(f"Drift Score: {drift_score:.4f}")

# Detect Overfitting
train_preds = np.linspace(0, 1, 100).reshape(-1, 1)
val_preds = train_preds + np.random.normal(scale=0.02, size=(100, 1))
overfit_score = monitor.check_overfitting(train_preds, val_preds)
print(f"Overfit Score: {overfit_score:.4f}")
```

---

## Benchmarks

PhiMonitor demonstrates strong scalability and low memory usage across various dataset sizes and batch configurations. See the `benchmarks` directory for detailed results.

---

## Limitations and Future Work

- **Real-Time Monitoring**: Integration with real-time streaming platforms (e.g., Kafka, Spark).
- **Explainability**: Adding SHAP-based explanations for drift scores.
- **Cloud Integration**: Optimized workflows for cloud environments.

---

## Contributing

We welcome contributions! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
