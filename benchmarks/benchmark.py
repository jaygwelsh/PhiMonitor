# benchmarks/benchmark.py

import time
import numpy as np
import pandas as pd
import logging
from phi_monitor.core import PhiMonitor
from phi_monitor.utils import process_in_batches
from typing import Tuple
from memory_profiler import memory_usage
import matplotlib.pyplot as plt


def drift_alert(drift_score: float, threshold: float):
    print(f"[ALERT CALLBACK] Drift Score {drift_score:.4f} exceeded threshold {threshold}")


def overfit_alert(overfit_score: float, threshold: float):
    print(f"[ALERT CALLBACK] Overfit Score {overfit_score:.4f} exceeded threshold {threshold}")


def generate_large_data(
    samples: int, features: int, noise: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate large synthetic datasets with optional noise.

    Parameters:
    - samples: Number of samples.
    - features: Number of features.
    - noise: Standard deviation of Gaussian noise added to the test set.

    Returns:
    - X_val: Validation dataset.
    - X_test: Test dataset with added noise.
    """
    X_val = pd.DataFrame(
        np.random.normal(size=(samples, features)),
        columns=[f'feature_{i}' for i in range(features)]
    )
    X_test = X_val + pd.DataFrame(
        np.random.normal(scale=noise, size=(samples, features)),
        columns=[f'feature_{i}' for i in range(features)]
    )
    return X_val, X_test


def run_benchmark(
    size: int,
    features: int,
    batch_size: int,
    noise: float,
    runs: int = 3
) -> Tuple[float, float, float]:
    """
    Run the benchmark multiple times and return average execution time and memory usage.

    Parameters:
    - size: Number of samples.
    - features: Number of features.
    - batch_size: Number of rows per batch.
    - noise: Noise level added to test data.
    - runs: Number of repeated runs.

    Returns:
    - avg_time: Average execution time in seconds.
    - max_memory: Maximum memory usage in MB.
    - avg_overfit_score: Average overfit score across runs.
    """
    times = []
    memories = []
    overfit_scores = []

    for i in range(runs):
        # Generate data
        X_val, X_test = generate_large_data(samples=size, features=features, noise=noise)

        # Initialize PhiMonitor
        monitor = PhiMonitor(
            config={
                "drift_threshold": 0.05,
                "overfit_threshold": 0.05,
                "categorical_features": ['feature_0', 'feature_1'],
                "imputation_strategy": 'mean',
                "similarity_metrics": ["wasserstein", "jensen_shannon", "cosine"]
            },
            verbose=False
        )

        # Measure execution time and memory usage
        start_time = time.time()
        mem_usage = memory_usage(
            (monitor.track_drift_batch, (X_test, X_val, batch_size)),
            max_iterations=1,
            interval=0.1,
            timeout=None
        )
        end_time = time.time()

        # Drift Score
        drift_score = monitor.track_drift_batch(X_test, X_val, batch_size=batch_size)

        # Overfit Score (using simple predictions for benchmark)
        train_preds = pd.DataFrame(
            np.linspace(0, 1, size).reshape(-1, 1),
            columns=['pred']
        )
        val_preds = train_preds + pd.DataFrame(
            np.random.normal(scale=0.02, size=(size, 1)),
            columns=['pred']
        )
        overfit_score = monitor.check_overfitting_batch(
            train_preds, val_preds, batch_size=batch_size
        )

        # Record metrics
        elapsed_time = end_time - start_time
        peak_memory = max(mem_usage) - min(mem_usage)  # Memory usage during the function
        times.append(elapsed_time)
        memories.append(peak_memory)
        overfit_scores.append(overfit_score)

    avg_time = np.mean(times)
    max_memory = np.max(memories)
    avg_overfit_score = np.mean(overfit_scores)

    return avg_time, max_memory, avg_overfit_score


def stress_test_scalability(logger):
    """
    Stress-test PhiMonitor for scalability by varying dataset sizes and batch sizes.

    Parameters:
    - logger: Logger for logging progress and results.
    """
    logger.info("\n=== Scalability Stress Test ===")
    dataset_sizes = [100, 1000, 10000, 100000, 500000, 1000000]  # Increased dataset sizes
    features = 20
    noise = 0.1
    batch_sizes = [1000, 10000]  # Different batch sizes to test
    runs = 3  # Number of repeated runs for averaging

    results = []

    for size in dataset_sizes:
        for batch_size in batch_sizes:
            if batch_size > size:
                continue  # Skip invalid batch sizes
            logger.info(f"\nDataset Size: {size} samples, Batch Size: {batch_size}")
            try:
                avg_time, max_memory, avg_overfit = run_benchmark(
                    size=size,
                    features=features,
                    batch_size=batch_size,
                    noise=noise,
                    runs=runs
                )
                logger.info(f"Average Execution Time over {runs} runs: {avg_time:.4f} seconds")
                logger.info(f"Peak Memory Usage: {max_memory:.2f} MB")
                logger.info(f"Average Overfit Score: {avg_overfit:.4f}")
                results.append({
                    "Dataset Size": size,
                    "Batch Size": batch_size,
                    "Avg Execution Time (s)": avg_time,
                    "Peak Memory (MB)": max_memory,
                    "Avg Overfit Score": avg_overfit
                })
            except MemoryError as e:
                logger.error(f"MemoryError: {e} for dataset size {size} and batch size {batch_size}")
            except Exception as e:
                logger.error(f"Error: {e} for dataset size {size} and batch size {batch_size}")

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv("scalability_benchmark_results.csv", index=False)
    logger.info("\nScalability Benchmarking Completed. Results saved to 'scalability_benchmark_results.csv'.")

    # Plotting Execution Time
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        subset = results_df[results_df["Batch Size"] == batch_size]
        plt.plot(subset["Dataset Size"], subset["Avg Execution Time (s)"], marker='o', label=f'Batch Size {batch_size}')
    plt.xlabel("Dataset Size (samples)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.title("PhiMonitor Scalability: Execution Time vs Dataset Size")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig("execution_time_scalability.png")
    plt.close()
    logger.info("Execution Time vs Dataset Size plot saved as 'execution_time_scalability.png'.")

    # Plotting Memory Usage
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        subset = results_df[results_df["Batch Size"] == batch_size]
        plt.plot(subset["Dataset Size"], subset["Peak Memory (MB)"], marker='o', label=f'Batch Size {batch_size}')
    plt.xlabel("Dataset Size (samples)")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.title("PhiMonitor Scalability: Memory Usage vs Dataset Size")
    plt.legend()
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig("memory_usage_scalability.png")
    plt.close()
    logger.info("Memory Usage vs Dataset Size plot saved as 'memory_usage_scalability.png'.")

    # Save plots and results for further analysis
    logger.info("Scalability benchmarking completed successfully.")


def edge_case_tests(logger):
    """
    Test PhiMonitor on edge cases.

    Parameters:
    - logger: Logger for logging progress and results.
    """
    logger.info("\n=== Edge Case Tests ===")

    # Identical datasets with non-zero variance and handling missing values
    logger.info("\nCase: Identical Datasets")
    X = pd.DataFrame(
        np.tile(
            np.linspace(1, 2, 100).reshape(100, 1),
            (1, 20)
        ),
        columns=[f'feature_{i}' for i in range(20)]
    )
    # Introduce missing values
    X.iloc[0, 0] = np.nan
    X.iloc[10, 5] = np.nan
    X.iloc[50, 9] = np.nan
    monitor = PhiMonitor(
        config={
            "drift_threshold": 0.05,
            "overfit_threshold": 0.05,
            "categorical_features": ['feature_0', 'feature_1', 'feature_5', 'feature_9'],
            "imputation_strategy": 'mean',
            "similarity_metrics": ["wasserstein", "jensen_shannon", "cosine"]
        },
        verbose=False
    )
    drift_score = monitor.track_drift(X, X)
    logger.info(f"Drift Score (Identical): {drift_score:.4f}")

    # Completely dissimilar datasets
    logger.info("\nCase: Completely Dissimilar Datasets")
    X_val = pd.DataFrame(
        np.random.normal(size=(100, 20)),
        columns=[f'feature_{i}' for i in range(20)]
    )
    X_test = pd.DataFrame(
        np.random.normal(loc=10, scale=5, size=(100, 20)),
        columns=[f'feature_{i}' for i in range(20)]
    )
    drift_score = monitor.track_drift(X_test, X_val)
    logger.info(f"Drift Score (Dissimilar): {drift_score:.4f}")

    # Perfectly matching predictions
    logger.info("\nCase: Perfectly Matching Predictions")
    train_preds = pd.DataFrame(
        np.linspace(0, 1, 100).reshape(-1, 1),
        columns=['pred']
    )
    val_preds = train_preds.copy()
    overfit_score = monitor.check_overfitting(train_preds, val_preds)
    logger.info(f"Overfit Score (Matching): {overfit_score:.4f}")

    # Random, uncorrelated predictions
    logger.info("\nCase: Random, Uncorrelated Predictions")
    train_preds = pd.DataFrame(
        np.random.uniform(size=(100, 1)),
        columns=['pred']
    )
    val_preds = pd.DataFrame(
        np.random.uniform(size=(100, 1)),
        columns=['pred']
    )
    overfit_score = monitor.check_overfitting(train_preds, val_preds)
    logger.info(f"Overfit Score (Uncorrelated): {overfit_score:.4f}")


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger()

    logger.info("=== Benchmarking PhiMonitor ===")
    stress_test_scalability(logger)
    edge_case_tests(logger)


if __name__ == "__main__":
    main()
