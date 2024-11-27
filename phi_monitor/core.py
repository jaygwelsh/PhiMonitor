# phi_monitor/core.py

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable, Dict, Any
from .utils import process_in_batches, aggregate_metrics, compute_jensenshannon
from .alerts import AlertManager


class PhiMonitor:
    def __init__(
        self,
        config: Dict[str, Any] = None,
        verbose: bool = True,
        overfit_callback: Callable = None
    ):
        """
        Initialize the PhiMonitor object.

        Parameters:
        - config: Configuration dictionary containing thresholds and other settings.
        - verbose: Whether to enable logging output.
        - overfit_callback: Callback function to execute on overfit detection.
        """
        default_config = {
            "drift_threshold": 0.05,
            "overfit_threshold": 0.05,
            "categorical_features": [],
            "imputation_strategy": 'mean',
            "similarity_metrics": ["wasserstein", "jensen_shannon", "cosine"]
        }
        if config is None:
            config = default_config
        else:
            default_config.update(config)
            config = default_config

        self.config = config
        self.verbose = verbose
        self.overfit_callback = overfit_callback
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.history = {"overfit_scores": [], "drift_scores": []}
        self.alert_manager = AlertManager(verbose=verbose)

        # Configure logger
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def check_overfitting(
        self, training_predictions: pd.DataFrame, validation_predictions: pd.DataFrame
    ) -> float:
        """
        Detect overfitting by comparing training and validation predictions.

        Parameters:
        - training_predictions: Predictions on training data (pandas DataFrame).
        - validation_predictions: Predictions on validation data (pandas DataFrame).

        Returns:
        - overfit_score: A score indicating the level of overfitting.
        """
        if training_predictions.empty or validation_predictions.empty:
            self.logger.error("Training or validation predictions are empty.")
            raise ValueError("Training and validation predictions must not be empty.")

        if training_predictions.shape[1] != validation_predictions.shape[1]:
            self.logger.error(
                "Training and validation predictions must have the same number of columns."
            )
            raise ValueError(
                "Training and validation predictions must have the same number of columns."
            )

        similarity = self.compute_combined_similarity(
            training_predictions, validation_predictions
        )
        overfit_score = max(0, 1 - similarity)
        overfit_score = min(overfit_score, 1.0)  # Ensure it's bounded by 1

        self.logger.debug(f"Overfit Score: {overfit_score:.4f}")
        self.history["overfit_scores"].append(overfit_score)

        if overfit_score > self.config["overfit_threshold"]:
            alert_message = (
                f"Overfitting detected! Overfit Score: {overfit_score:.4f} "
                f"exceeds threshold of {self.config['overfit_threshold']}"
            )
            self.logger.warning(alert_message)
            self.alert_manager.trigger_alert(
                score=overfit_score,
                threshold=self.config["overfit_threshold"],
                alert_type="overfit",
                alert_methods=["console", "log"]
            )
            if self.overfit_callback:
                self.overfit_callback(overfit_score, self.config["overfit_threshold"])

        return overfit_score

    def track_drift(
        self, data_test: pd.DataFrame, data_val: pd.DataFrame
    ) -> float:
        """
        Track data drift between test and validation datasets.

        Parameters:
        - data_test: Test dataset (pandas DataFrame).
        - data_val: Validation dataset (pandas DataFrame).

        Returns:
        - drift_score: A score indicating the level of data drift.
        """
        if data_test.empty or data_val.empty:
            self.logger.error("Test or validation datasets are empty.")
            raise ValueError("Test and validation datasets must not be empty.")

        if set(data_test.columns) != set(data_val.columns):
            self.logger.error("Test and validation datasets must have the same columns.")
            raise ValueError("Test and validation datasets must have the same columns.")

        similarity = self.compute_combined_similarity(data_test, data_val)
        drift_score = max(0, 1 - similarity)
        drift_score = min(drift_score, 1.0)  # Ensure it's bounded by 1

        self.logger.debug(f"Drift Score: {drift_score:.4f}")
        self.history["drift_scores"].append(drift_score)

        if drift_score > self.config["drift_threshold"]:
            alert_message = (
                f"Data drift detected! Drift Score: {drift_score:.4f} "
                f"exceeds threshold of {self.config['drift_threshold']}"
            )
            self.logger.warning(alert_message)
            self.alert_manager.trigger_alert(
                score=drift_score,
                threshold=self.config["drift_threshold"],
                alert_type="drift",
                alert_methods=["console", "log"]
            )

        return drift_score

    def compute_combined_similarity(
        self, data1: pd.DataFrame, data2: pd.DataFrame
    ) -> float:
        """
        Compute a combined similarity score between two datasets.

        Parameters:
        - data1: First dataset (pandas DataFrame).
        - data2: Second dataset (pandas DataFrame).

        Returns:
        - combined_similarity: A score indicating similarity between the datasets.
        """
        metrics = []

        for column in data1.columns:
            if column in data2.columns:
                col1 = data1[column].dropna().to_numpy()
                col2 = data2[column].dropna().to_numpy()

                if (
                    col1.dtype.kind in 'biufc'
                    and col2.dtype.kind in 'biufc'
                ):  # Numeric
                    if "wasserstein" in self.config["similarity_metrics"]:
                        wd = wasserstein_distance(col1, col2)
                        # Normalize Wasserstein distance to [0,1] using exponential decay
                        wd_sim = np.exp(-wd)
                        metrics.append(wd_sim)
                        self.logger.debug(
                            f"Wasserstein similarity for {column}: {wd_sim:.4f}"
                        )
                    if "jensen_shannon" in self.config["similarity_metrics"]:
                        js = compute_jensenshannon(col1, col2)
                        js_sim = 1 - js  # Jensen-Shannon divergence is between 0 and 1
                        metrics.append(js_sim)
                        self.logger.debug(
                            f"Jensen-Shannon similarity for {column}: {js_sim:.4f}"
                        )
                    if "cosine" in self.config["similarity_metrics"]:
                        if len(col1) == 0 or len(col2) == 0:
                            cosine_sim = 0.0
                        else:
                            # Ensure the vectors are the same length
                            min_length = min(len(col1), len(col2))
                            cosine_sim = cosine_similarity(
                                col1[:min_length].reshape(1, -1),
                                col2[:min_length].reshape(1, -1)
                            )[0][0]
                            cosine_sim = np.clip(cosine_sim, 0.0, 1.0)  # Ensure between 0 and 1
                        metrics.append(cosine_sim)
                        self.logger.debug(
                            f"Cosine similarity for {column}: {cosine_sim:.4f}"
                        )
                elif column in self.config["categorical_features"]:
                    sim = self.compute_categorical_similarity(col1, col2)
                    metrics.append(sim)
                    self.logger.debug(
                        f"Categorical similarity for {column}: {sim:.4f}"
                    )

        if metrics:
            combined_similarity = aggregate_metrics(metrics)
            combined_similarity = np.clip(combined_similarity, 0.0, 1.0)  # Ensure within [0,1]
        else:
            combined_similarity = 1.0  # If no metrics were computed, assume identical

        self.logger.debug(f"Combined Similarity: {combined_similarity:.4f}")
        return combined_similarity

    def compute_categorical_similarity(
        self, col1: np.ndarray, col2: np.ndarray
    ) -> float:
        """
        Compute similarity for categorical data using OneHotEncoder.

        Parameters:
        - col1: First column as numpy array.
        - col2: Second column as numpy array.

        Returns:
        - similarity: Categorical similarity score.
        """
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        combined = np.concatenate([col1, col2]).reshape(-1, 1)
        encoder.fit(combined)

        encoded1 = encoder.transform(col1.reshape(-1, 1))
        encoded2 = encoder.transform(col2.reshape(-1, 1))

        # Compute cosine similarity between the two one-hot encoded matrices
        if encoded1.shape[1] == 0 or encoded2.shape[1] == 0:
            return 0.0  # No similarity if encoding fails

        cosine_sim_matrix = cosine_similarity(encoded1, encoded2)
        cosine_sim = cosine_sim_matrix.mean()
        cosine_sim = np.clip(cosine_sim, 0.0, 1.0)  # Ensure within [0,1]
        return cosine_sim

    def track_drift_batch(
        self, data_test: pd.DataFrame, data_val: pd.DataFrame, batch_size: int = 1000
    ) -> float:
        """
        Compute drift scores in batches for large datasets.

        Parameters:
        - data_test: Test dataset (pandas DataFrame).
        - data_val: Validation dataset (pandas DataFrame).
        - batch_size: Number of rows to process in each batch.

        Returns:
        - aggregated_drift_score: A single drift score aggregated over all batches.
        """
        if data_test.empty or data_val.empty:
            self.logger.error("Test or validation datasets are empty.")
            raise ValueError("Test and validation datasets must not be empty.")

        if set(data_test.columns) != set(data_val.columns):
            self.logger.error("Test and validation datasets must have the same columns.")
            raise ValueError("Test and validation datasets must have the same columns.")

        drift_scores = []
        for test_batch, val_batch in zip(
            process_in_batches(data_test, batch_size),
            process_in_batches(data_val, batch_size)
        ):
            drift_score = self.track_drift(test_batch, val_batch)
            drift_scores.append(drift_score)

        aggregated_drift_score = np.mean(drift_scores)
        aggregated_drift_score = np.clip(aggregated_drift_score, 0.0, 1.0)  # Ensure within [0,1]
        self.logger.info(f"Aggregated Drift Score: {aggregated_drift_score:.4f}")
        return aggregated_drift_score

    def check_overfitting_batch(
        self,
        training_predictions: pd.DataFrame,
        validation_predictions: pd.DataFrame,
        batch_size: int = 1000
    ) -> float:
        """
        Compute overfit scores in batches for large datasets.

        Parameters:
        - training_predictions: Predictions on training data (pandas DataFrame).
        - validation_predictions: Predictions on validation data (pandas DataFrame).
        - batch_size: Number of rows to process in each batch.

        Returns:
        - aggregated_overfit_score: A single overfit score aggregated over all batches.
        """
        if training_predictions.empty or validation_predictions.empty:
            self.logger.error("Training or validation predictions are empty.")
            raise ValueError("Training and validation predictions must not be empty.")

        if training_predictions.shape[1] != validation_predictions.shape[1]:
            self.logger.error(
                "Training and validation predictions must have the same number of columns."
            )
            raise ValueError(
                "Training and validation predictions must have the same number of columns."
            )

        overfit_scores = []
        suppression_percentile = 95  # Percentile for outlier suppression

        for train_batch, val_batch in zip(
            process_in_batches(training_predictions, batch_size),
            process_in_batches(validation_predictions, batch_size)
        ):
            overfit_score = self.check_overfitting(train_batch, val_batch)
            overfit_scores.append(overfit_score)

        # Compute summary statistics
        max_score = max(overfit_scores)
        median_score = np.median(overfit_scores)
        iqr_score = np.percentile(overfit_scores, 75) - np.percentile(overfit_scores, 25)
        std_score = np.std(overfit_scores)
        num_exceeding_batches = sum(
            1 for score in overfit_scores
            if score > self.config["overfit_threshold"] + 0.01  # Margin
        )

        # Suppress rare outliers using percentile-based threshold
        adjusted_threshold = np.percentile(overfit_scores, suppression_percentile)
        if max_score > adjusted_threshold:
            self.logger.warning(
                f"Rare outliers suppressed. Max Overfit Score adjusted to {adjusted_threshold:.4f}."
            )
            max_score = adjusted_threshold

        aggregated_overfit_score = np.mean(overfit_scores)
        aggregated_overfit_score = np.clip(aggregated_overfit_score, 0.0, 1.0)  # Ensure within [0,1]

        # Log detailed summary for large datasets
        if num_exceeding_batches > 0:
            self.logger.warning(
                f"Overfitting detected in {num_exceeding_batches} out of {len(overfit_scores)} batches. "
                f"Max Overfit Score: {max_score:.4f}, Mean Score: {aggregated_overfit_score:.4f}, "
                f"Median Score: {median_score:.4f}, IQR: {iqr_score:.4f}, Std Dev: {std_score:.4f}."
            )

        self.logger.info(f"Aggregated Overfit Score: {aggregated_overfit_score:.4f}")
        return aggregated_overfit_score
