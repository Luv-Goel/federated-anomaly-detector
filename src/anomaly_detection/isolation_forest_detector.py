"""
Anomaly Detection Algorithms Module

This module implements various anomaly detection algorithms:
- Isolation Forest
- Autoencoder-based detection
- Statistical methods (Z-score, IQR)
"""

import numpy as np
from typing import Dict, Any, Optional
import copy

class BaseAnomalyDetector:
    """Base class for all anomaly detectors."""
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model_state: Optional[Dict[str, Any]] = None

    def fit(self, data: np.ndarray):
        """Train the anomaly detector on the provided data."""
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomaly scores or labels for the provided data."""
        raise NotImplementedError("Subclasses must implement predict()")

    def get_model_state(self) -> Dict[str, Any]:
        """Returns the current state of the model."""
        raise NotImplementedError("Subclasses must implement get_model_state()")

    def set_model_state(self, state: Dict[str, Any]):
        """Sets the model state from a dictionary."""
        raise NotImplementedError("Subclasses must implement set_model_state()")


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest anomaly detector.
    Works by isolating observations using random partitioning.
    Anomalies are easier to isolate (shorter path lengths) than normal points.
    """
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, contamination: float = 0.1, random_state: int = 42):
        super().__init__(contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees = []
        self.n_features = None
        self.threshold = None

    def _build_tree(self, data: np.ndarray, depth: int = 0, max_depth: int = 10) -> Dict[str, Any]:
        """Recursively builds an isolation tree."""
        if depth >= max_depth or len(data) <= 1:
            return {"leaf": True, "size": len(data), "node_index": np.random.randint(0, 1000)}
        
        # Random feature and split point
        feature_idx = np.random.randint(0, data.shape[1])
        min_val, max_val = data[:, feature_idx].min(), data[:, feature_idx].max()
        
        if min_val == max_val:
            return {"leaf": True, "size": len(data), "node_index": np.random.randint(0, 1000)}
        
        split_value = np.random.uniform(min_val, max_val)
        
        left_mask = data[:, feature_idx] < split_value
        right_mask = ~left_mask
        
        return {
            "leaf": False,
            "feature_idx": feature_idx,
            "split_value": split_value,
            "depth": depth,
            "left": self._build_tree(data[left_mask], depth + 1, max_depth),
            "right": self._build_tree(data[right_mask], depth + 1, max_depth)
        }

    def _path_length(self, point: np.ndarray, tree: Dict[str, Any], depth: int = 0) -> float:
        """Calculates the path length for a point in an isolation tree."""
        if tree["leaf"]:
            # For a leaf with 'c' points, the average path length is:
            # c > 2: 2 * (ln(c - 1) + 0.5772156649) - 2 * (c - 1) / n
            c = max(tree["size"], 2)
            return depth + 2 * (np.log(c - 1) + 0.5772156649) - 2 * (c - 1) / c
        
        if point[tree["feature_idx"]] < tree["split_value"]:
            return self._path_length(point, tree["left"], depth + 1)
        else:
            return self._path_length(point, tree["right"], depth + 1)

    def fit(self, data: np.ndarray):
        """Builds an ensemble of isolation trees."""
        np.random.seed(self.random_state)
        self.n_features = data.shape[1]
        self.trees = []
        
        sample_size = min(self.max_samples, len(data))
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[indices]
        
        for _ in range(self.n_estimators):
            self.trees.append(self._build_tree(sample_data))
        
        # Calculate anomaly scores for the training data to set threshold
        train_scores = self.score_samples(data)
        self.threshold = np.percentile(train_scores, (1 - self.contamination) * 100)
        
        self.model_state = self.get_model_state()

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Returns anomaly scores (path lengths) for samples. Higher = more anomalous."""
        scores = []
        for point in data:
            # Average path length across all trees
            avg_path = np.mean([self._path_length(point, tree) for tree in self.trees])
            scores.append(avg_path)
        return np.array(scores)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predicts anomalies. Returns binary array where 1 indicates an anomaly.
        """
        if self.threshold is None:
            self.fit(data)
        
        scores = self.score_samples(data)
        # For isolation forest: shorter path = more anomalous
        # We invert the scores so higher score = more anomalous
        max_path = np.log2(self.max_samples) * 2
        anomaly_scores = max_path - scores
        
        return (anomaly_scores > self.threshold).astype(int)

    def get_model_state(self) -> Dict[str, Any]:
        """Returns the model's state for federated learning."""
        return {
            "type": "IsolationForestDetector",
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "trees": self.trees
        }

    def set_model_state(self, state: Dict[str, Any]):
        """Sets the model state from a dictionary."""
        self.n_estimators = state["n_estimators"]
        self.max_samples = state["max_samples"]
        self.contamination = state["contamination"]
        self.threshold = state["threshold"]
        self.n_features = state["n_features"]
        self.trees = state["trees"]


class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """
    Statistical anomaly detector using Z-score method.
    Points with Z-score > threshold are considered anomalies.
    """
    def __init__(self, threshold: float = 3.0):
        super().__init__()
        self.threshold = threshold
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Computes mean and standard deviation."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-10  # Avoid division by zero
        self.model_state = self.get_model_state()

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predicts anomalies using Z-score."""
        if self.mean is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        z_scores = np.abs((data - self.mean) / self.std)
        # A point is an anomaly if any of its features has Z-score > threshold
        return (np.max(z_scores, axis=1) > self.threshold).astype(int)

    def get_model_state(self) -> Dict[str, Any]:
        return {"type": "StatisticalAnomalyDetector", "threshold": self.threshold, "mean": self.mean, "std": self.std}

    def set_model_state(self, state: Dict[str, Any]):
        self.threshold = state["threshold"]
        self.mean = state["mean"]
        self.std = state["std"]


class AutoencoderDetector(BaseAnomalyDetector):
    """
    Simple Autoencoder-based anomaly detector.
    Trains on normal data and identifies anomalies based on high reconstruction error.
    """
    def __init__(self, encoding_dim: int = 4, epochs: int = 50, contamination: float = 0.1):
        super().__init__(contamination)
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.weights = None
        self.threshold = None
        self.input_dim = None

    def fit(self, data: np.ndarray):
        """Simple linear autoencoder training."""
        self.input_dim = data.shape[1]
        
        # Simple Xavier initialization for a single hidden layer
        np.random.seed(42)
        self.weights = {
            'encoder': np.random.randn(self.input_dim, self.encoding_dim) * 0.1,
            'decoder': np.random.randn(self.encoding_dim, self.input_dim) * 0.1
        }
        
        # Simple gradient descent
        lr = 0.01
        for epoch in range(self.epochs):
            # Forward pass
            encoded = data @ self.weights['encoder']
            decoded = encoded @ self.weights['decoder']
            
            # Compute reconstruction error
            error = data - decoded
            loss = np.mean(error ** 2)
            
            # Backward pass (simplified)
            grad_decoder = encoded.T @ error / len(data)
            grad_encoder = data.T @ (error @ self.weights['decoder'].T) / len(data)
            
            self.weights['encoder'] += lr * grad_encoder
            self.weights['decoder'] += lr * grad_decoder
        
        # Set threshold based on reconstruction error distribution
        _, reconstructed = self._predict_internal(data)
        reconstruction_errors = np.mean((data - reconstructed) ** 2, axis=1)
        self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
        self.model_state = self.get_model_state()

    def _predict_internal(self, data: np.ndarray):
        encoded = data @ self.weights['encoder']
        decoded = encoded @ self.weights['decoder']
        return encoded, decoded

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predicts anomalies based on reconstruction error."""
        _, reconstructed = self._predict_internal(data)
        reconstruction_errors = np.mean((data - reconstructed) ** 2, axis=1)
        return (reconstruction_errors > self.threshold).astype(int)

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "type": "AutoencoderDetector",
            "encoding_dim": self.encoding_dim,
            "threshold": self.threshold,
            "input_dim": self.input_dim,
            "weights": self.weights
        }

    def set_model_state(self, state: Dict[str, Any]):
        self.encoding_dim = state["encoding_dim"]
        self.threshold = state["threshold"]
        self.input_dim = state["input_dim"]
        self.weights = state["weights"]

if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    normal_data = np.random.randn(500, 5)  # Normal data
    anomaly_data = np.random.randn(50, 5) * 5  # Anomalies (with higher variance)
    
    all_data = np.vstack([normal_data, anomaly_data])
    true_labels = np.array([0]*500 + [1]*50)

    # Test Isolation Forest
    print("Testing IsolationForestDetector...")
    if_model = IsolationForestDetector(contamination=0.1, n_estimators=20)
    if_model.fit(normal_data)
    if_predictions = if_model.predict(all_data)
    print(f"  Predicted {np.sum(if_predictions)} anomalies")
    
    # Test Statistical
    print("Testing StatisticalAnomalyDetector...")
    stat_model = StatisticalAnomalyDetector(threshold=2.5)
    stat_model.fit(normal_data)
    stat_predictions = stat_model.predict(all_data)
    print(f"  Predicted {np.sum(stat_predictions)} anomalies")
    
    # Test Autoencoder
    print("Testing AutoencoderDetector...")
    ae_model = AutoencoderDetector(encoding_dim=2, epochs=100)
    ae_model.fit(normal_data)
    ae_predictions = ae_model.predict(all_data)
    print(f"  Predicted {np.sum(ae_predictions)} anomalies")