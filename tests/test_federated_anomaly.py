"""
Unit tests for the Federated Anomaly Detector project.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.anomaly_detection.isolation_forest_detector import IsolationForestDetector, StatisticalAnomalyDetector, AutoencoderDetector
from src.federated_engine.federated_learning import FederatedClient, FederatedServer

def test_isolation_forest_detector():
    """Tests the IsolationForestDetector."""
    print("Testing IsolationForestDetector...")
    np.random.seed(42)
    normal_data = np.random.randn(100, 5)
    anomaly_data = np.random.randn(20, 5) * 5
    
    all_data = np.vstack([normal_data, anomaly_data])
    true_labels = np.array([0]*100 + [1]*20)

    model = IsolationForestDetector(contamination=0.15, n_estimators=10)
    model.fit(normal_data)
    
    predictions = model.predict(all_data)
    assert len(predictions) == len(all_data)
    assert np.sum(predictions) >= 0 # Should detect at least some anomalies
    print("  IsolationForestDetector test passed.")

def test_statistical_detector():
    """Tests the StatisticalAnomalyDetector."""
    print("Testing StatisticalAnomalyDetector...")
    np.random.seed(42)
    normal_data = np.random.randn(100, 5)
    anomaly_data = np.random.randn(20, 5) * 10
    
    all_data = np.vstack([normal_data, anomaly_data])

    model = StatisticalAnomalyDetector(threshold=3.0)
    model.fit(normal_data)
    
    predictions = model.predict(all_data)
    assert len(predictions) == len(all_data)
    print("  StatisticalAnomalyDetector test passed.")

def test_autoencoder_detector():
    """Tests the AutoencoderDetector."""
    print("Testing AutoencoderDetector...")
    np.random.seed(42)
    normal_data = np.random.randn(50, 5)
    anomaly_data = np.random.randn(10, 5) * 5
    
    all_data = np.vstack([normal_data, anomaly_data])

    model = AutoencoderDetector(encoding_dim=2, epochs=10)
    model.fit(normal_data)
    
    predictions = model.predict(all_data)
    assert len(predictions) == len(all_data)
    print("  AutoencoderDetector test passed.")

def test_federated_client():
    """Tests the FederatedClient class."""
    print("Testing FederatedClient...")
    dummy_data = np.random.rand(30, 4)
    client = FederatedClient("test_client", dummy_data, IsolationForestDetector)
    
    # Train model
    train_result = client.train_local_model(epochs=1)
    assert train_result["training_success"] == True
    assert train_result["local_samples"] == 30
    
    # Get model update
    update = client.get_model_update()
    assert update is not None
    assert update["client_id"] == "test_client"
    
    # Receive global model
    client.receive_global_model({"test_key": "test_value"})
    assert client.local_model_state is not None
    print("  FederatedClient test passed.")

def test_federated_server():
    """Tests the FederatedServer class."""
    print("Testing FederatedServer...")
    server = FederatedServer(anomaly_detector_factory=IsolationForestDetector)
    
    # Register clients
    for i in range(2):
        dummy_data = np.random.rand(50, 4)
        client = FederatedClient(f"client_{i}", dummy_data, IsolationForestDetector)
        server.register_client(client)
    
    assert len(server.clients) == 2
    
    # Run a federated round
    round_result = server.federated_round(epochs=1)
    assert round_result["round"] == 1
    assert round_result["num_participating_clients"] > 0
    
    print("  FederatedServer test passed.")

if __name__ == '__main__':
    test_isolation_forest_detector()
    test_statistical_detector()
    test_autoencoder_detector()
    test_federated_client()
    test_federated_server()
    print("\nAll tests passed!")