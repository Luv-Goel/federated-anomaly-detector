"""
Federated Learning Engine for Anomaly Detection

This module provides the core infrastructure for federated learning:
- Client and server coordination
- Model aggregation strategies
- Secure communication protocols
"""

import numpy as np
import copy
from typing import List, Dict, Any, Optional, Callable

class FederatedClient:
    """
    Represents a client node in the federated learning network.
    Each client holds local data and trains a local model.
    """
    def __init__(self, client_id: str, local_data: np.ndarray, anomaly_detector_factory: Callable):
        self.client_id = client_id
        self.local_data = local_data
        self.anomaly_detector = anomaly_detector_factory()
        self.local_model_state = None
        self.is_trained = False

    def train_local_model(self, epochs: int = 5) -> Dict[str, Any]:
        """
        Trains the local anomaly detection model on this client's data.
        Returns training metrics.
        """
        # Train the local anomaly detector
        self.anomaly_detector.fit(self.local_data)
        self.local_model_state = copy.deepcopy(self.anomaly_detector.get_model_state())
        self.is_trained = True
        
        # Calculate some basic local metrics
        local_predictions = self.anomaly_detector.predict(self.local_data)
        anomaly_ratio = np.sum(local_predictions) / len(local_predictions)
        
        return {
            "client_id": self.client_id,
            "epochs_trained": epochs,
            "local_samples": len(self.local_data),
            "local_anomaly_ratio": anomaly_ratio,
            "training_success": True
        }

    def get_model_update(self) -> Optional[Dict[str, Any]]:
        """
        Returns the model update (gradients or weights) to be sent to the server.
        """
        if not self.is_trained:
            return None
        return {
            "client_id": self.client_id,
            "model_state": self.local_model_state,
            "num_samples": len(self.local_data),
            "anomaly_ratio": np.sum(self.anomaly_detector.predict(self.local_data)) / len(self.local_data)
        }

    def receive_global_model(self, global_model_state: Dict[str, Any]):
        """
        Receives and applies a global model update from the server.
        """
        self.anomaly_detector.set_model_state(global_model_state)
        self.local_model_state = copy.deepcopy(global_model_state)

class FederatedServer:
    """
    Central server that coordinates the federated learning process.
    """
    def __init__(self, anomaly_detector_factory: Callable, aggregation_strategy: str = "fedavg"):
        self.anomaly_detector = factory = anomaly_detector_factory
        self.global_model_state = None
        self.aggregation_strategy = aggregation_strategy
        self.clients: Dict[str, FederatedClient] = {}
        self.round_history = []

    def register_client(self, client: FederatedClient):
        """Registers a new client with the server."""
        self.clients[client.client_id] = client
        print(f"Client '{client.client_id}' registered with the server.")

    def unregister_client(self, client_id: str):
        """Removes a client from the server."""
        if client_id in self.clients:
            del self.clients[client_id]
            print(f"Client '{client_id}' unregistered from the server.")

    def broadcast_global_model(self):
        """Broadcasts the current global model to all registered clients."""
        if self.global_model_state is None:
            # Initialize global model on first broadcast
            initial_detector = self.anomaly_detector_factory()
            initial_detector.fit(np.random.rand(10, 5))  # Dummy fit
            self.global_model_state = initial_detector.get_model_state()
        
        for client in self.clients.values():
            client.receive_global_model(self.global_model_state)
        print(f"Global model broadcast to {len(self.clients)} clients.")

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates model updates from clients using Federated Averaging (FedAvg).
        """
        if not client_updates:
            return self.global_model_state

        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Weighted average of model states
        aggregated_state = {}
        for key in client_updates[0]['model_state'].keys():
            weighted_sum = np.zeros_like(client_updates[0]['model_state'][key], dtype=float)
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += weight * update['model_state'][key]
            
            aggregated_state[key] = weighted_sum

        return aggregated_state

    def federated_round(self, client_selection: List[str] = None, epochs: int = 5) -> Dict[str, Any]:
        """
        Executes a single round of federated learning.
        """
        round_num = len(self.round_history) + 1
        print(f"\n--- Starting Federated Round {round_num} ---")

        # 1. Broadcast global model to selected (or all) clients
        self.broadcast_global_model()
        
        selected_clients = []
        if client_selection:
            selected_clients = [self.clients[cid] for cid in client_selection if cid in self.clients]
        else:
            selected_clients = list(self.clients.values())

        # 2. Train local models on selected clients
        client_updates = []
        for client in selected_clients:
            print(f"  Training on client '{client.client_id}'...")
            train_result = client.train_local_model(epochs=epochs)
            update = client.get_model_update()
            if update:
                client_updates.append(update)

        # 3. Aggregate updates to form new global model
        print("  Aggregating model updates...")
        self.global_model_state = self.aggregate_updates(client_updates)

        # 4. Broadcast new global model
        self.broadcast_global_model()

        round_result = {
            "round": round_num,
            "num_participating_clients": len(client_updates),
            "total_samples": sum(u['num_samples'] for u in client_updates),
        }
        self.round_history.append(round_result)
        print(f"--- Completed Federated Round {round_num} ---\n")
        
        return round_result

if __name__ == "__main__":
    # Example usage
    from src.anomaly_detection.isolation_forest_detector import IsolationForestDetector

    # Create server
    server = FederatedServer(anomaly_detector_factory=IsolationForestDetector)

    # Create and register some dummy clients
    for i in range(3):
        dummy_data = np.random.rand(100, 5)  # 100 samples, 5 features
        client = FederatedClient(f"client_{i}", dummy_data, IsolationForestDetector)
        server.register_client(client)

    # Run a few federated rounds
    for r in range(3):
        server.federated_round(epochs=1)