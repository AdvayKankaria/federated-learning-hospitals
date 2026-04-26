"""
Federated Learning Server
=========================
Central aggregation server using Flower.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from collections import OrderedDict
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from loguru import logger

from ..models import create_model, PneumoniaClassifier
from .aggregation import create_aggregation_strategy
from .metrics import MetricsCollector


class FederatedServer:
    """
    Federated Learning Server.
    
    Coordinates training across multiple hospital clients.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        strategy: Optional[Strategy] = None,
        checkpoint_dir: str = "./checkpoints",
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the federated server.
        
        Args:
            model: Global model
            config: Server configuration
            strategy: Aggregation strategy
            checkpoint_dir: Directory for checkpoints
            metrics_collector: Metrics collector instance
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategy
        if strategy is None:
            strategy = create_aggregation_strategy(
                strategy_name=config.get("aggregation_strategy", "adaptive"),
                initial_parameters=self._get_initial_parameters(),
                config=config
            )
        self.strategy = strategy
        
        # Metrics
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.current_round = 0
        self.best_accuracy = 0.0
        
        logger.info("Federated server initialized")
        logger.info(f"  - Strategy: {type(strategy).__name__}")
        logger.info(f"  - Num rounds: {config.get('num_rounds', 50)}")
        logger.info(f"  - Min clients: {config.get('min_fit_clients', 5)}")
    
    def _get_initial_parameters(self) -> Parameters:
        """Get initial model parameters."""
        ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return ndarrays_to_parameters(ndarrays)
    
    def _set_model_parameters(self, parameters: Parameters) -> None:
        """Set model parameters from Flower Parameters."""
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def save_checkpoint(
        self, 
        round_num: int, 
        is_best: bool = False,
        metrics: Optional[Dict] = None
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            round_num: Current round number
            is_best: Whether this is the best model so far
            metrics: Optional metrics to save
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "metrics": metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_round = checkpoint.get("round", 0)
        logger.info(f"Loaded checkpoint from round {self.current_round}")
        return checkpoint
    
    def get_evaluate_fn(
        self,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Callable:
        """
        Create centralized evaluation function.
        
        Args:
            test_loader: Test data loader
            device: Device for evaluation
        
        Returns:
            Evaluation function
        """
        def evaluate(
            server_round: int,
            parameters: NDArrays,
            config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate global model on test set."""
            # Set parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            
            self.model.eval()
            self.model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            total_loss = 0.0
            correct = 0
            total = 0
            
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for images, labels, _ in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)
                    
                    total_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
            
            avg_loss = total_loss / total if total > 0 else 0
            accuracy = correct / total if total > 0 else 0
            
            # Calculate additional metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            
            # Sensitivity and Specificity
            pneumonia_mask = all_labels == 1
            normal_mask = all_labels == 0
            
            sensitivity = (
                np.sum((all_preds == 1) & pneumonia_mask) / np.sum(pneumonia_mask)
                if np.sum(pneumonia_mask) > 0 else 0
            )
            specificity = (
                np.sum((all_preds == 0) & normal_mask) / np.sum(normal_mask)
                if np.sum(normal_mask) > 0 else 0
            )
            
            # AUC-ROC
            try:
                from sklearn.metrics import roc_auc_score
                auc_roc = roc_auc_score(all_labels, all_probs)
            except:
                auc_roc = 0.0
            
            metrics = {
                "test_loss": float(avg_loss),
                "test_accuracy": float(accuracy),
                "test_sensitivity": float(sensitivity),
                "test_specificity": float(specificity),
                "test_auc_roc": float(auc_roc),
                "round": float(server_round)
            }
            
            # Update metrics collector
            self.metrics_collector.add_round_metrics(server_round, metrics)
            
            # Save checkpoint if best
            is_best = accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = accuracy
            
            if server_round % self.config.get("save_every_n_rounds", 5) == 0 or is_best:
                self.save_checkpoint(server_round, is_best, metrics)
            
            logger.info(
                f"[Server] Round {server_round} - Global Eval: "
                f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
                f"AUC={auc_roc:.4f}"
            )
            
            return avg_loss, metrics
        
        return evaluate
    
    def start(
        self,
        server_address: str = "[::]:8080",
        num_rounds: Optional[int] = None
    ) -> Dict:
        """
        Start the federated learning server.
        
        Args:
            server_address: Server address
            num_rounds: Number of federated rounds
        
        Returns:
            Training history
        """
        num_rounds = num_rounds or self.config.get("num_rounds", 50)
        
        logger.info(f"Starting federated server at {server_address}")
        logger.info(f"Running for {num_rounds} rounds")
        
        # Start Flower server
        history = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=self.strategy
        )
        
        return history


def create_server(
    config: Dict[str, Any],
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    device: Optional[torch.device] = None
) -> FederatedServer:
    """
    Factory function to create a federated server.
    
    Args:
        config: Server configuration
        test_loader: Optional test data loader for centralized evaluation
        device: Device for evaluation
    
    Returns:
        Initialized FederatedServer
    """
    # Create model
    model = create_model(
        model_name=config.get("model_name", "efficientnet_b0"),
        num_classes=config.get("num_classes", 2),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3)
    )
    
    # Create metrics collector
    metrics_collector = MetricsCollector(
        save_dir=config.get("metrics_dir", "./metrics")
    )
    
    # Create server
    server = FederatedServer(
        model=model,
        config=config,
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
        metrics_collector=metrics_collector
    )
    
    return server

