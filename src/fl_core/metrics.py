"""
Metrics Collection and Tracking
================================
Tracks and stores federated learning metrics.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field

import numpy as np

from loguru import logger


@dataclass
class RoundMetrics:
    """Metrics for a single federated learning round."""
    round_num: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Global metrics
    global_loss: float = 0.0
    global_accuracy: float = 0.0
    global_auc_roc: float = 0.0
    global_sensitivity: float = 0.0
    global_specificity: float = 0.0
    
    # Aggregated client metrics
    avg_train_loss: float = 0.0
    avg_train_accuracy: float = 0.0
    avg_val_loss: float = 0.0
    avg_val_accuracy: float = 0.0
    
    # Client participation
    num_clients: int = 0
    num_failures: int = 0
    total_samples: int = 0
    
    # Privacy metrics
    epsilon_spent: float = 0.0
    delta: float = 0.0
    
    # Per-client metrics
    client_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass 
class FederatedMetrics:
    """Complete federated learning metrics."""
    experiment_name: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    
    # Configuration
    num_hospitals: int = 0
    num_rounds: int = 0
    partition_type: str = "non_iid"
    dp_enabled: bool = False
    aggregation_strategy: str = "adaptive"
    
    # Final metrics
    final_accuracy: float = 0.0
    final_auc_roc: float = 0.0
    best_accuracy: float = 0.0
    best_round: int = 0
    
    # Round history
    rounds: List[RoundMetrics] = field(default_factory=list)


class MetricsCollector:
    """
    Collects and manages federated learning metrics.
    
    Provides real-time tracking and persistence.
    """
    
    def __init__(
        self,
        experiment_name: str = "federated_experiment",
        save_dir: str = "./metrics",
        auto_save: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save metrics
            auto_save: Whether to auto-save after each round
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        
        self.metrics = FederatedMetrics(experiment_name=experiment_name)
        self.current_round_metrics: Dict[str, Any] = {}
        
        logger.info(f"MetricsCollector initialized: {experiment_name}")
    
    def set_config(
        self,
        num_hospitals: int,
        num_rounds: int,
        partition_type: str = "non_iid",
        dp_enabled: bool = False,
        aggregation_strategy: str = "adaptive"
    ) -> None:
        """Set experiment configuration."""
        self.metrics.num_hospitals = num_hospitals
        self.metrics.num_rounds = num_rounds
        self.metrics.partition_type = partition_type
        self.metrics.dp_enabled = dp_enabled
        self.metrics.aggregation_strategy = aggregation_strategy
    
    def add_round_metrics(
        self,
        round_num: int,
        metrics: Dict[str, Any],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """
        Add metrics for a round.
        
        Args:
            round_num: Round number
            metrics: Dictionary of metrics
            client_metrics: Optional per-client metrics
        """
        round_metrics = RoundMetrics(
            round_num=round_num,
            global_loss=metrics.get("test_loss", metrics.get("loss", 0)),
            global_accuracy=metrics.get("test_accuracy", metrics.get("accuracy", 0)),
            global_auc_roc=metrics.get("test_auc_roc", metrics.get("auc_roc", 0)),
            global_sensitivity=metrics.get("test_sensitivity", metrics.get("sensitivity", 0)),
            global_specificity=metrics.get("test_specificity", metrics.get("specificity", 0)),
            avg_train_loss=metrics.get("avg_train_loss", 0),
            avg_train_accuracy=metrics.get("avg_train_accuracy", 0),
            avg_val_loss=metrics.get("avg_val_loss", 0),
            avg_val_accuracy=metrics.get("avg_val_accuracy", 0),
            num_clients=int(metrics.get("num_clients", 0)),
            num_failures=int(metrics.get("num_failures", 0)),
            total_samples=int(metrics.get("total_samples", 0)),
            epsilon_spent=metrics.get("epsilon_spent", 0),
            delta=metrics.get("delta", 0),
            client_metrics=client_metrics or {}
        )
        
        self.metrics.rounds.append(round_metrics)
        
        # Update best metrics
        if round_metrics.global_accuracy > self.metrics.best_accuracy:
            self.metrics.best_accuracy = round_metrics.global_accuracy
            self.metrics.best_round = round_num
        
        # Auto-save
        if self.auto_save:
            self.save()
        
        logger.debug(f"Round {round_num} metrics recorded")
    
    def add_client_round_metrics(
        self,
        round_num: int,
        client_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Add metrics for a specific client in a round."""
        if round_num not in self.current_round_metrics:
            self.current_round_metrics[round_num] = {}
        
        self.current_round_metrics[round_num][client_id] = metrics
    
    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.metrics.end_time = datetime.now().isoformat()
        
        if self.metrics.rounds:
            last_round = self.metrics.rounds[-1]
            self.metrics.final_accuracy = last_round.global_accuracy
            self.metrics.final_auc_roc = last_round.global_auc_roc
        
        self.save()
        logger.info("Metrics collection finalized")
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save metrics to JSON file.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{self.metrics.experiment_name}_metrics.json"
        
        filepath = self.save_dir / filename
        
        # Convert to dict
        metrics_dict = asdict(self.metrics)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        return str(filepath)
    
    def load(self, filepath: str) -> None:
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct metrics
        rounds = [RoundMetrics(**r) for r in data.pop('rounds', [])]
        self.metrics = FederatedMetrics(**data)
        self.metrics.rounds = rounds
        
        logger.info(f"Loaded metrics from {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        if not self.metrics.rounds:
            return {}
        
        losses = [r.global_loss for r in self.metrics.rounds]
        accuracies = [r.global_accuracy for r in self.metrics.rounds]
        
        return {
            "experiment_name": self.metrics.experiment_name,
            "num_rounds_completed": len(self.metrics.rounds),
            "best_accuracy": self.metrics.best_accuracy,
            "best_round": self.metrics.best_round,
            "final_accuracy": self.metrics.final_accuracy,
            "final_auc_roc": self.metrics.final_auc_roc,
            "avg_loss": np.mean(losses),
            "min_loss": np.min(losses),
            "avg_accuracy": np.mean(accuracies),
            "convergence_round": self._find_convergence_round(accuracies)
        }
    
    def _find_convergence_round(
        self, 
        accuracies: List[float], 
        threshold: float = 0.01,
        window: int = 5
    ) -> int:
        """Find round where accuracy converged."""
        if len(accuracies) < window:
            return len(accuracies)
        
        for i in range(len(accuracies) - window):
            window_std = np.std(accuracies[i:i+window])
            if window_std < threshold:
                return i
        
        return len(accuracies)
    
    def get_round_metrics(self, round_num: int) -> Optional[RoundMetrics]:
        """Get metrics for a specific round."""
        for r in self.metrics.rounds:
            if r.round_num == round_num:
                return r
        return None
    
    def get_learning_curves(self) -> Dict[str, List[float]]:
        """Get learning curves data."""
        return {
            "rounds": [r.round_num for r in self.metrics.rounds],
            "train_loss": [r.avg_train_loss for r in self.metrics.rounds],
            "train_accuracy": [r.avg_train_accuracy for r in self.metrics.rounds],
            "val_loss": [r.avg_val_loss for r in self.metrics.rounds],
            "val_accuracy": [r.avg_val_accuracy for r in self.metrics.rounds],
            "global_loss": [r.global_loss for r in self.metrics.rounds],
            "global_accuracy": [r.global_accuracy for r in self.metrics.rounds],
            "auc_roc": [r.global_auc_roc for r in self.metrics.rounds],
            "epsilon_spent": [r.epsilon_spent for r in self.metrics.rounds]
        }
    
    def get_privacy_consumption(self) -> Dict[str, Any]:
        """Get privacy budget consumption over time."""
        if not self.metrics.rounds:
            return {}
        
        return {
            "rounds": [r.round_num for r in self.metrics.rounds],
            "epsilon_spent": [r.epsilon_spent for r in self.metrics.rounds],
            "total_epsilon_spent": self.metrics.rounds[-1].epsilon_spent if self.metrics.rounds else 0
        }

