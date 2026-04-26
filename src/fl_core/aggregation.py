"""
Aggregation Strategies for Federated Learning
==============================================
Implements FedAvg, FedProx, and Custom Adaptive Aggregation.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from collections import OrderedDict
import numpy as np

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
    EvaluateIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from loguru import logger


class AdaptiveAggregation(Strategy):
    """
    Adaptive Aggregation Strategy.
    
    Weights hospital updates based on:
    - Sample count
    - Update stability (gradient variance)
    - Loss improvement
    - Data quality metrics
    """
    
    def __init__(
        self,
        initial_parameters: Parameters,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        # Adaptive weight factors
        sample_weight: float = 0.3,
        stability_weight: float = 0.25,
        loss_weight: float = 0.25,
        quality_weight: float = 0.2,
        # Stability tracking
        stability_window: int = 5,
        # Callbacks
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True
    ):
        """
        Initialize Adaptive Aggregation strategy.
        
        Args:
            initial_parameters: Initial model parameters
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
            sample_weight: Weight factor for sample count
            stability_weight: Weight factor for update stability
            loss_weight: Weight factor for loss improvement
            quality_weight: Weight factor for data quality
            stability_window: Number of rounds to track for stability
            on_fit_config_fn: Function to configure fit
            on_evaluate_config_fn: Function to configure evaluate
            accept_failures: Whether to accept client failures
        """
        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        # Adaptive weights
        self.sample_weight = sample_weight
        self.stability_weight = stability_weight
        self.loss_weight = loss_weight
        self.quality_weight = quality_weight
        
        # History tracking
        self.stability_window = stability_window
        self.client_history: Dict[str, List[Dict]] = {}
        self.current_round = 0
        
        # Callbacks
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        
        logger.info("Adaptive Aggregation initialized")
        logger.info(f"  Weights: sample={sample_weight}, stability={stability_weight}, "
                   f"loss={loss_weight}, quality={quality_weight}")
    
    def initialize_parameters(
        self, 
        client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global parameters."""
        return self.initial_parameters
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure training for selected clients."""
        self.current_round = server_round
        
        # Sample clients
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        
        # Create fit config
        config = {"current_round": server_round}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)
        
        fit_ins = FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation for selected clients."""
        if self.fraction_evaluate == 0.0:
            return []
        
        sample_size = max(
            int(client_manager.num_available() * self.fraction_evaluate),
            self.min_evaluate_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients
        )
        
        config = {}
        if self.on_evaluate_config_fn:
            config = self.on_evaluate_config_fn(server_round)
        
        evaluate_ins = EvaluateIns(parameters, config)
        
        return [(client, evaluate_ins) for client in clients]
    
    def _calculate_client_weights(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights for each client.
        
        Returns:
            Dictionary mapping client_id to weight
        """
        weights = {}
        
        for client, fit_res in results:
            client_id = client.cid
            metrics = fit_res.metrics or {}
            num_samples = fit_res.num_examples
            
            # Initialize history for new clients
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            
            # Store current metrics
            self.client_history[client_id].append({
                "round": self.current_round,
                "loss": metrics.get("train_loss", 0),
                "accuracy": metrics.get("train_accuracy", 0),
                "num_samples": num_samples
            })
            
            # Keep only recent history
            if len(self.client_history[client_id]) > self.stability_window:
                self.client_history[client_id] = self.client_history[client_id][-self.stability_window:]
            
            # Calculate component weights
            
            # 1. Sample weight - more samples = higher weight
            sample_score = num_samples
            
            # 2. Stability weight - lower variance = higher weight
            history = self.client_history[client_id]
            if len(history) >= 2:
                losses = [h["loss"] for h in history]
                loss_variance = np.var(losses) if len(losses) > 1 else 0
                stability_score = 1.0 / (1.0 + loss_variance)
            else:
                stability_score = 1.0
            
            # 3. Loss improvement weight
            if len(history) >= 2:
                loss_improvement = history[-2]["loss"] - history[-1]["loss"]
                loss_score = max(0, loss_improvement) + 0.1  # Add small constant
            else:
                loss_score = 1.0
            
            # 4. Quality weight (based on accuracy trend)
            if len(history) >= 2:
                acc_improvement = history[-1]["accuracy"] - history[-2]["accuracy"]
                quality_score = max(0.1, 0.5 + acc_improvement)
            else:
                quality_score = 1.0
            
            # Combine weights
            combined_weight = (
                self.sample_weight * sample_score +
                self.stability_weight * stability_score * 1000 +  # Scale to match sample weight
                self.loss_weight * loss_score * 1000 +
                self.quality_weight * quality_score * 1000
            )
            
            weights[client_id] = combined_weight
            
            logger.debug(
                f"Client {client_id} weights: sample={sample_score:.2f}, "
                f"stability={stability_score:.4f}, loss={loss_score:.4f}, "
                f"quality={quality_score:.4f}, combined={combined_weight:.2f}"
            )
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using adaptive weights.
        
        Args:
            server_round: Current round number
            results: List of client results
            failures: List of failures
        
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}
        
        # Calculate adaptive weights
        client_weights = self._calculate_client_weights(results)
        
        # Aggregate parameters with adaptive weights
        weights_results = []
        for client, fit_res in results:
            weight = client_weights.get(client.cid, 1.0 / len(results))
            weights_results.append((
                parameters_to_ndarrays(fit_res.parameters),
                weight
            ))
        
        # Weighted average
        aggregated_ndarrays = self._weighted_average(weights_results)
        
        # Aggregate metrics
        metrics_aggregated = {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        
        for client, fit_res in results:
            weight = fit_res.num_examples / total_examples
            for key, value in (fit_res.metrics or {}).items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = 0.0
                metrics_aggregated[key] += weight * value
        
        # Add aggregation info
        metrics_aggregated["num_clients"] = float(len(results))
        metrics_aggregated["num_failures"] = float(len(failures))
        
        logger.info(
            f"[Adaptive] Round {server_round}: Aggregated {len(results)} clients, "
            f"{len(failures)} failures"
        )
        
        return ndarrays_to_parameters(aggregated_ndarrays), metrics_aggregated
    
    def _weighted_average(
        self,
        weights_results: List[Tuple[NDArrays, float]]
    ) -> NDArrays:
        """Compute weighted average of model parameters."""
        # Get total weight
        total_weight = sum(weight for _, weight in weights_results)
        
        # Initialize with zeros
        averaged = [
            np.zeros_like(layer) for layer in weights_results[0][0]
        ]
        
        # Weighted sum
        for parameters, weight in weights_results:
            for i, layer in enumerate(parameters):
                averaged[i] += (weight / total_weight) * layer
        
        return averaged
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        # Weighted average of losses
        loss_aggregated = weighted_loss_avg([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])
        
        # Aggregate metrics
        metrics_aggregated = {}
        total_examples = sum(res.num_examples for _, res in results)
        
        for _, evaluate_res in results:
            weight = evaluate_res.num_examples / total_examples
            for key, value in (evaluate_res.metrics or {}).items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = 0.0
                metrics_aggregated[key] += weight * value
        
        return loss_aggregated, metrics_aggregated
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (if configured)."""
        return None


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """Standard FedAvg with logging."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with logging."""
        logger.info(f"[FedAvg] Round {server_round}: {len(results)} clients, {len(failures)} failures")
        return super().aggregate_fit(server_round, results, failures)


class FedProxStrategy(fl.server.strategy.FedProx):
    """FedProx with logging."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with logging."""
        logger.info(f"[FedProx] Round {server_round}: {len(results)} clients, {len(failures)} failures")
        return super().aggregate_fit(server_round, results, failures)


def create_aggregation_strategy(
    strategy_name: str,
    initial_parameters: Parameters,
    config: Dict[str, Any]
) -> Strategy:
    """
    Factory function to create aggregation strategy.
    
    Args:
        strategy_name: Name of strategy ('fedavg', 'fedprox', 'adaptive')
        initial_parameters: Initial model parameters
        config: Configuration dictionary
    
    Returns:
        Strategy instance
    """
    common_args = {
        "fraction_fit": config.get("fraction_fit", 1.0),
        "fraction_evaluate": config.get("fraction_evaluate", 1.0),
        "min_fit_clients": config.get("min_fit_clients", 2),
        "min_evaluate_clients": config.get("min_evaluate_clients", 2),
        "min_available_clients": config.get("min_available_clients", 2),
        "initial_parameters": initial_parameters
    }
    
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "current_round": server_round,
            "local_epochs": config.get("local_epochs", 3)
        }
    
    if strategy_name == "fedavg":
        return FedAvgStrategy(
            **common_args,
            on_fit_config_fn=fit_config
        )
    
    elif strategy_name == "fedprox":
        return FedProxStrategy(
            **common_args,
            proximal_mu=config.get("proximal_mu", 0.01),
            on_fit_config_fn=fit_config
        )
    
    elif strategy_name == "adaptive":
        return AdaptiveAggregation(
            initial_parameters=initial_parameters,
            fraction_fit=common_args["fraction_fit"],
            fraction_evaluate=common_args["fraction_evaluate"],
            min_fit_clients=common_args["min_fit_clients"],
            min_evaluate_clients=common_args["min_evaluate_clients"],
            min_available_clients=common_args["min_available_clients"],
            sample_weight=config.get("sample_weight", 0.3),
            stability_weight=config.get("stability_weight", 0.25),
            loss_weight=config.get("loss_weight", 0.25),
            quality_weight=config.get("quality_weight", 0.2),
            stability_window=config.get("stability_window", 5),
            on_fit_config_fn=fit_config
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

