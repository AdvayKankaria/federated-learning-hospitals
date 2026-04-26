"""
Federated Learning Hospital Client
===================================
Flower client implementation for hospital-side training.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    FitRes,
    EvaluateRes,
    Parameters,
    FitIns,
    EvaluateIns,
    Status,
    Code
)

from loguru import logger

from ..models import PneumoniaClassifier, create_model
from ..data import HospitalDataLoader
from .privacy import DifferentialPrivacyEngine


class HospitalClient(fl.client.NumPyClient):
    """
    Flower client representing a single hospital.
    
    Handles local training with optional differential privacy.
    """
    
    def __init__(
        self,
        hospital_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dp_engine: Optional[DifferentialPrivacyEngine] = None,
        proximal_mu: float = 0.0  # For FedProx
    ):
        """
        Initialize hospital client.
        
        Args:
            hospital_id: Unique identifier for this hospital
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            local_epochs: Number of local training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            dp_engine: Optional differential privacy engine
            proximal_mu: Proximal term coefficient for FedProx
        """
        self.hospital_id = hospital_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dp_engine = dp_engine
        self.proximal_mu = proximal_mu
        
        # Training state
        self.current_round = 0
        self.training_history: List[Dict[str, float]] = []
        
        # Store global model for FedProx
        self.global_model_params: Optional[List[torch.Tensor]] = None
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Hospital {hospital_id} client initialized")
        logger.info(f"  - Train samples: {len(train_loader.dataset)}")
        logger.info(f"  - Val samples: {len(val_loader.dataset)}")
        logger.info(f"  - DP enabled: {dp_engine is not None}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # Store for FedProx
        if self.proximal_mu > 0:
            self.global_model_params = [
                p.clone().detach() for p in self.model.parameters()
            ]
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        self.set_parameters(parameters)
        self.current_round = config.get("current_round", self.current_round + 1)
        
        # Get local epochs from config or use default
        local_epochs = config.get("local_epochs", self.local_epochs)
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Wrap with DP if enabled
        if self.dp_engine is not None:
            self.model, optimizer, self.train_loader = self.dp_engine.make_private(
                self.model, optimizer, self.train_loader
            )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (images, labels, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Add proximal term for FedProx
                if self.proximal_mu > 0 and self.global_model_params is not None:
                    proximal_term = 0.0
                    for local_param, global_param in zip(
                        self.model.parameters(), 
                        self.global_model_params
                    ):
                        proximal_term += ((local_param - global_param) ** 2).sum()
                    loss += (self.proximal_mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item() * images.size(0)
                epoch_samples += images.size(0)
                _, predicted = outputs.max(1)
                epoch_correct += predicted.eq(labels).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct += epoch_correct
            
            epoch_acc = epoch_correct / epoch_samples if epoch_samples > 0 else 0
            logger.debug(
                f"Hospital {self.hospital_id} - Epoch {epoch+1}/{local_epochs}: "
                f"Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_acc:.4f}"
            )
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct / total_samples if total_samples > 0 else 0
        
        # Get privacy spent if DP enabled
        privacy_spent = None
        if self.dp_engine is not None:
            privacy_spent = self.dp_engine.get_privacy_spent()
        
        metrics = {
            "hospital_id": float(self.hospital_id),
            "train_loss": float(avg_loss),
            "train_accuracy": float(accuracy),
            "num_samples": float(len(self.train_loader.dataset)),
            "local_epochs": float(local_epochs),
            "round": float(self.current_round)
        }
        
        if privacy_spent:
            metrics["epsilon_spent"] = float(privacy_spent.get("epsilon", 0))
            metrics["delta"] = float(privacy_spent.get("delta", 0))
        
        # Store history
        self.training_history.append(metrics)
        
        logger.info(
            f"Hospital {self.hospital_id} - Round {self.current_round}: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            metrics
        )
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For detailed metrics
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        # Calculate additional metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Sensitivity (Recall for pneumonia class)
        pneumonia_mask = all_labels == 1
        sensitivity = (
            np.sum((all_preds == 1) & pneumonia_mask) / np.sum(pneumonia_mask)
            if np.sum(pneumonia_mask) > 0 else 0
        )
        
        # Specificity (Recall for normal class)
        normal_mask = all_labels == 0
        specificity = (
            np.sum((all_preds == 0) & normal_mask) / np.sum(normal_mask)
            if np.sum(normal_mask) > 0 else 0
        )
        
        metrics = {
            "hospital_id": float(self.hospital_id),
            "val_loss": float(avg_loss),
            "val_accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "num_samples": float(total)
        }
        
        logger.info(
            f"Hospital {self.hospital_id} - Eval: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
            f"Sens={sensitivity:.4f}, Spec={specificity:.4f}"
        )
        
        return float(avg_loss), total, metrics
    
    def get_gradient_statistics(self) -> Dict[str, float]:
        """
        Get gradient statistics for adaptive aggregation.
        
        Returns:
            Dictionary with gradient statistics
        """
        grad_norms = []
        grad_vars = []
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                grad_vars.append(param.grad.var().item())
        
        return {
            "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
            "grad_norm_std": np.std(grad_norms) if grad_norms else 0,
            "grad_var_mean": np.mean(grad_vars) if grad_vars else 0
        }


def create_client(
    hospital_id: int,
    data_dir: str,
    partition: Dict,
    config: Dict[str, Any],
    device: torch.device
) -> HospitalClient:
    """
    Factory function to create a hospital client.
    
    Args:
        hospital_id: Hospital identifier
        data_dir: Path to dataset
        partition: Data partition dictionary
        config: Configuration dictionary
        device: Device to use
    
    Returns:
        Initialized HospitalClient
    """
    # Create data loaders
    data_loader = HospitalDataLoader(
        hospital_id=hospital_id,
        data_dir=data_dir,
        partition=partition,
        batch_size=config.get("batch_size", 32),
        image_size=config.get("image_size", 224),
        num_workers=config.get("num_workers", 4)
    )
    
    # Create model
    model = create_model(
        model_name=config.get("model_name", "efficientnet_b0"),
        num_classes=config.get("num_classes", 2),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3)
    )
    
    # Setup DP if enabled
    dp_engine = None
    if config.get("dp_enabled", False):
        dp_engine = DifferentialPrivacyEngine(
            target_epsilon=config.get("epsilon", 8.0),
            target_delta=config.get("delta", 1e-5),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            epochs=config.get("local_epochs", 3) * config.get("num_rounds", 50),
            sample_size=len(data_loader.train_dataset),
            batch_size=config.get("batch_size", 32)
        )
    
    return HospitalClient(
        hospital_id=hospital_id,
        model=model,
        train_loader=data_loader.get_train_loader(),
        val_loader=data_loader.get_val_loader(),
        device=device,
        local_epochs=config.get("local_epochs", 3),
        learning_rate=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0001),
        dp_engine=dp_engine,
        proximal_mu=config.get("proximal_mu", 0.0)
    )

