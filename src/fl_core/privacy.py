"""
Differential Privacy Module
============================
Implements differential privacy using Opacus for privacy-preserving training.
"""

from typing import Dict, Optional, Tuple, Any
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants import RDPAccountant

from loguru import logger


class DifferentialPrivacyEngine:
    """
    Wrapper for Opacus differential privacy.
    
    Provides ε-differential privacy guarantees for model training.
    """
    
    def __init__(
        self,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        epochs: int = 50,
        sample_size: int = 10000,
        batch_size: int = 32,
        secure_mode: bool = False
    ):
        """
        Initialize Differential Privacy Engine.
        
        Args:
            target_epsilon: Target privacy budget (epsilon)
            target_delta: Target delta for (ε,δ)-DP
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier (computed if None)
            epochs: Total training epochs
            sample_size: Total number of training samples
            batch_size: Batch size
            secure_mode: Use secure RNG for noise generation
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.secure_mode = secure_mode
        
        # Calculate noise multiplier if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier
        
        self.privacy_engine: Optional[PrivacyEngine] = None
        self._is_attached = False
        
        logger.info(f"DP Engine initialized:")
        logger.info(f"  - Target ε: {target_epsilon}")
        logger.info(f"  - Target δ: {target_delta}")
        logger.info(f"  - Max grad norm: {max_grad_norm}")
        logger.info(f"  - Noise multiplier: {self.noise_multiplier:.4f}")
    
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier to achieve target epsilon.
        
        Uses binary search to find the right noise level.
        """
        # Sampling rate
        sample_rate = self.batch_size / self.sample_size
        steps = self.epochs * (self.sample_size // self.batch_size)
        
        # Binary search for noise multiplier
        low, high = 0.01, 100.0
        
        while high - low > 0.001:
            mid = (low + high) / 2
            
            # Create accountant to check epsilon
            accountant = RDPAccountant()
            
            for _ in range(steps):
                accountant.step(
                    noise_multiplier=mid,
                    sample_rate=sample_rate
                )
            
            eps = accountant.get_epsilon(delta=self.target_delta)
            
            if eps > self.target_epsilon:
                low = mid
            else:
                high = mid
        
        return high
    
    def validate_model(self, model: nn.Module) -> nn.Module:
        """
        Validate and fix model for DP training.
        
        Replaces incompatible layers (BatchNorm -> GroupNorm, etc.)
        
        Args:
            model: PyTorch model
        
        Returns:
            DP-compatible model
        """
        # Check if model is compatible
        errors = ModuleValidator.validate(model, strict=False)
        
        if errors:
            logger.warning(f"Model has {len(errors)} DP compatibility issues. Fixing...")
            model = ModuleValidator.fix(model)
            logger.info("Model fixed for DP compatibility")
        else:
            logger.info("Model is DP-compatible")
        
        return model
    
    def make_private(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """
        Wrap model, optimizer, and data loader with DP.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            data_loader: Training data loader
        
        Returns:
            Tuple of (dp_model, dp_optimizer, dp_data_loader)
        """
        if self._is_attached:
            logger.warning("DP already attached. Detaching first.")
            self.detach()
        
        # Validate model
        model = self.validate_model(model)
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine(
            secure_mode=self.secure_mode
        )
        
        # Make private
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        
        self._is_attached = True
        
        logger.info("Model wrapped with differential privacy")
        
        return model, optimizer, data_loader
    
    def make_private_with_epsilon(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        epochs: int
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """
        Wrap with DP using target epsilon (auto-compute noise).
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            data_loader: Training data loader
            target_epsilon: Target privacy budget
            epochs: Number of training epochs
        
        Returns:
            Tuple of (dp_model, dp_optimizer, dp_data_loader)
        """
        if self._is_attached:
            logger.warning("DP already attached. Detaching first.")
            self.detach()
        
        # Validate model
        model = self.validate_model(model)
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine(
            secure_mode=self.secure_mode
        )
        
        # Make private with epsilon
        model, optimizer, data_loader = self.privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=self.target_delta,
            epochs=epochs,
            max_grad_norm=self.max_grad_norm
        )
        
        self._is_attached = True
        
        logger.info(f"Model wrapped with DP (target ε={target_epsilon})")
        
        return model, optimizer, data_loader
    
    def detach(self) -> None:
        """Detach privacy engine from model."""
        if self.privacy_engine is not None and self._is_attached:
            # Note: Opacus doesn't have a clean detach, but we can reset
            self._is_attached = False
            logger.info("DP engine detached")
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Get current privacy budget spent.
        
        Returns:
            Dictionary with epsilon and delta
        """
        if self.privacy_engine is None:
            return {"epsilon": 0.0, "delta": 0.0}
        
        try:
            epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
            return {
                "epsilon": epsilon,
                "delta": self.target_delta,
                "epsilon_remaining": max(0, self.target_epsilon - epsilon)
            }
        except Exception as e:
            logger.warning(f"Could not get privacy spent: {e}")
            return {"epsilon": 0.0, "delta": self.target_delta}
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """
        Get detailed privacy report.
        
        Returns:
            Dictionary with privacy statistics
        """
        spent = self.get_privacy_spent()
        
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "epsilon_spent": spent["epsilon"],
            "epsilon_remaining": spent.get("epsilon_remaining", self.target_epsilon),
            "budget_fraction_used": spent["epsilon"] / self.target_epsilon if self.target_epsilon > 0 else 0,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "is_private": self._is_attached
        }


def apply_dp_to_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    config: Dict[str, Any]
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, DifferentialPrivacyEngine]:
    """
    Convenience function to apply DP to a model.
    
    Args:
        model: Neural network
        optimizer: Optimizer
        data_loader: Data loader
        config: DP configuration
    
    Returns:
        Tuple of (model, optimizer, data_loader, dp_engine)
    """
    dp_engine = DifferentialPrivacyEngine(
        target_epsilon=config.get("epsilon", 8.0),
        target_delta=config.get("delta", 1e-5),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        noise_multiplier=config.get("noise_multiplier"),
        epochs=config.get("epochs", 50),
        sample_size=len(data_loader.dataset),
        batch_size=config.get("batch_size", 32)
    )
    
    model, optimizer, data_loader = dp_engine.make_private(
        model, optimizer, data_loader
    )
    
    return model, optimizer, data_loader, dp_engine


class GradientClipping:
    """
    Gradient clipping utilities for manual DP implementation.
    """
    
    @staticmethod
    def clip_gradients(
        model: nn.Module,
        max_norm: float
    ) -> float:
        """
        Clip gradients by norm.
        
        Args:
            model: Neural network
            max_norm: Maximum gradient norm
        
        Returns:
            Original gradient norm before clipping
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if not parameters:
            return 0.0
        
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in parameters])
        ).item()
        
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        return total_norm
    
    @staticmethod
    def add_noise(
        model: nn.Module,
        noise_multiplier: float,
        max_norm: float,
        device: torch.device
    ) -> None:
        """
        Add Gaussian noise to gradients.
        
        Args:
            model: Neural network
            noise_multiplier: Noise multiplier
            max_norm: Maximum gradient norm (determines noise scale)
            device: Device for noise tensor
        """
        noise_scale = noise_multiplier * max_norm
        
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad, device=device) * noise_scale
                p.grad.add_(noise)

