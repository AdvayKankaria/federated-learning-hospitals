"""
Model Utilities
===============
Helper functions for model management.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size in various units.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with size in different units
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        "params_bytes": param_size,
        "buffers_bytes": buffer_size,
        "total_bytes": total_size,
        "total_kb": total_size / 1024,
        "total_mb": total_size / (1024 ** 2)
    }


def get_layer_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get information about each layer.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with layer information
    """
    layer_info = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layer_info[name] = {
                "type": type(module).__name__,
                "parameters": params,
                "trainable": trainable
            }
    
    return layer_info


def copy_model_weights(source: nn.Module, target: nn.Module) -> None:
    """
    Copy weights from source to target model.
    
    Args:
        source: Source model
        target: Target model
    """
    target.load_state_dict(source.state_dict())


def average_model_weights(models: list) -> Dict[str, torch.Tensor]:
    """
    Average weights from multiple models.
    
    Args:
        models: List of models
    
    Returns:
        Dictionary of averaged state dict
    """
    if not models:
        raise ValueError("No models provided")
    
    avg_state_dict = {}
    state_dicts = [m.state_dict() for m in models]
    
    for key in state_dicts[0].keys():
        avg_state_dict[key] = torch.stack(
            [sd[key].float() for sd in state_dicts]
        ).mean(dim=0)
    
    return avg_state_dict


def weighted_average_weights(
    state_dicts: list,
    weights: list
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of model weights.
    
    Args:
        state_dicts: List of state dictionaries
        weights: List of weights for each state dict
    
    Returns:
        Weighted averaged state dict
    """
    if not state_dicts:
        raise ValueError("No state dicts provided")
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    avg_state_dict = {}
    
    for key in state_dicts[0].keys():
        weighted_sum = sum(
            w * sd[key].float() 
            for w, sd in zip(normalized_weights, state_dicts)
        )
        avg_state_dict[key] = weighted_sum
    
    return avg_state_dict

