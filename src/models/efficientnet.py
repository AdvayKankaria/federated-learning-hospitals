"""
EfficientNet-B0 Model for Pneumonia Detection
==============================================
Transfer learning with EfficientNet-B0 pretrained on ImageNet.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from loguru import logger


class PneumoniaClassifier(nn.Module):
    """
    Pneumonia classifier based on EfficientNet-B0.
    
    Uses transfer learning from ImageNet pretrained weights.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        model_name: str = "efficientnet_b0"
    ):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            model_name: Name of the backbone model
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
        logger.info(f"Initialized {model_name} with {self.feature_dim} features")
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classifier.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
    def get_grad_cam_target_layer(self):
        """Get target layer for Grad-CAM visualization."""
        # For EfficientNet, use the last conv layer
        return self.backbone.conv_head


class PneumoniaClassifierWithFeatures(PneumoniaClassifier):
    """
    Extended classifier that also outputs features for adaptive aggregation.
    """
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional feature return.
        
        Args:
            x: Input tensor
            return_features: Whether to return features
        
        Returns:
            Tuple of (logits, features) if return_features else just logits
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None
) -> PneumoniaClassifier:
    """
    Factory function to create a model.
    
    Args:
        model_name: Name of backbone architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        checkpoint_path: Optional path to load weights from
    
    Returns:
        Initialized model
    """
    # Map of supported models
    supported_models = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b1": "efficientnet_b1",
        "efficientnet_b2": "efficientnet_b2",
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "densenet121": "densenet121",
    }
    
    if model_name not in supported_models:
        logger.warning(f"Model {model_name} not in supported list, using directly")
    
    model = PneumoniaClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        model_name=model_name
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "size_mb": total_params * 4 / (1024 ** 2)  # Assuming float32
    }

