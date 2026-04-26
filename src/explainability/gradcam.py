"""
Grad-CAM Explainability
========================
Gradient-weighted Class Activation Mapping for model interpretability.
Custom implementation compatible with all Python versions.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from loguru import logger


class GradCAM:
    """
    Custom Grad-CAM implementation.
    
    Computes Gradient-weighted Class Activation Maps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Neural network model
            target_layer: Layer to compute CAM for
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index (None for predicted class)
        
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Compute weighted combination
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam[0, 0].cpu().numpy()


class GradCAMExplainer:
    """
    Grad-CAM based explainability for pneumonia detection.
    
    Highlights regions of X-rays that contribute to the model's prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[nn.Module]] = None,
        device: torch.device = torch.device('cpu'),
        method: str = "gradcam"
    ):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: Neural network model
            target_layers: Layers to use for CAM (auto-detect if None)
            device: Device for computation
            method: CAM method (currently only gradcam supported)
        """
        self.model = model.to(device)
        self.device = device
        self.method = method
        
        # Auto-detect target layer if not provided
        target_layer = self._find_target_layer() if target_layers is None else target_layers[0]
        
        self.cam = GradCAM(model=model, target_layer=target_layer)
        
        logger.info(f"GradCAM Explainer initialized")
    
    def _find_target_layer(self) -> nn.Module:
        """
        Auto-detect target layer for Grad-CAM.
        
        Returns:
            Target layer
        """
        # For EfficientNet
        if hasattr(self.model, 'backbone'):
            if hasattr(self.model.backbone, 'conv_head'):
                return self.model.backbone.conv_head
            elif hasattr(self.model.backbone, 'features'):
                # Get last conv layer
                for layer in reversed(list(self.model.backbone.features.modules())):
                    if isinstance(layer, nn.Conv2d):
                        return layer
        
        # For ResNet
        if hasattr(self.model, 'layer4'):
            return self.model.layer4[-1]
        
        # For DenseNet
        if hasattr(self.model, 'features'):
            for layer in reversed(list(self.model.features.modules())):
                if isinstance(layer, nn.Conv2d):
                    return layer
        
        # Default: find last conv layer
        last_conv = None
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
        
        if last_conv is None:
            raise ValueError("Could not find target layer for Grad-CAM")
        
        return last_conv
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Grad-CAM explanation.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class (None for predicted class)
            normalize: Whether to normalize the heatmap
        
        Returns:
            Grad-CAM heatmap (H, W)
        """
        input_tensor = input_tensor.to(self.device)
        heatmap = self.cam(input_tensor, target_class)
        
        if normalize:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def explain_batch(
        self,
        input_batch: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM explanations for a batch.
        
        Args:
            input_batch: Batch of images (B, C, H, W)
            target_classes: List of target classes
        
        Returns:
            List of heatmaps
        """
        heatmaps = []
        
        for i in range(input_batch.size(0)):
            target = target_classes[i] if target_classes else None
            heatmap = self.explain(input_batch[i:i+1], target)
            heatmaps.append(heatmap)
        
        return heatmaps
    
    def visualize(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> np.ndarray:
        """
        Overlay heatmap on image.
        
        Args:
            image: Original image (H, W, C) in range [0, 1]
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Transparency of heatmap
            colormap: Matplotlib colormap name
        
        Returns:
            Visualization image (H, W, C)
        """
        # Ensure image is float in [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]
        
        # Resize heatmap to match image if needed
        if heatmap_colored.shape[:2] != image.shape[:2]:
            from PIL import Image as PILImage
            heatmap_pil = PILImage.fromarray((heatmap_colored * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]))
            heatmap_colored = np.array(heatmap_pil) / 255.0
        
        # Ensure image has 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        
        # Overlay
        visualization = (1 - alpha) * image + alpha * heatmap_colored
        visualization = np.clip(visualization, 0, 1)
        
        return visualization
    
    def generate_report(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        patient_id: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete explanation report.
        
        Args:
            image: Original image
            input_tensor: Preprocessed input tensor
            patient_id: Patient identifier
            save_path: Optional path to save visualization
        
        Returns:
            Report dictionary
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Generate heatmap
        heatmap = self.explain(input_tensor, pred_class)
        
        # Create visualization
        if image.size == 0 or image.shape[0] == 0:
            # Create placeholder image
            image = np.zeros((224, 224, 3))
        
        visualization = self.visualize(image, heatmap)
        
        # Find most important region
        important_region = self._find_important_region(heatmap)
        
        report = {
            "patient_id": patient_id,
            "prediction": "Pneumonia" if pred_class == 1 else "Normal",
            "confidence": confidence,
            "class_probabilities": {
                "Normal": probs[0, 0].item(),
                "Pneumonia": probs[0, 1].item()
            },
            "heatmap": heatmap,
            "visualization": visualization,
            "important_region": important_region
        }
        
        # Save if path provided
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _find_important_region(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, int]:
        """Find bounding box of most important region."""
        binary = heatmap > threshold
        
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        if not rows.any() or not cols.any():
            return {"x": 0, "y": 0, "width": heatmap.shape[1], "height": heatmap.shape[0]}
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return {
            "x": int(cmin),
            "y": int(rmin),
            "width": int(cmax - cmin),
            "height": int(rmax - rmin)
        }
    
    def _save_report(self, report: Dict, save_path: str) -> None:
        """Save visualization to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Original (grayscale representation)
        axes[0].imshow(report["visualization"][:, :, 0], cmap='gray')
        axes[0].set_title("X-ray", color='white')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(report["heatmap"], cmap='jet')
        axes[1].set_title("Attention Map", color='white')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(report["visualization"])
        axes[2].set_title(
            f"Prediction: {report['prediction']} ({report['confidence']:.1%})",
            color='white'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        logger.info(f"Saved explanation to {save_path}")


def generate_gradcam_visualization(
    model: nn.Module,
    image: np.ndarray,
    input_tensor: torch.Tensor,
    device: torch.device,
    target_class: Optional[int] = None,
    method: str = "gradcam"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate Grad-CAM visualization.
    
    Args:
        model: Neural network
        image: Original image
        input_tensor: Preprocessed input
        device: Computation device
        target_class: Target class
        method: CAM method
    
    Returns:
        Tuple of (heatmap, visualization)
    """
    explainer = GradCAMExplainer(model=model, device=device, method=method)
    
    heatmap = explainer.explain(input_tensor, target_class)
    visualization = explainer.visualize(image, heatmap)
    
    return heatmap, visualization
