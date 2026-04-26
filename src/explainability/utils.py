"""
Explainability Utilities
========================
Helper functions for visualizations and explanations.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Base image (H, W) or (H, W, C)
        heatmap: Heatmap array (H, W) in range [0, 1]
        alpha: Heatmap transparency
        colormap: Colormap name
    
    Returns:
        Overlaid image (H, W, 3)
    """
    # Ensure image is 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Normalize image to [0, 1]
    if image.max() > 1:
        image = image / 255.0
    
    # Resize heatmap to match image
    if heatmap.shape != image.shape[:2]:
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]))
        heatmap = np.array(heatmap_pil) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]
    
    # Blend
    result = (1 - alpha) * image + alpha * heatmap_colored
    return np.clip(result, 0, 1)


def save_explanation(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    visualization: np.ndarray,
    prediction: str,
    confidence: float,
    save_path: str,
    patient_id: Optional[str] = None
) -> None:
    """
    Save explanation visualization to file.
    
    Args:
        original_image: Original image
        heatmap: Grad-CAM heatmap
        visualization: Overlaid visualization
        prediction: Model prediction
        confidence: Prediction confidence
        save_path: Path to save image
        patient_id: Optional patient identifier
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Style
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
    
    # Original image
    if original_image.ndim == 3:
        axes[0].imshow(original_image[:, :, 0], cmap='gray')
    else:
        axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original X-ray", color='white', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Attention Map", color='white', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(visualization)
    title = f"Prediction: {prediction}\nConfidence: {confidence:.1%}"
    if patient_id:
        title = f"Patient: {patient_id}\n" + title
    axes[2].set_title(title, color='white', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()


def create_comparison_figure(
    images: list,
    heatmaps: list,
    predictions: list,
    confidences: list,
    save_path: str,
    title: str = "Model Explanations"
) -> None:
    """
    Create comparison figure for multiple images.
    
    Args:
        images: List of original images
        heatmaps: List of heatmaps
        predictions: List of predictions
        confidences: List of confidences
        save_path: Path to save
        title: Figure title
    """
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold')
    
    for i in range(n):
        # Original
        if images[i].ndim == 3:
            axes[0, i].imshow(images[i][:, :, 0], cmap='gray')
        else:
            axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Sample {i+1}", color='white')
        
        # Heatmap overlay
        vis = overlay_heatmap(images[i], heatmaps[i])
        axes[1, i].imshow(vis)
        axes[1, i].axis('off')
        pred_color = '#00ff88' if predictions[i] == 'Normal' else '#ff6b6b'
        axes[1, i].set_title(
            f"{predictions[i]}: {confidences[i]:.1%}",
            color=pred_color
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > 0:
        return (image - img_min) / (img_max - img_min)
    return image


def denormalize_tensor(
    tensor: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize a tensor that was normalized with ImageNet stats.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (H, W, C)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized array
    """
    if tensor.shape[0] == 3:  # (C, H, W)
        tensor = tensor.transpose(1, 2, 0)
    
    mean = np.array(mean)
    std = np.array(std)
    
    denorm = tensor * std + mean
    return np.clip(denorm, 0, 1)

