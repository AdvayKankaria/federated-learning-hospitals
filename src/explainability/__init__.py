# Explainability Module
# =====================
# Grad-CAM and other explainability methods for medical imaging

from .gradcam import GradCAMExplainer, generate_gradcam_visualization
from .utils import overlay_heatmap, save_explanation

__all__ = [
    "GradCAMExplainer",
    "generate_gradcam_visualization",
    "overlay_heatmap",
    "save_explanation"
]

