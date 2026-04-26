"""
Chest X-Ray Pneumonia Dataset Classes
=====================================
PyTorch Dataset implementation for Pneumonia Detection.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from loguru import logger


def read_image(path: str) -> np.ndarray:
    """
    Read an image file and return as numpy array.
    
    Args:
        path: Path to image file (JPEG, PNG, or DICOM)
    
    Returns:
        Image as numpy array (H, W)
    """
    path = str(path)
    
    # Check if DICOM
    if path.endswith('.dcm'):
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            dicom = pydicom.dcmread(path)
            data = apply_voi_lut(dicom.pixel_array, dicom)
            # Normalize to 0-255
            if data.max() != data.min():
                data = (data - data.min()) / (data.max() - data.min())
            data = (data * 255).astype(np.uint8)
            return data
        except Exception as e:
            logger.warning(f"Failed to read DICOM {path}: {e}")
    
    # Regular image
    img = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(img)


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    augmentation_strength: str = "medium"
) -> A.Compose:
    """
    Get image transforms for training or validation.
    
    Args:
        image_size: Target image size
        is_training: Whether these are training transforms
        augmentation_strength: 'light', 'medium', or 'strong'
    
    Returns:
        Albumentations Compose object
    """
    if is_training:
        if augmentation_strength == "light":
            transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        elif augmentation_strength == "medium":
            transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ], p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:  # strong
            transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.2,
                    rotate_limit=20,
                    p=0.7
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=5),
                ], p=0.4),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.05),
                    A.GridDistortion(num_steps=5, distort_limit=0.05),
                ], p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=image_size // 16,
                    max_width=image_size // 16,
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    else:
        # Validation/Test transforms - no augmentation
        transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transforms


class RSNAPneumoniaDataset(Dataset):
    """
    PyTorch Dataset for Chest X-Ray Pneumonia Detection.
    
    Handles both DICOM and regular images (JPEG/PNG).
    """
    
    def __init__(
        self,
        data_dir: str,
        image_ids: Optional[List[str]] = None,
        labels_df: Optional[pd.DataFrame] = None,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        is_training: bool = True,
        cache_images: bool = False,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to dataset directory
            image_ids: List of image IDs to include (for partitioning)
            labels_df: DataFrame with labels (if not provided, loads from file)
            transform: Image transforms to apply
            image_size: Target image size
            is_training: Whether this is training data
            cache_images: Whether to cache images in memory
            split: Data split ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.cache_images = cache_images
        self.split = split
        self.image_cache: Dict[str, np.ndarray] = {}
        
        # Build file list directly from directory structure
        self._build_file_list(image_ids)
        
        # Set transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(image_size, is_training)
        
        # Calculate class weights for imbalanced data
        self._calculate_class_weights()
        
        logger.info(f"Dataset initialized with {len(self)} images")
        logger.info(f"Class distribution - Normal: {self.class_counts.get(0, 0)}, Pneumonia: {self.class_counts.get(1, 0)}")
    
    def _build_file_list(self, image_ids: Optional[List[str]] = None):
        """Build list of image files and labels from directory structure."""
        self.image_paths = []
        self.labels = []
        self.image_ids = []
        
        # Check for folder-based structure
        split_dir = self.data_dir / self.split
        if split_dir.exists():
            for class_name in ["NORMAL", "PNEUMONIA"]:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue
                
                label = 1 if class_name == "PNEUMONIA" else 0
                
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png', '.dcm']:
                        img_id = img_path.stem
                        
                        # Filter by image_ids if provided
                        if image_ids is not None and img_id not in image_ids:
                            continue
                        
                        self.image_paths.append(str(img_path))
                        self.labels.append(label)
                        self.image_ids.append(img_id)
        else:
            # Fall back to CSV-based loading
            labels_path = self.data_dir / "stage_2_train_labels.csv"
            if labels_path.exists():
                df = pd.read_csv(labels_path)
                
                # Filter by split if available
                if 'split' in df.columns:
                    df = df[df['split'] == self.split]
                
                # Filter by image_ids if provided
                if image_ids is not None:
                    df = df[df['patientId'].isin(image_ids)]
                
                for _, row in df.iterrows():
                    if 'filepath' in row:
                        img_path = self.data_dir / row['filepath']
                    else:
                        # Try to find the image
                        img_id = row['patientId']
                        img_path = None
                        for ext in ['.jpeg', '.jpg', '.png', '.dcm']:
                            candidate = self.data_dir / f"{self.split}" / ("PNEUMONIA" if row['Target'] else "NORMAL") / f"{img_id}{ext}"
                            if candidate.exists():
                                img_path = candidate
                                break
                    
                    if img_path and Path(img_path).exists():
                        self.image_paths.append(str(img_path))
                        self.labels.append(int(row['Target']))
                        self.image_ids.append(row['patientId'])
    
    def _calculate_class_weights(self):
        """Calculate class weights for handling imbalanced data."""
        labels_array = np.array(self.labels)
        unique_labels = np.unique(labels_array)
        
        self.class_counts = {}
        self.class_weights = {}
        
        for label in [0, 1]:
            count = np.sum(labels_array == label)
            self.class_counts[label] = count
        
        total = len(labels_array)
        for label in [0, 1]:
            count = self.class_counts.get(label, 0)
            self.class_weights[label] = total / (2 * count) if count > 0 else 1.0
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single item.
        
        Returns:
            Tuple of (image_tensor, label, patient_id)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        patient_id = self.image_ids[idx]
        
        # Load image
        if self.cache_images and patient_id in self.image_cache:
            image = self.image_cache[patient_id]
        else:
            image = read_image(img_path)
            
            if self.cache_images:
                self.image_cache[patient_id] = image
        
        # Convert grayscale to RGB (3 channels for pretrained models)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label, patient_id
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for weighted sampling."""
        weights = [self.class_weights[label] for label in self.labels]
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        return self.class_counts.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "num_samples": len(self),
            "num_normal": self.class_counts.get(0, 0),
            "num_pneumonia": self.class_counts.get(1, 0),
            "pneumonia_ratio": self.class_counts.get(1, 0) / len(self) if len(self) > 0 else 0,
            "class_weights": self.class_weights
        }


class RSNAInferenceDataset(Dataset):
    """
    Dataset for inference without labels.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(image_size, is_training=False)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        
        # Convert to RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, image_path
