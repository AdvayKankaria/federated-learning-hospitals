"""
Data Partitioning for Federated Learning
=========================================
Creates Non-IID data partitions to simulate different hospitals.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from loguru import logger

from .dataset import RSNAPneumoniaDataset, get_transforms


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Partition data using Dirichlet distribution for Non-IID split.
    
    Lower alpha = more heterogeneous (non-IID)
    Higher alpha = more homogeneous (closer to IID)
    
    Args:
        labels: Array of labels
        num_clients: Number of clients/hospitals
        alpha: Dirichlet concentration parameter
        min_samples: Minimum samples per client
        seed: Random seed
    
    Returns:
        Dictionary mapping client_id to array of sample indices
    """
    np.random.seed(seed)
    
    num_samples = len(labels)
    num_classes = len(np.unique(labels))
    
    # Get indices for each class
    class_indices = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    
    # Initialize client indices
    client_indices = {i: [] for i in range(num_clients)}
    
    # For each class, distribute samples according to Dirichlet
    for c in class_indices.keys():
        indices = class_indices[c].copy()
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Ensure minimum samples
        proportions = np.array([
            max(p, min_samples / len(indices)) if len(indices) > 0 else p 
            for p in proportions
        ])
        proportions = proportions / proportions.sum()
        
        # Split indices according to proportions
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, split_points)
        
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    
    # Convert to numpy arrays
    client_indices = {k: np.array(v) for k, v in client_indices.items()}
    
    return client_indices


def quantity_skew_partition(
    labels: np.ndarray,
    num_clients: int,
    min_samples: int = 100,
    max_samples: int = 2000,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Partition with quantity skew - different hospitals have different amounts of data.
    """
    np.random.seed(seed)
    
    num_samples = len(labels)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Generate random sample sizes for each client
    sample_sizes = np.random.randint(min_samples, max_samples + 1, num_clients)
    
    # Scale to fit total samples
    total_requested = sample_sizes.sum()
    if total_requested > num_samples:
        sample_sizes = (sample_sizes * num_samples / total_requested).astype(int)
    
    # Distribute samples
    client_indices = {}
    start_idx = 0
    
    for client_id in range(num_clients):
        end_idx = min(start_idx + sample_sizes[client_id], num_samples)
        client_indices[client_id] = indices[start_idx:end_idx]
        start_idx = end_idx
        
        if start_idx >= num_samples:
            break
    
    # Distribute remaining samples
    remaining = indices[start_idx:]
    for i, idx in enumerate(remaining):
        client_id = i % num_clients
        client_indices[client_id] = np.append(client_indices[client_id], idx)
    
    return client_indices


def iid_partition(
    labels: np.ndarray,
    num_clients: int,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    IID partition - randomly distribute samples equally.
    """
    np.random.seed(seed)
    
    num_samples = len(labels)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Split equally
    splits = np.array_split(indices, num_clients)
    
    return {i: splits[i] for i in range(num_clients)}


def create_hospital_partitions(
    data_dir: str,
    num_hospitals: int = 5,
    partition_type: str = "non_iid",
    dirichlet_alpha: float = 0.5,
    min_samples: int = 100,
    max_samples: int = 2000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    save_partition: bool = True
) -> Dict[int, Dict[str, List[str]]]:
    """
    Create data partitions for simulating multiple hospitals.
    
    Args:
        data_dir: Path to dataset directory
        num_hospitals: Number of hospitals to simulate
        partition_type: 'iid', 'non_iid', or 'quantity_skew'
        dirichlet_alpha: Alpha for Dirichlet distribution (non_iid)
        min_samples: Minimum samples per hospital
        max_samples: Maximum samples per hospital
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        seed: Random seed
        save_partition: Whether to save partition to file
    
    Returns:
        Dictionary mapping hospital_id to dict with train/val/test image_ids
    """
    data_path = Path(data_dir)
    
    # Build file list from directory structure
    all_files = []
    all_labels = []
    all_ids = []
    
    # Check train directory
    train_dir = data_path / "train"
    if train_dir.exists():
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = train_dir / class_name
            if class_dir.exists():
                label = 1 if class_name == "PNEUMONIA" else 0
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png', '.dcm']:
                        all_files.append(str(img_path))
                        all_labels.append(label)
                        all_ids.append(img_path.stem)
    
    # Also add test data if exists
    test_dir = data_path / "test"
    if test_dir.exists():
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = test_dir / class_name
            if class_dir.exists():
                label = 1 if class_name == "PNEUMONIA" else 0
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png', '.dcm']:
                        all_files.append(str(img_path))
                        all_labels.append(label)
                        all_ids.append(img_path.stem)
    
    # Add val data if exists
    val_dir = data_path / "val"
    if val_dir.exists():
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = val_dir / class_name
            if class_dir.exists():
                label = 1 if class_name == "PNEUMONIA" else 0
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png', '.dcm']:
                        all_files.append(str(img_path))
                        all_labels.append(label)
                        all_ids.append(img_path.stem)
    
    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)
    
    logger.info(f"Total images: {len(all_ids)}")
    logger.info(f"Class distribution: Normal={np.sum(all_labels==0)}, Pneumonia={np.sum(all_labels==1)}")
    
    if len(all_ids) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    # Create partition based on type
    if partition_type == "iid":
        partition = iid_partition(all_labels, num_hospitals, seed)
    elif partition_type == "non_iid":
        partition = dirichlet_partition(all_labels, num_hospitals, dirichlet_alpha, seed=seed)
    elif partition_type == "quantity_skew":
        partition = quantity_skew_partition(
            all_labels, num_hospitals, min_samples, max_samples, seed
        )
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    
    # Create train/val/test splits for each hospital
    hospital_data = {}
    
    for hospital_id in range(num_hospitals):
        indices = partition[hospital_id]
        hospital_ids = all_ids[indices]
        hospital_labels = all_labels[indices]
        
        if len(hospital_ids) < 10:
            logger.warning(f"Hospital {hospital_id} has only {len(hospital_ids)} samples")
            # Give minimum samples
            hospital_data[hospital_id] = {
                'train': hospital_ids.tolist(),
                'val': hospital_ids[:max(1, len(hospital_ids)//5)].tolist(),
                'test': hospital_ids[:max(1, len(hospital_ids)//5)].tolist()
            }
            continue
        
        # Split into train/val/test
        test_ratio = 1.0 - train_ratio - val_ratio
        
        try:
            # First split: train+val vs test
            train_val_ids, test_ids, train_val_labels, _ = train_test_split(
                hospital_ids,
                hospital_labels,
                test_size=max(0.1, test_ratio),
                stratify=hospital_labels if len(np.unique(hospital_labels)) > 1 else None,
                random_state=seed
            )
            
            # Second split: train vs val
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            train_ids, val_ids, _, _ = train_test_split(
                train_val_ids,
                train_val_labels,
                test_size=max(0.1, val_ratio_adjusted),
                stratify=train_val_labels if len(np.unique(train_val_labels)) > 1 else None,
                random_state=seed
            )
        except ValueError as e:
            # Fall back to simple split if stratification fails
            logger.warning(f"Stratification failed for hospital {hospital_id}, using simple split: {e}")
            n = len(hospital_ids)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            train_ids = hospital_ids[:train_end]
            val_ids = hospital_ids[train_end:val_end]
            test_ids = hospital_ids[val_end:]
        
        hospital_data[hospital_id] = {
            'train': train_ids.tolist() if hasattr(train_ids, 'tolist') else list(train_ids),
            'val': val_ids.tolist() if hasattr(val_ids, 'tolist') else list(val_ids),
            'test': test_ids.tolist() if hasattr(test_ids, 'tolist') else list(test_ids)
        }
        
        # Log statistics
        train_labels = all_labels[np.isin(all_ids, train_ids)]
        logger.info(
            f"Hospital {hospital_id}: "
            f"Train={len(train_ids)} (P:{np.sum(train_labels==1)}), "
            f"Val={len(val_ids)}, Test={len(test_ids)}"
        )
    
    # Save partition to file
    if save_partition:
        partition_file = data_path / f"partition_{partition_type}_{num_hospitals}hospitals.json"
        
        # Convert to serializable format
        serializable_data = {}
        for h_id, data in hospital_data.items():
            serializable_data[str(h_id)] = {
                k: [str(x) for x in v] for k, v in data.items()
            }
        
        with open(partition_file, 'w') as f:
            json.dump({
                'config': {
                    'partition_type': partition_type,
                    'num_hospitals': num_hospitals,
                    'dirichlet_alpha': dirichlet_alpha if partition_type == 'non_iid' else None,
                    'seed': seed
                },
                'partitions': serializable_data
            }, f, indent=2)
        
        logger.info(f"Partition saved to {partition_file}")
    
    return hospital_data


def load_partition(partition_file: str) -> Dict[int, Dict[str, List[str]]]:
    """Load a previously saved partition."""
    with open(partition_file, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to int
    partitions = {}
    for h_id, splits in data['partitions'].items():
        partitions[int(h_id)] = splits
    
    return partitions


class HospitalDataLoader:
    """
    Creates DataLoaders for a specific hospital.
    """
    
    def __init__(
        self,
        hospital_id: int,
        data_dir: str,
        partition: Dict[int, Dict[str, List[str]]],
        batch_size: int = 32,
        image_size: int = 224,
        num_workers: int = 4,
        pin_memory: bool = True,
        weighted_sampling: bool = True
    ):
        """
        Initialize hospital data loader.
        """
        self.hospital_id = hospital_id
        self.data_dir = data_dir
        self.partition = partition[hospital_id]
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.weighted_sampling = weighted_sampling
        
        # Create datasets
        self._create_datasets()
    
    def _create_datasets(self):
        """Create train, val, test datasets."""
        self.train_dataset = RSNAPneumoniaDataset(
            data_dir=self.data_dir,
            image_ids=self.partition['train'],
            image_size=self.image_size,
            is_training=True,
            split="train"
        )
        
        self.val_dataset = RSNAPneumoniaDataset(
            data_dir=self.data_dir,
            image_ids=self.partition['val'],
            image_size=self.image_size,
            is_training=False,
            split="train"  # Load from train folder structure
        )
        
        self.test_dataset = RSNAPneumoniaDataset(
            data_dir=self.data_dir,
            image_ids=self.partition['test'],
            image_size=self.image_size,
            is_training=False,
            split="train"  # Load from train folder structure
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        if self.weighted_sampling and len(self.train_dataset) > 0:
            sample_weights = self.train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=min(self.batch_size, len(self.train_dataset)),
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=len(self.train_dataset) > self.batch_size
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=min(self.batch_size, max(1, len(self.train_dataset))),
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=len(self.train_dataset) > self.batch_size
            )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, max(1, len(self.val_dataset))),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=min(self.batch_size, max(1, len(self.test_dataset))),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data statistics for this hospital."""
        return {
            'hospital_id': self.hospital_id,
            'train': self.train_dataset.get_statistics(),
            'val': self.val_dataset.get_statistics(),
            'test': self.test_dataset.get_statistics()
        }
