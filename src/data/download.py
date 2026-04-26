"""
Chest X-Ray Pneumonia Dataset Downloader
=========================================
Downloads and extracts the Chest X-Ray Pneumonia dataset from Kaggle.
"""

import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional
import subprocess
import sys

from loguru import logger


def setup_kaggle_credentials(kaggle_json_path: Optional[str] = None) -> bool:
    """
    Setup Kaggle API credentials.
    
    Args:
        kaggle_json_path: Path to kaggle.json file. If None, looks for it in
                         current directory or ~/.kaggle/
    
    Returns:
        True if credentials are set up successfully
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    target_path = kaggle_dir / "kaggle.json"
    
    # Check if already exists
    if target_path.exists():
        logger.info("Kaggle credentials already configured")
        os.chmod(target_path, 0o600)
        return True
    
    # Look for kaggle.json
    search_paths = [
        Path(kaggle_json_path) if kaggle_json_path else None,
        Path("./kaggle.json"),
        Path("../kaggle.json"),
    ]
    
    for path in search_paths:
        if path and path.exists():
            shutil.copy(path, target_path)
            os.chmod(target_path, 0o600)
            logger.info(f"Kaggle credentials copied from {path}")
            return True
    
    # Check environment variable
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logger.info("Using Kaggle credentials from environment variables")
        return True
    
    logger.error("Kaggle credentials not found. Please provide kaggle.json")
    return False


def download_rsna_dataset(
    data_dir: str = "./data/rsna-pneumonia",
    kaggle_json_path: Optional[str] = None,
    force_download: bool = False
) -> Path:
    """
    Download Chest X-Ray Pneumonia dataset from Kaggle.
    
    Uses the popular paultimothymooney/chest-xray-pneumonia dataset.
    
    Args:
        data_dir: Directory to save the dataset
        kaggle_json_path: Path to kaggle.json credentials file
        force_download: If True, re-download even if data exists
    
    Returns:
        Path to the downloaded dataset directory
    """
    data_path = Path(data_dir)
    
    # Check if already downloaded
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not force_download and train_dir.exists() and test_dir.exists():
        # Check if we have images
        train_images = list(train_dir.rglob("*.jpeg")) + list(train_dir.rglob("*.jpg")) + list(train_dir.rglob("*.png"))
        if len(train_images) > 0:
            logger.info(f"Dataset already exists at {data_path} with {len(train_images)} training images")
            return data_path
    
    # Setup credentials
    if not setup_kaggle_credentials(kaggle_json_path):
        raise RuntimeError("Failed to setup Kaggle credentials")
    
    # Create data directory
    data_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading Chest X-Ray Pneumonia dataset from Kaggle...")
    logger.info("This may take several minutes (~2GB download)...")
    
    try:
        # Import kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset (not competition)
        dataset = "paultimothymooney/chest-xray-pneumonia"
        
        logger.info(f"Downloading dataset: {dataset}")
        api.dataset_download_files(
            dataset=dataset,
            path=str(data_path),
            quiet=False,
            unzip=False
        )
        
        # Extract zip files
        logger.info("Extracting downloaded files...")
        for zip_file in data_path.glob("*.zip"):
            logger.info(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(data_path)
            # Remove zip after extraction
            zip_file.unlink()
        
        # The dataset extracts to chest_xray folder, move contents up
        chest_xray_dir = data_path / "chest_xray"
        if chest_xray_dir.exists():
            for item in chest_xray_dir.iterdir():
                shutil.move(str(item), str(data_path / item.name))
            chest_xray_dir.rmdir()
        
        # Create labels CSV for compatibility
        _create_labels_csv(data_path)
        
        # Verify download
        train_images = list((data_path / "train").rglob("*.jpeg")) + \
                      list((data_path / "train").rglob("*.jpg")) + \
                      list((data_path / "train").rglob("*.png"))
        
        logger.success(f"Download complete! {len(train_images)} training images available")
        
        return data_path
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def _create_labels_csv(data_path: Path) -> None:
    """
    Create a labels CSV file for the dataset.
    
    The Chest X-Ray dataset has folder structure:
    - train/NORMAL/*.jpeg
    - train/PNEUMONIA/*.jpeg
    - test/NORMAL/*.jpeg
    - test/PNEUMONIA/*.jpeg
    """
    import pandas as pd
    
    records = []
    
    for split in ["train", "test", "val"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name.upper()
            target = 1 if "PNEUMONIA" in class_name else 0
            
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    records.append({
                        'patientId': img_file.stem,
                        'Target': target,
                        'split': split,
                        'filepath': str(img_file.relative_to(data_path))
                    })
    
    df = pd.DataFrame(records)
    df.to_csv(data_path / "stage_2_train_labels.csv", index=False)
    logger.info(f"Created labels CSV with {len(df)} entries")


def verify_dataset(data_dir: str = "./data/rsna-pneumonia") -> dict:
    """
    Verify the downloaded dataset and return statistics.
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    stats = {
        "valid": False,
        "num_train_images": 0,
        "num_test_images": 0,
        "num_val_images": 0,
        "has_labels": False,
        "class_distribution": {}
    }
    
    # Count images in each split
    for split in ["train", "test", "val"]:
        split_dir = data_path / split
        if split_dir.exists():
            images = list(split_dir.rglob("*.jpeg")) + \
                    list(split_dir.rglob("*.jpg")) + \
                    list(split_dir.rglob("*.png"))
            stats[f"num_{split}_images"] = len(images)
            
            # Count by class
            for class_name in ["NORMAL", "PNEUMONIA"]:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    class_images = list(class_dir.glob("*"))
                    stats["class_distribution"][f"{split}_{class_name}"] = len(class_images)
    
    # Check labels
    labels_file = data_path / "stage_2_train_labels.csv"
    stats["has_labels"] = labels_file.exists()
    
    # Validate
    stats["valid"] = stats["num_train_images"] > 0 and stats["has_labels"]
    
    return stats


if __name__ == "__main__":
    # Test download
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Chest X-Ray dataset")
    parser.add_argument("--data-dir", default="./data/rsna-pneumonia")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    download_rsna_dataset(args.data_dir, force_download=args.force)
    
    stats = verify_dataset(args.data_dir)
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
