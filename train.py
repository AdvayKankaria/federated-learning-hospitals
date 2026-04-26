#!/usr/bin/env python3
"""
Federated Learning Training Script
===================================
Main orchestration script for hospital federated learning with
differential privacy and adaptive aggregation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import asyncio
import threading

import yaml
import torch
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters

from loguru import logger

# Add source to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import download_rsna_dataset, create_hospital_partitions, HospitalDataLoader
from src.models import create_model, get_model_info
from src.fl_core import (
    HospitalClient,
    create_aggregation_strategy,
    DifferentialPrivacyEngine,
    MetricsCollector
)
from src.explainability import GradCAMExplainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def client_fn(
    cid: str,
    data_dir: str,
    partition: Dict,
    config: Dict,
    device: torch.device
) -> fl.client.NumPyClient:
    """
    Create a Flower client for a hospital.
    
    Args:
        cid: Client ID (hospital index)
        data_dir: Path to dataset
        partition: Data partition
        config: Configuration dictionary
        device: Device for training
    
    Returns:
        Flower NumPyClient
    """
    hospital_id = int(cid)
    
    # Create data loader
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
    
    # Setup DP engine if enabled
    dp_engine = None
    if config.get("dp_enabled", True):
        dp_engine = DifferentialPrivacyEngine(
            target_epsilon=config.get("epsilon", 8.0),
            target_delta=config.get("delta", 1e-5),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            epochs=config.get("local_epochs", 3) * config.get("num_rounds", 50),
            sample_size=len(data_loader.train_dataset),
            batch_size=config.get("batch_size", 32)
        )
    
    # Create client
    client = HospitalClient(
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
    
    return client


def run_simulation(
    config: Dict[str, Any],
    data_dir: str,
    partition: Dict,
    device: torch.device,
    api_url: Optional[str] = None
) -> Dict:
    """
    Run federated learning simulation.
    
    Args:
        config: Configuration dictionary
        data_dir: Path to dataset
        partition: Data partition
        device: Device for computation
        api_url: Optional API URL for dashboard updates
    
    Returns:
        Training history
    """
    num_hospitals = config.get("num_hospitals", 5)
    num_rounds = config.get("num_rounds", 50)
    
    logger.info(f"Starting federated learning simulation")
    logger.info(f"  - Hospitals: {num_hospitals}")
    logger.info(f"  - Rounds: {num_rounds}")
    logger.info(f"  - Device: {device}")
    
    # Create global model for initial parameters
    global_model = create_model(
        model_name=config.get("model_name", "efficientnet_b0"),
        num_classes=config.get("num_classes", 2),
        pretrained=config.get("pretrained", True)
    )
    
    model_info = get_model_info(global_model)
    logger.info(f"Model: {config.get('model_name')} - {model_info['total_parameters']:,} parameters")
    
    # Get initial parameters
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    )
    
    # Create aggregation strategy
    strategy = create_aggregation_strategy(
        strategy_name=config.get("aggregation_strategy", "adaptive"),
        initial_parameters=initial_parameters,
        config={
            "fraction_fit": config.get("fraction_fit", 1.0),
            "fraction_evaluate": config.get("fraction_evaluate", 1.0),
            "min_fit_clients": num_hospitals,
            "min_evaluate_clients": num_hospitals,
            "min_available_clients": num_hospitals,
            "local_epochs": config.get("local_epochs", 3),
            "sample_weight": config.get("sample_weight", 0.3),
            "stability_weight": config.get("stability_weight", 0.25),
            "loss_weight": config.get("loss_weight", 0.25),
            "quality_weight": config.get("quality_weight", 0.2),
            "stability_window": config.get("stability_window", 5),
            "proximal_mu": config.get("proximal_mu", 0.01)
        }
    )
    
    # Metrics collector
    metrics_collector = MetricsCollector(
        experiment_name=f"fl_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_dir=config.get("metrics_dir", "./metrics")
    )
    
    metrics_collector.set_config(
        num_hospitals=num_hospitals,
        num_rounds=num_rounds,
        partition_type=config.get("partition_type", "non_iid"),
        dp_enabled=config.get("dp_enabled", True),
        aggregation_strategy=config.get("aggregation_strategy", "adaptive")
    )
    
    # Client function factory
    def create_client_fn(cid: str):
        return client_fn(cid, data_dir, partition, config, device)
    
    # Dashboard update callback
    if api_url:
        import requests
        
        def send_metrics_to_dashboard(metrics: Dict):
            try:
                requests.post(f"{api_url}/api/metrics/update", json=metrics, timeout=5)
            except Exception as e:
                logger.warning(f"Failed to send metrics to dashboard: {e}")
    
    # Run simulation
    logger.info("Starting Flower simulation...")
    
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn,
        num_clients=num_hospitals,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.2 if device.type == "cuda" else 0}
    )
    
    # Finalize metrics
    metrics_collector.finalize()
    
    logger.info("Simulation completed!")
    logger.info(f"Final metrics saved to: {metrics_collector.save()}")
    
    return {
        "history": history,
        "metrics": metrics_collector.get_summary()
    }


def run_explainability(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_samples: int = 10,
    device: torch.device = torch.device('cpu')
) -> None:
    """
    Generate Grad-CAM explanations for model predictions.
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to dataset
        output_dir: Directory to save explanations
        num_samples: Number of samples to explain
        device: Device for computation
    """
    from src.data import RSNAPneumoniaDataset, get_transforms
    from src.explainability import GradCAMExplainer, save_explanation
    
    logger.info("Generating Grad-CAM explanations...")
    
    # Load model
    model = create_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    
    # Create explainer
    explainer = GradCAMExplainer(model=model, device=device)
    
    # Load some test samples
    dataset = RSNAPneumoniaDataset(
        data_dir=data_dir,
        is_training=False
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate explanations
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        image_tensor, label, patient_id = dataset[idx]
        
        # Get original image (before normalization)
        raw_image = dataset.train_dataset[idx] if hasattr(dataset, 'train_dataset') else None
        
        # Generate report
        report = explainer.generate_report(
            image=np.zeros((224, 224, 3)),  # Placeholder
            input_tensor=image_tensor.unsqueeze(0),
            patient_id=patient_id,
            save_path=str(output_path / f"explanation_{patient_id}.png")
        )
        
        logger.info(f"Generated explanation for {patient_id}: {report['prediction']} ({report['confidence']:.2%})")
    
    logger.info(f"Explanations saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hospital Federated Learning Training"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download the RSNA dataset"
    )
    
    parser.add_argument(
        "--partition-data",
        action="store_true",
        help="Create data partitions for hospitals"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run federated training"
    )
    
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate Grad-CAM explanations"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint for explanations"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Dashboard API URL for live updates"
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Override number of training rounds"
    )
    
    parser.add_argument(
        "--num-hospitals",
        type=int,
        default=None,
        help="Override number of hospitals"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.num_rounds:
        config["federated"]["num_rounds"] = args.num_rounds
    if args.num_hospitals:
        config["federated"]["num_hospitals"] = args.num_hospitals
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Data directory
    data_dir = config["dataset"]["data_dir"]
    
    # Download data
    if args.download_data:
        logger.info("Downloading RSNA Pneumonia Detection dataset...")
        download_rsna_dataset(
            data_dir=data_dir,
            kaggle_json_path="./kaggle.json"
        )
    
    # Create partitions
    if args.partition_data:
        logger.info("Creating hospital data partitions...")
        partition = create_hospital_partitions(
            data_dir=data_dir,
            num_hospitals=config["federated"]["num_hospitals"],
            partition_type=config["data_distribution"]["type"],
            dirichlet_alpha=config["data_distribution"]["dirichlet_alpha"],
            seed=config["seed"]
        )
        logger.info("Partitions created successfully!")
    
    # Run training
    if args.train:
        # Load partition
        partition_file = Path(data_dir) / f"partition_{config['data_distribution']['type']}_{config['federated']['num_hospitals']}hospitals.json"
        
        if not partition_file.exists():
            logger.error(f"Partition file not found: {partition_file}")
            logger.info("Run with --partition-data first to create partitions")
            sys.exit(1)
        
        with open(partition_file, 'r') as f:
            partition_data = json.load(f)
        
        partition = {int(k): v for k, v in partition_data['partitions'].items()}
        
        # Flatten config for training
        train_config = {
            **config["federated"],
            **config["training"],
            **config["model"],
            **config["privacy"],
            **config["aggregation"],
            "num_classes": config["dataset"]["num_classes"],
            "image_size": config["dataset"]["image_size"],
            "metrics_dir": config["logging"]["save_dir"],
            "dp_enabled": config["privacy"]["enabled"]
        }
        
        # Run simulation
        results = run_simulation(
            config=train_config,
            data_dir=data_dir,
            partition=partition,
            device=device,
            api_url=args.api_url
        )
        
        logger.info("Training completed!")
        logger.info(f"Summary: {results['metrics']}")
    
    # Generate explanations
    if args.explain:
        run_explainability(
            model_path=args.model_path,
            data_dir=data_dir,
            output_dir="./explanations",
            num_samples=config["explainability"].get("num_samples_to_explain", 10),
            device=device
        )


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    logger.add(
        "logs/training_{time}.log",
        rotation="100 MB",
        level="DEBUG"
    )
    
    main()

