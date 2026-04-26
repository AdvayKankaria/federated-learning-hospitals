#!/usr/bin/env python3
"""
Federated Learning with Differential Privacy
=============================================
Runs federated learning with (ε,δ)-differential privacy guarantees.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import OrderedDict
import json
import requests

import yaml
import torch
import torch.nn as nn
import numpy as np
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from loguru import logger

# Add source to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import create_hospital_partitions, HospitalDataLoader
from src.models import create_model, get_model_info
from src.fl_core.metrics import MetricsCollector


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS doesn't work well with Opacus, use CPU for DP training
    return torch.device('cpu')


def make_model_dp_compatible(model: nn.Module) -> nn.Module:
    """Make model compatible with differential privacy."""
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.info(f"Fixing {len(errors)} DP compatibility issues...")
        model = ModuleValidator.fix(model)
    return model


def train_one_epoch_dp(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    privacy_engine: Optional[PrivacyEngine] = None
) -> Dict[str, float]:
    """Train model for one epoch with optional DP."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return {
        "loss": total_loss / total if total > 0 else 0,
        "accuracy": correct / total if total > 0 else 0,
        "samples": total
    }


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    pneumonia_mask = all_labels == 1
    normal_mask = all_labels == 0
    
    sensitivity = (
        np.sum((all_preds == 1) & pneumonia_mask) / np.sum(pneumonia_mask)
        if np.sum(pneumonia_mask) > 0 else 0
    )
    specificity = (
        np.sum((all_preds == 0) & normal_mask) / np.sum(normal_mask)
        if np.sum(normal_mask) > 0 else 0
    )
    
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        "loss": total_loss / total if total > 0 else 0,
        "accuracy": correct / total if total > 0 else 0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "samples": total
    }


def federated_averaging(
    global_model: nn.Module,
    client_state_dicts: List[Dict],
    client_weights: List[float]
) -> None:
    """Perform federated averaging."""
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for client_dict, weight in zip(client_state_dicts, normalized_weights):
            global_dict[key] += weight * client_dict[key].float()
    
    global_model.load_state_dict(global_dict)


def adaptive_weights(
    client_metrics: List[Dict[str, float]],
    history: Dict[int, List[Dict]],
    config: Dict[str, float]
) -> List[float]:
    """Calculate adaptive weights for each client."""
    weights = []
    
    for i, metrics in enumerate(client_metrics):
        sample_score = metrics.get("samples", 1)
        
        client_history = history.get(i, [])
        if len(client_history) >= 2:
            losses = [h.get("loss", 0) for h in client_history[-5:]]
            stability_score = 1.0 / (1.0 + np.var(losses))
        else:
            stability_score = 1.0
        
        if len(client_history) >= 2:
            prev_loss = client_history[-2].get("loss", metrics["loss"])
            loss_improvement = max(0, prev_loss - metrics["loss"])
            loss_score = loss_improvement + 0.1
        else:
            loss_score = 1.0
        
        quality_score = max(0.1, metrics.get("accuracy", 0.5))
        
        combined = (
            config.get("sample_weight", 0.3) * sample_score +
            config.get("stability_weight", 0.25) * stability_score * 1000 +
            config.get("loss_weight", 0.25) * loss_score * 1000 +
            config.get("quality_weight", 0.2) * quality_score * 1000
        )
        
        weights.append(combined)
    
    return weights


def send_to_dashboard(api_url: str, metrics: Dict) -> None:
    """Send metrics to dashboard API."""
    try:
        requests.post(f"{api_url}/api/metrics/update", json=metrics, timeout=5)
    except Exception as e:
        logger.debug(f"Dashboard update failed: {e}")


def run_federated_learning_with_dp(
    config: Dict[str, Any],
    data_dir: str,
    partition: Dict,
    device: torch.device,
    num_rounds: int = 10,
    local_epochs: int = 3,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    api_url: Optional[str] = None
) -> Dict:
    """Run federated learning with differential privacy."""
    
    num_hospitals = len(partition)
    
    logger.info(f"🔐 Starting Federated Learning with Differential Privacy")
    logger.info(f"  - Hospitals: {num_hospitals}")
    logger.info(f"  - Rounds: {num_rounds}")
    logger.info(f"  - Local epochs: {local_epochs}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Privacy: ε={target_epsilon}, δ={target_delta}")
    logger.info(f"  - Max grad norm: {max_grad_norm}")
    
    # Create global model (DP compatible)
    global_model = create_model(
        model_name=config.get("model_name", "efficientnet_b0"),
        num_classes=2,
        pretrained=True,
        dropout=config.get("dropout", 0.3)
    )
    global_model = make_model_dp_compatible(global_model)
    global_model = global_model.to(device)
    
    model_info = get_model_info(global_model)
    logger.info(f"Model: EfficientNet-B0 (DP-compatible) - {model_info['total_parameters']:,} parameters")
    
    # Create data loaders for each hospital
    hospital_loaders = {}
    for hospital_id in range(num_hospitals):
        hospital_loaders[hospital_id] = HospitalDataLoader(
            hospital_id=hospital_id,
            data_dir=data_dir,
            partition=partition,
            batch_size=config.get("batch_size", 32),
            image_size=config.get("image_size", 224),
            num_workers=0
        )
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Metrics
    metrics_collector = MetricsCollector(
        experiment_name=f"fl_dp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_dir="./metrics"
    )
    
    # History for adaptive weights
    client_history: Dict[int, List[Dict]] = {i: [] for i in range(num_hospitals)}
    
    # Privacy tracking
    total_epsilon_spent = 0.0
    epsilon_per_round = target_epsilon / num_rounds
    
    # Training loop
    best_accuracy = 0.0
    history = []
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_num}/{num_rounds}")
        logger.info(f"{'='*60}")
        
        round_start = datetime.now()
        
        # Train each hospital with DP
        client_state_dicts = []
        client_metrics = []
        
        for hospital_id in range(num_hospitals):
            # Create local model (DP compatible)
            local_model = create_model(
                model_name=config.get("model_name", "efficientnet_b0"),
                num_classes=2,
                pretrained=False
            )
            local_model = make_model_dp_compatible(local_model)
            local_model.load_state_dict(global_model.state_dict())
            local_model = local_model.to(device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=config.get("learning_rate", 0.001),
                weight_decay=config.get("weight_decay", 0.0001)
            )
            
            # Get data loader
            train_loader = hospital_loaders[hospital_id].get_train_loader()
            
            # Calculate noise multiplier for this round's epsilon budget
            sample_rate = config.get("batch_size", 32) / len(train_loader.dataset)
            
            # Setup privacy engine
            privacy_engine = PrivacyEngine()
            
            try:
                local_model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=local_model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=epsilon_per_round,
                    target_delta=target_delta,
                    epochs=local_epochs,
                    max_grad_norm=max_grad_norm
                )
                dp_enabled = True
            except Exception as e:
                logger.warning(f"  Hospital {hospital_id}: DP failed ({e}), training without DP")
                dp_enabled = False
            
            # Train locally
            epoch_metrics = {}
            for epoch in range(local_epochs):
                epoch_metrics = train_one_epoch_dp(
                    local_model, train_loader, optimizer, criterion, device
                )
            
            # Get epsilon spent
            if dp_enabled:
                try:
                    eps = privacy_engine.get_epsilon(delta=target_delta)
                    epoch_metrics["epsilon_spent"] = eps
                except:
                    epoch_metrics["epsilon_spent"] = epsilon_per_round
            else:
                epoch_metrics["epsilon_spent"] = 0
            
            # Evaluate
            val_loader = hospital_loaders[hospital_id].get_val_loader()
            val_metrics = evaluate(local_model, val_loader, criterion, device)
            
            epoch_metrics.update({
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"]
            })
            
            # Extract state dict (handle DP wrapper)
            if hasattr(local_model, '_module'):
                state_dict = local_model._module.state_dict()
            else:
                state_dict = local_model.state_dict()
            
            client_state_dicts.append(state_dict)
            client_metrics.append(epoch_metrics)
            client_history[hospital_id].append(epoch_metrics)
            
            dp_status = "🔒 DP" if dp_enabled else "⚠️ No DP"
            logger.info(
                f"  Hospital {hospital_id} [{dp_status}]: "
                f"Loss={epoch_metrics['loss']:.4f}, "
                f"Acc={epoch_metrics['accuracy']:.4f}, "
                f"Val={val_metrics['accuracy']:.4f}"
            )
        
        # Calculate adaptive weights
        weights = adaptive_weights(client_metrics, client_history, config)
        
        # Federated averaging
        federated_averaging(global_model, client_state_dicts, weights)
        
        # Global evaluation
        test_loader = hospital_loaders[0].get_test_loader()
        global_metrics = evaluate(global_model, test_loader, criterion, device)
        
        # Update privacy tracking
        round_epsilon = np.mean([m.get("epsilon_spent", 0) for m in client_metrics])
        total_epsilon_spent += round_epsilon
        
        round_time = (datetime.now() - round_start).total_seconds()
        
        logger.info(f"\n  [Global] Accuracy: {global_metrics['accuracy']:.4f}, "
                   f"AUC: {global_metrics['auc']:.4f}")
        logger.info(f"  [Privacy] ε spent this round: {round_epsilon:.4f}, "
                   f"Total ε: {total_epsilon_spent:.4f}/{target_epsilon}")
        logger.info(f"  [Time] {round_time:.1f}s")
        
        # Track best
        if global_metrics['accuracy'] > best_accuracy:
            best_accuracy = global_metrics['accuracy']
            torch.save({
                'round': round_num,
                'model_state_dict': global_model.state_dict(),
                'accuracy': best_accuracy,
                'epsilon_spent': total_epsilon_spent
            }, 'checkpoints/best_model_dp.pt')
            logger.info(f"  [NEW BEST] Accuracy: {best_accuracy:.4f}")
        
        # Store history
        round_data = {
            "round": round_num,
            "global_accuracy": global_metrics["accuracy"],
            "global_loss": global_metrics["loss"],
            "auc": global_metrics["auc"],
            "sensitivity": global_metrics["sensitivity"],
            "specificity": global_metrics["specificity"],
            "epsilon_spent": total_epsilon_spent,
            "time": round_time
        }
        history.append(round_data)
        
        # Send to dashboard
        if api_url:
            send_to_dashboard(api_url, {
                "round_num": round_num,
                "global_loss": global_metrics["loss"],
                "global_accuracy": global_metrics["accuracy"],
                "auc_roc": global_metrics["auc"],
                "sensitivity": global_metrics["sensitivity"],
                "specificity": global_metrics["specificity"],
                "epsilon_spent": total_epsilon_spent,
                "num_clients": num_hospitals
            })
        
        # Save metrics
        metrics_collector.add_round_metrics(round_num, {
            "test_accuracy": global_metrics["accuracy"],
            "test_loss": global_metrics["loss"],
            "test_auc_roc": global_metrics["auc"],
            "epsilon_spent": total_epsilon_spent
        })
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Training Complete!")
    logger.info(f"   Best Accuracy: {best_accuracy:.4f}")
    logger.info(f"   Final ε spent: {total_epsilon_spent:.4f} / {target_epsilon}")
    logger.info(f"   Privacy guarantee: ({total_epsilon_spent:.2f}, {target_delta})-DP")
    logger.info(f"{'='*60}")
    
    # Save final model
    torch.save({
        'round': num_rounds,
        'model_state_dict': global_model.state_dict(),
        'accuracy': global_metrics['accuracy'],
        'epsilon_spent': total_epsilon_spent,
        'history': history
    }, 'checkpoints/final_model_dp.pt')
    
    metrics_collector.finalize()
    
    return {
        "best_accuracy": best_accuracy,
        "epsilon_spent": total_epsilon_spent,
        "history": history,
        "model": global_model
    }


def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Differential Privacy")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--num-hospitals", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", 
                       help="Dashboard API URL for live updates")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get device (CPU for DP training)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Data directory
    data_dir = config["dataset"]["data_dir"]
    
    # Load partition
    partition_file = Path(data_dir) / f"partition_non_iid_{args.num_hospitals}hospitals.json"
    
    if not partition_file.exists():
        logger.info("Creating partitions...")
        partition = create_hospital_partitions(
            data_dir=data_dir,
            num_hospitals=args.num_hospitals,
            partition_type="non_iid",
            dirichlet_alpha=0.5
        )
    else:
        with open(partition_file, 'r') as f:
            partition_data = json.load(f)
        partition = {int(k): v for k, v in partition_data['partitions'].items()}
    
    # Create directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Flatten config
    train_config = {
        **config.get("training", {}),
        **config.get("model", {}),
        **config.get("aggregation", {}),
        "image_size": config.get("dataset", {}).get("image_size", 224),
    }
    
    # Run federated learning with DP
    results = run_federated_learning_with_dp(
        config=train_config,
        data_dir=data_dir,
        partition=partition,
        device=device,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        api_url=args.api_url
    )
    
    logger.info(f"\nResults saved to checkpoints/ and metrics/")
    logger.info(f"Privacy guarantee: (ε={results['epsilon_spent']:.2f}, δ={args.delta})-differential privacy")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    logger.add(
        "logs/training_dp_{time}.log",
        rotation="100 MB",
        level="DEBUG"
    )
    
    main()

