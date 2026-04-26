# Federated Learning Core
# =======================
# Core components for federated learning with Flower

from .client import HospitalClient, create_client
from .server import FederatedServer, create_server
from .aggregation import (
    AdaptiveAggregation,
    FedAvgStrategy,
    FedProxStrategy,
    create_aggregation_strategy
)
from .privacy import DifferentialPrivacyEngine, apply_dp_to_model
from .metrics import MetricsCollector, FederatedMetrics

__all__ = [
    "HospitalClient",
    "create_client",
    "FederatedServer",
    "create_server",
    "AdaptiveAggregation",
    "FedAvgStrategy",
    "FedProxStrategy",
    "create_aggregation_strategy",
    "DifferentialPrivacyEngine",
    "apply_dp_to_model",
    "MetricsCollector",
    "FederatedMetrics"
]

