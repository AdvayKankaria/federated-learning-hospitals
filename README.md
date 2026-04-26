# 🏥 Hospital Federated Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flower](https://img.shields.io/badge/Flower-1.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Privacy-Preserving Federated Learning for Medical Imaging**

*Train powerful AI models across hospitals without sharing sensitive patient data*

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Dashboard](#-dashboard) • [Documentation](#-documentation)

</div>

---

## 🌟 Overview

This project implements a complete **Federated Learning** system for pneumonia detection from chest X-rays, designed for multi-hospital collaboration while maintaining strict patient privacy.

### Key Innovations

| Feature | Description |
|---------|-------------|
| 🔐 **Differential Privacy** | (ε,δ)-DP guarantees using Opacus |
| ⚖️ **Adaptive Aggregation** | Smart weighting based on data quality, stability, and sample size |
| 🧠 **EfficientNet-B0** | State-of-the-art transfer learning for medical imaging |
| 📊 **Real-time Dashboard** | Beautiful web UI for monitoring training |
| 🔍 **Grad-CAM Explainability** | Visualize what the model sees |
| 🐳 **Docker Ready** | One-command deployment |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU training)
- Docker & Docker Compose (for containerized deployment)
- Kaggle account (for dataset download)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/hospital-fed-learning.git
cd hospital-fed-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Kaggle Credentials

Place your `kaggle.json` in the project root or configure environment variables:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### Download Dataset & Train

```bash
# Step 1: Download RSNA Pneumonia Detection dataset (~1.3GB)
python train.py --download-data

# Step 2: Create Non-IID partitions for 5 hospitals
python train.py --partition-data

# Step 3: Run federated training
python train.py --train

# (Optional) Generate Grad-CAM explanations
python train.py --explain --model-path checkpoints/best_model.pt
```

### Quick Docker Deployment

```bash
# Build and run everything
docker-compose up -d

# Access dashboard at http://localhost:3000
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│    │Hospital 1│    │Hospital 2│    │Hospital 3│    ...          │
│    │  (Local) │    │  (Local) │    │  (Local) │                │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘                │
│         │               │               │                        │
│         │  DP Noise     │  DP Noise     │  DP Noise            │
│         ▼               ▼               ▼                        │
│    ┌─────────────────────────────────────────────────────┐      │
│    │              ADAPTIVE AGGREGATION                    │      │
│    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │      │
│    │  │ Sample  │ │Stability│ │  Loss   │ │ Quality │   │      │
│    │  │ Weight  │ │ Weight  │ │ Weight  │ │ Weight  │   │      │
│    │  │  30%    │ │  25%    │ │  25%    │ │  20%    │   │      │
│    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │      │
│    └─────────────────────────────────────────────────────┘      │
│                            │                                     │
│                            ▼                                     │
│                   ┌────────────────┐                            │
│                   │  GLOBAL MODEL  │                            │
│                   │ (EfficientNet) │                            │
│                   └────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| **FL Framework** | Flower | Orchestrates federated training |
| **Model** | EfficientNet-B0 (timm) | Feature extraction & classification |
| **Privacy** | Opacus | Differential privacy guarantees |
| **Aggregation** | Custom Adaptive | Intelligent weight combination |
| **Backend** | FastAPI | REST API & WebSocket server |
| **Frontend** | React + Recharts | Real-time dashboard |
| **Deployment** | Docker Compose | Container orchestration |

---

## 📊 Dashboard

The web dashboard provides real-time monitoring of federated training:

### Features

- **Live Training Metrics**: Accuracy, Loss, AUC-ROC updated in real-time
- **Hospital Status**: Per-hospital statistics and contribution
- **Privacy Budget**: Visual tracking of ε consumption
- **Adaptive Weights**: See how each hospital's contribution is weighted
- **Grad-CAM Viewer**: Explainability visualizations

### Access

```bash
# Start the dashboard
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (development)
cd dashboard/frontend
npm install
npm start  # Opens at http://localhost:3000

# Or use Docker
docker-compose up dashboard
```

---

## 🔐 Differential Privacy

This implementation provides formal (ε,δ)-differential privacy guarantees:

### Configuration

```yaml
# config/config.yaml
privacy:
  enabled: true
  epsilon: 8.0        # Privacy budget (lower = more private)
  delta: 1e-5         # Failure probability
  max_grad_norm: 1.0  # Gradient clipping threshold
```

### Privacy Mechanisms

1. **Gradient Clipping**: Bounds sensitivity per sample
2. **Gaussian Noise**: Calibrated noise addition
3. **Privacy Accounting**: RDP accountant tracks budget consumption

### Privacy Levels

| Epsilon | Privacy Level | Use Case |
|---------|--------------|----------|
| ε ≤ 1 | Strong | Maximum privacy, lower accuracy |
| ε = 8 | Moderate | Balanced privacy-utility |
| ε ≥ 10 | Relaxed | Higher accuracy, weaker guarantees |

---

## ⚖️ Adaptive Aggregation

Unlike standard FedAvg, our adaptive strategy weights hospitals intelligently:

### Weighting Factors

```python
final_weight = (
    0.30 * sample_weight +      # More data = higher weight
    0.25 * stability_weight +   # Consistent updates = higher weight
    0.25 * loss_weight +        # Better improvement = higher weight
    0.20 * quality_weight       # Higher accuracy = higher weight
)
```

### Benefits

- Handles **Non-IID data** better than FedAvg
- Reduces impact of **low-quality updates**
- Automatically adapts to **heterogeneous hospital capabilities**

---

## 📁 Project Structure

```
hospital-fed-learning/
├── config/
│   └── config.yaml           # Main configuration
├── src/
│   ├── data/
│   │   ├── download.py       # Dataset downloader
│   │   ├── dataset.py        # PyTorch Dataset
│   │   └── partition.py      # Non-IID partitioning
│   ├── models/
│   │   ├── efficientnet.py   # Model architecture
│   │   └── utils.py          # Model utilities
│   ├── fl_core/
│   │   ├── client.py         # Flower client
│   │   ├── server.py         # Flower server
│   │   ├── aggregation.py    # Adaptive aggregation
│   │   ├── privacy.py        # DP engine
│   │   └── metrics.py        # Metrics collection
│   └── explainability/
│       ├── gradcam.py        # Grad-CAM implementation
│       └── utils.py          # Visualization helpers
├── dashboard/
│   ├── backend/
│   │   └── main.py           # FastAPI server
│   └── frontend/
│       └── src/App.js        # React dashboard
├── docker/
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   └── Dockerfile.dashboard
├── docker-compose.yml
├── train.py                  # Main training script
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

All settings are in `config/config.yaml`:

### Key Settings

```yaml
# Federated Learning
federated:
  num_hospitals: 5
  num_rounds: 50
  fraction_fit: 1.0

# Training
training:
  local_epochs: 3
  batch_size: 32
  learning_rate: 0.001

# Privacy
privacy:
  enabled: true
  epsilon: 8.0
  delta: 1e-5

# Data Distribution
data_distribution:
  type: "non_iid"
  dirichlet_alpha: 0.5  # Lower = more heterogeneous

# Aggregation
aggregation:
  strategy: "adaptive"
  sample_weight: 0.3
  stability_weight: 0.25
  loss_weight: 0.25
  quality_weight: 0.2
```

---

## 📈 Results & Metrics

### Expected Performance

| Metric | Value (50 rounds) |
|--------|-------------------|
| Global Accuracy | ~85-90% |
| AUC-ROC | ~0.90-0.95 |
| Sensitivity | ~85% |
| Specificity | ~88% |
| Privacy Budget | ε = 8.0 |

### Metrics Tracked

- Per-round global accuracy, loss, AUC-ROC
- Per-hospital training metrics
- Privacy budget consumption
- Aggregation weights history
- Convergence statistics

---

## 🐳 Docker Deployment

### Full Stack Deployment

```bash
# Build all images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f fl-server

# Stop services
docker-compose down
```

### Individual Services

```bash
# Dashboard only
docker-compose up dashboard

# Server only
docker-compose up fl-server

# Scale hospitals
docker-compose up --scale hospital-1=1 --scale hospital-2=1
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test with coverage
pytest --cov=src tests/

# Lint code
flake8 src/
```

---

## 📚 Documentation

### API Reference

The FastAPI backend provides interactive documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current experiment status |
| `/api/metrics/history` | GET | All round metrics |
| `/api/hospitals` | GET | Hospital statuses |
| `/api/privacy` | GET | Privacy budget info |
| `/api/experiment/start` | POST | Start training |
| `/ws` | WebSocket | Real-time updates |

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Flower](https://flower.dev/) - Federated Learning Framework
- [Opacus](https://opacus.ai/) - Differential Privacy Library
- [RSNA](https://www.rsna.org/) - Pneumonia Detection Dataset
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models

---

<div align="center">

**Built with ❤️ for Privacy-Preserving Healthcare AI**

</div>

