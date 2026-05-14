# Person 2 Script - Code and Implementation

## Suggested Duration
4 to 5 minutes

## Transition

Thanks. I will now explain how this architecture is implemented in the codebase.

---

## Code Structure Overview

The project is organized into these main areas:

- `src/data` for dataset handling and hospital partitioning
- `src/fl_core` for federated client, aggregation, and metrics
- `train.py` for Flower-based federated training
- `train_simple.py` for easy-to-understand manual FL
- `train_dp.py` for manual FL with differential privacy

---

## Data Pipeline

In the data layer:
- We prepare and load the RSNA dataset.
- We create per-hospital partitions using non-IID logic.
- `HospitalDataLoader` provides each hospital's train/val/test loaders.

So each client receives only its own partition and trains locally.

---

## Model Setup

The model is created through model factory utilities.  
By default we use **EfficientNet-B0** with 2 output classes:
- Normal
- Pneumonia

Input image size is configured from YAML, typically 224.

---

## Federated Training in `train.py`

`train.py` is the main orchestration script.

Main flow:
1. Load config
2. Prepare partition mapping
3. Build initial global model
4. Build aggregation strategy
5. Start Flower simulation

The key object is `HospitalClient`, which handles:
- `fit()` for local training
- `evaluate()` for local evaluation
- parameter exchange with server

---

## Adaptive Aggregation

Inside `src/fl_core/aggregation.py`, adaptive strategy computes client weights from four signals:

1. Sample count score
2. Stability score
3. Loss improvement score
4. Quality score

These are combined using configured coefficients:
- sample weight
- stability weight
- loss weight
- quality weight

Then client weights are normalized and used for weighted averaging of model parameters.

---

## Differential Privacy Integration

There are two DP paths:

1. `train_dp.py` uses Opacus directly:
   - DP-compatible model checks
   - gradient clipping
   - noise addition
   - epsilon tracking

2. `train.py` can use a custom DP engine in the FL client path.

Privacy parameters come from config:
- epsilon
- delta
- max gradient norm

---

## Metrics and Outputs

`src/fl_core/metrics.py` stores round-wise metrics including:
- global loss and accuracy
- AUC, sensitivity, specificity
- client participation
- epsilon spent for DP runs

Artifacts produced:
- checkpoint files in `checkpoints/`
- metrics JSON in `metrics/`
- logs in `logs/`
- explanations in `explanations/`

Now I will hand over to Person 3 for the live demo and result walkthrough.

