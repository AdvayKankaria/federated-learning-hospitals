# Hospital Federated Learning
## Design and Architecture Document

## 1. Introduction

### 1.1 Purpose
This document describes the design and architecture of the **Hospital Federated Learning for Pneumonia Detection** system.  
The platform enables multiple hospitals to collaboratively train a chest X-ray classifier while keeping patient data local to each hospital.

### 1.2 Goals
- Build a global pneumonia detection model without centralizing raw medical data.
- Preserve privacy using federated learning and differential privacy.
- Improve training robustness with adaptive aggregation.
- Provide transparent outputs via metrics and explainability artifacts.

### 1.3 Scope
- Federated training simulation with multiple hospital clients.
- Local model training and evaluation at each client.
- Centralized orchestration and aggregation.
- Differential privacy support.
- Metrics tracking, checkpointing, and optional dashboard integration.
- Grad-CAM explainability output generation.

---

## 2. System Overview

### 2.1 Problem Context
Healthcare data is sensitive and distributed across institutions. Moving all data to one central server is often not possible due to privacy and compliance constraints. This system addresses that by moving the model to the data, not the data to the model.

### 2.2 High-Level Architecture

```mermaid
flowchart LR
    DS[RSNA Dataset] --> PART[Partitioning Engine]
    PART --> H1[Hospital Client 1]
    PART --> H2[Hospital Client 2]
    PART --> HN[Hospital Client N]

    S[Federated Server] -->|Global Weights| H1
    S -->|Global Weights| H2
    S -->|Global Weights| HN

    H1 -->|Local Updates + Metrics| S
    H2 -->|Local Updates + Metrics| S
    HN -->|Local Updates + Metrics| S

    S --> AGG[Adaptive Aggregation]
    AGG --> GM[Updated Global Model]

    GM --> MET[Metrics Store]
    GM --> CKPT[Model Checkpoints]
    GM --> EXPL[Grad-CAM Explanations]
    MET --> DASH[Dashboard]
```

---

## 3. Architectural Layers

```mermaid
flowchart TB
    subgraph L1[Presentation and Monitoring Layer]
        UI[Dashboard UI]
        API[Metrics API]
    end

    subgraph L2[Orchestration Layer]
        ORCH[Flower Server Orchestrator]
        ROUND[Round Controller]
        SELECT[Client Selection]
    end

    subgraph L3[Federated Intelligence Layer]
        STRAT[Aggregation Strategies]
        EVAL[Global Evaluation]
        PRIVT[Privacy Tracking]
    end

    subgraph L4[Client Training Layer]
        CLIENT[Hospital Client Runtime]
        TRAIN[Local Train and Validate]
        DP[Differential Privacy Engine]
    end

    subgraph L5[Data and Model Layer]
        DATA[Hospital Data Loaders]
        MODEL[Model Factory]
        META[Partition Metadata]
    end

    subgraph L6[Persistence Layer]
        MJSON[Metrics JSON]
        CPK[Checkpoint Files]
        LOGS[Training Logs]
        PNG[Explanation Images]
    end

    UI --> API --> MJSON
    ORCH --> ROUND --> SELECT
    ORCH --> STRAT --> EVAL
    CLIENT --> TRAIN --> DP
    CLIENT --> DATA
    TRAIN --> MODEL
    DATA --> META
    EVAL --> MJSON
    EVAL --> CPK
    EVAL --> LOGS
    EVAL --> PNG
```

---

## 4. Core Components

### 4.1 Data Layer
- Downloads and prepares the RSNA Pneumonia dataset.
- Splits data into hospital-wise partitions (supports non-IID distribution).
- Provides train, validation, and test loaders per hospital.

### 4.2 Model Layer
- Uses EfficientNet-B0 as the default classifier.
- Supports binary output: Normal vs Pneumonia.
- Model weights are exchanged between server and clients per training round.

### 4.3 Federated Client Layer
- Each hospital receives global weights.
- Trains locally on private hospital partition.
- Sends only model updates and metrics to the server.

### 4.4 Federated Server Layer
- Coordinates rounds and client participation.
- Aggregates client updates (FedAvg / FedProx / Adaptive).
- Produces and distributes updated global model.

### 4.5 Privacy Layer
- Differential privacy is applied at local training level.
- Uses clipping and noise addition to protect individual sample contributions.
- Tracks epsilon usage over training.

### 4.6 Monitoring and Explainability Layer
- Collects round-wise metrics (loss, accuracy, AUC, sensitivity, specificity).
- Stores checkpoints and experiment logs.
- Generates Grad-CAM explanation images.

---

## 5. Federated Training Workflow

```mermaid
sequenceDiagram
    participant Server as Federated Server
    participant C1 as Hospital 1
    participant C2 as Hospital 2
    participant CN as Hospital N

    loop Global rounds (1..R)
        Server->>C1: Send global model
        Server->>C2: Send global model
        Server->>CN: Send global model

        C1->>C1: Local training (E epochs)
        C2->>C2: Local training (E epochs)
        CN->>CN: Local training (E epochs)

        C1-->>Server: Updated weights + metrics
        C2-->>Server: Updated weights + metrics
        CN-->>Server: Updated weights + metrics

        Server->>Server: Compute client weights
        Server->>Server: Aggregate updates
        Server->>Server: Evaluate global model
        Server->>Server: Save metrics and checkpoints
    end
```

---

## 6. Differential Privacy Design

### 6.1 Privacy Objective
Ensure that a single patient record has limited impact on model updates.

### 6.2 DP Mechanism
- Per-sample gradient clipping.
- Gaussian noise injection.
- Privacy accounting with \((\epsilon, \delta)\)-DP.

```mermaid
flowchart LR
    BATCH[Local Batch] --> FWD[Forward Pass and Loss]
    FWD --> GRAD[Per-sample Gradients]
    GRAD --> CLIP[Clip Gradients by max_grad_norm]
    CLIP --> NOISE[Add Gaussian Noise]
    NOISE --> STEP[Optimizer Step]
    STEP --> UPDATE[Private Local Update]
    UPDATE --> SEND[Send update to server]
```

### 6.3 Privacy Controls
- **epsilon**: privacy budget.
- **delta**: failure probability.
- **max_grad_norm**: clipping threshold.

---

## 7. Adaptive Aggregation Design

Client update weights are computed from:
- Sample contribution.
- Stability (loss variance trend).
- Loss improvement.
- Quality score (accuracy behavior).

```mermaid
flowchart TB
    SAMP[Sample Score] --> COMB[Weight Combiner]
    STAB[Stability Score] --> COMB
    LOSS[Loss Improvement Score] --> COMB
    QUAL[Quality Score] --> COMB
    COMB --> NORM[Normalize Client Weights]
    NORM --> WAVG[Weighted Parameter Averaging]
    WAVG --> NEWG[New Global Model]
```

This strategy improves robustness in heterogeneous (non-IID) hospital settings.

---

## 8. Deployment Architecture (Current Project)

```mermaid
flowchart LR
    DEV[Developer Machine] --> TRAIN[Training Runtime]
    TRAIN --> FL[Flower Simulation]
    TRAIN --> CL[Simulated Hospital Clients]
    TRAIN --> FS[(Local File Storage)]

    FS --> METRICS[metrics/*.json]
    FS --> CHECKS[checkpoints/*.pt]
    FS --> LOGFILE[logs/*.log]
    FS --> EXPLAIN[explanations/*.png]

    TRAIN --> BACKEND[Dashboard Backend]
    BACKEND --> FRONTEND[Dashboard Frontend]
```

---

## 9. Non-Functional Architecture Considerations

### 9.1 Privacy and Security
- No raw medical image transfer to the server.
- Only model parameters and aggregated metrics are exchanged.
- Differential privacy reduces patient-level leakage risk.

### 9.2 Scalability
- Current setup supports simulation on a single machine.
- Architecture can be extended to distributed real hospital nodes.

### 9.3 Reliability
- Supports configurable minimum participating clients.
- Handles strategy-level control for client failures.

### 9.4 Observability
- Round-level metrics, logs, and checkpoints are persisted.
- Optional dashboard supports live monitoring.

---

## 10. Design Summary

This architecture provides:
- Privacy-preserving collaborative learning across hospitals.
- Flexible aggregation strategy with adaptive weighting.
- Differential privacy integration for stronger confidentiality guarantees.
- End-to-end traceability through metrics, logs, checkpoints, and explanations.

It is a strong base for academic demonstration and can be extended toward production-grade distributed federated deployment.

