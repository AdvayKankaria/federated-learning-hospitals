# Person 1 Script - Introduction and Architecture

## Suggested Duration
3 to 4 minutes

## Opening

Good morning/afternoon everyone.  
We are presenting our project: **Hospital Federated Learning for Pneumonia Detection**.

In healthcare, data is highly sensitive and distributed across hospitals.  
If we collect all data in one place, privacy and compliance become major challenges.  
So our goal is to train a strong global AI model **without sharing raw patient data**.

That is why we use:
- Federated Learning
- Differential Privacy
- Adaptive Aggregation

Today, we will explain:
1. System architecture  
2. Core implementation  
3. End-to-end demo and outputs

---

## Problem Statement

Traditional centralized ML requires all data in one server.  
Our project avoids that.

Each hospital trains locally on its own X-ray data, then sends only model updates to a central server.  
The server aggregates updates and sends back an improved global model.  
This process repeats for multiple rounds.

---

## Architecture Diagram Walkthrough

Please look at the architecture diagram.

### Step 1 - Data and Partitioning
We start with the RSNA pneumonia dataset.  
Then we partition data into multiple hospital clients, usually in a non-IID way to simulate real-world differences.

### Step 2 - Hospital Clients
Each hospital has:
- local train, validation, and test data,
- its own local model training loop.

No raw data leaves the hospital node.

### Step 3 - Federated Server
The central server:
- sends current global model weights to hospitals,
- receives updated local weights and metrics,
- applies aggregation strategy,
- produces the next global model.

### Step 4 - Adaptive Aggregation
Instead of simple averaging, we use adaptive weighting.  
Each client contribution is weighted based on:
- number of samples,
- stability,
- loss improvement,
- quality score.

### Step 5 - Privacy and Monitoring
For DP-enabled runs, local updates are privacy-protected before sharing.  
All round-level metrics are stored and can be visualized in dashboard.

### Step 6 - Outputs
Final outputs include:
- trained checkpoints,
- metrics JSON,
- training logs,
- Grad-CAM explanation images.

---

## Key Design Benefits

Our architecture provides:
- privacy-preserving collaborative learning,
- better realism with non-IID hospitals,
- flexible aggregation and privacy settings,
- complete traceability through metrics and logs.

Now I will hand over to Person 2, who will explain the implementation details in code.

