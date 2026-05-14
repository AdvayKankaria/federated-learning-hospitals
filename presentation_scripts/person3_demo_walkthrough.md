# Person 3 Script - Live Demo and Results

## Suggested Duration
3 to 4 minutes

## Transition

Thanks. I will now demonstrate how the project runs end-to-end and show generated outputs.

---

## Demo Plan

In this demo, I will show:
1. How to run training
2. Where outputs are generated
3. How to interpret results

---

## Step 1 - Run Training

First, we go to the project folder and run training.

Example command:

```bash
python train.py --train --config config/config.yaml --num-rounds 3 --num-hospitals 5
```

For a quick classroom demo, we keep rounds small, like 2 or 3, so execution is fast.

As training runs, we can see:
- each round number
- client participation
- aggregated metrics

---

## Step 2 - Differential Privacy Run (Optional Demo)

To show privacy-enabled training:

```bash
python train_dp.py --config config/config.yaml --num-rounds 2 --local-epochs 1 --epsilon 8.0 --delta 1e-5
```

This run shows:
- DP-enabled client training
- epsilon spending updates
- final privacy guarantee summary

---

## Step 3 - Show Generated Artifacts

After run completion, we open these folders:

- `metrics/` -> round-wise JSON metrics
- `checkpoints/` -> best/final model files
- `logs/` -> detailed training logs
- `explanations/` -> Grad-CAM explanation images

These prove the system trained, evaluated, and exported explainability results.

---

## Step 4 - Dashboard View (if available)

If dashboard is running, we show:
- global loss curve
- global accuracy curve
- AUC trend
- epsilon/privacy trend for DP runs

This gives a live and visual understanding of training behavior.

---

## Results Summary (What to Say)

From the demo we can conclude:
- federated setup runs correctly with multiple hospital clients,
- aggregation updates global model each round,
- DP mode provides measurable privacy budget tracking,
- explainability outputs make predictions more interpretable.

This completes our demonstration. Thank you.

