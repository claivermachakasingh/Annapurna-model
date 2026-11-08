---
title: Annapurna AI Food Quality & Demand Engine
emoji: 🍲
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🍲 Annapurna AI - Machine Learning Engine (SIH-26234)

Decision support models for institutional kitchen demand prediction, surplus prevention, and multimodal food quality assessment.

## Subsystems

1. **Food Quality & Safety Fusion Screen**:
   - Integrates computer vision risk probabilities (`tf_efficientnet_lite0` / DINOv2) with real-time temperature, humidity, and time-in-danger-zone traces.
   - Monotonic LightGBM calibrated with isotonic regression following FSSAI standards.
   - Outputs: `GOOD`, `RISK`, or `REJECTED` with specific violation reasons.

2. **Remaining Shelf-Life Estimation**:
   - LightGBM regressor with split conformal prediction intervals.
   - Outputs: Conservative safe hours lower bound, P10, P50 (median), and P90 confidence bounds.

3. **Demand Forecasting & Production Planning**:
   - LightGBM quantile regression ensemble at P10/P50/P75/P90/P95.
   - Conformalized Quantile Regression (CQR) with Newsvendor production optimization.
   - Avoids surplus food waste while preventing kitchen stockouts.

## Local Setup & Execution

```bash
git clone https://github.com/claivermachakasingh/Annapurna-model.git
cd Annapurna-model

# Install dependencies
pip install -r requirements.txt

# Run the Gradio interactive app
python app.py
```
