# Annapurna ML Inference & Model Serving (SIH-26234)

Production machine learning inference pipeline and calibrated models for institutional kitchen demand prediction, surplus prevention, and food safety quality fusion.

---

## Repository Structure

```text
Annapurna-model/
├── .gitignore
├── README.md                                # Deployment instructions
└── ml_pipeline/
    ├── ml_demand_features.py                # Feature extraction for demand forecasting
    ├── ml_demand_model.py                   # Demand forecast & Newsvendor production plan
    ├── ml_quality_inference.py              # Food safety fusion & shelf-life inference
    ├── ml_quality_time_temp.py              # IoT temperature & danger-zone summarization
    ├── ml_analytics_processing_anomaly.py   # Kitchen process anomaly detector
    ├── ml_analytics_sustainability.py       # Impact & emissions calculation
    ├── ml_logistics_match.py                # Bipartite matching for surplus distribution
    ├── ml_logistics_route.py                # OR-Tools VRPTW vehicle route planning
    ├── ml_utils.py                          # Paths, config loader, logging, determinism
    ├── ml_device.py                         # Device capability detection (CPU / GPU)
    ├── ml_configs_fusion.yaml               # FSSAI safety bands & category parameters
    ├── ml_configs_sustainability.yaml       # Carbon & cost factor configurations
    ├── ml_requirements.txt                  # Core inference runtime dependencies
    └── registry/
        └── models/
            ├── anomaly/                     # Cold-chain & processing anomaly model configs
            ├── anomaly_sih2026/             # SIH 2026 benchmark anomaly evaluation
            ├── demand/v10/                  # LightGBM demand forecast model (v10)
            ├── demand_sih2026/v15/          # SIH 2026 demand model bundle (v15)
            ├── fusion/v7/                   # Quality fusion model (v7)
            ├── fusion_sih2026/v13/          # SIH 2026 multimodal fusion model (v13)
            ├── shelf_life/v6/               # Remaining shelf-life regressor (v6)
            ├── shelf_life_sih2026/v8/       # SIH 2026 shelf-life model bundle (v8)
            └── demand_model.json            # Model active pointer & metadata
```

---

## 1. Setup & Installation

Ensure Python 3.10+ is installed:

```bash
git clone https://github.com/claivermachakasingh/Annapurna-model.git
cd Annapurna-model

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install runtime dependencies
pip install -r ml_pipeline/ml_requirements.txt
```

---

## 2. Model Inference Usage

### A. Food Safety & Shelf-Life Assessment (`assess_safety`)

Fuses sensor time-temperature traces with computer vision risk and applies FSSAI safety rules and conformal prediction intervals:

```python
import sys
from pathlib import Path
sys.path.append(str(Path("ml_pipeline").resolve()))

from ml_quality_inference import assess_safety

result = assess_safety(
    category="rice_dal",
    hours_since_prepared=2.0,
    temps=[65.0, 62.0, 58.0, 55.0],    # Temperature trace (°C)
    step_min=30.0,                      # 30-min sampling interval
    humidity_mean=55.0,                 # Relative humidity (%)
    cv_unsafe_prob=0.05,                # Spoilage probability from vision model
    hours_to_expiry=6.0                 # Label expiration in hours
)

print("Status:", result["status"])         # GOOD / RISK / REJECTED
print("Safe Hours Left:", result["hours_left"])  # Conformal lower bound, P10, P50, P90
print("Unsafe Probability:", result["p_unsafe"])
print("Flags / Reasons:", result["reasons"])
```

### B. Kitchen Demand Forecasting & Newsvendor Planning (`predict_demand`)

Generates calibrated quantile demand forecasts (P10/P50/P90) and calculates the optimal production target:

```python
import joblib
from pathlib import Path
from ml_demand_model import predict_demand
from ml_utils import get_paths

paths = get_paths()
bundle = joblib.load(paths.latest_model_dir("demand") / "model.joblib")

payload = {
    "attendance": 350,                  # Expected patrons
    "meal_type": "LUNCH",               # BREAKFAST / LUNCH / SNACK / DINNER
    "day_of_week": 2,                   # 0=Monday .. 6=Sunday
    "service_level": 0.85               # Target service level floor
}

forecast = predict_demand(bundle, payload)
print("Recommended Units to Cook:", forecast["plan"]["recommended_units"])
print("Quantiles:", forecast["quantiles"])
```

### C. Logistics Matching & Vehicle Routing

```python
from ml_logistics_route import solve_vrptw

# Matches surplus batches to destination nodes and calculates optimal vehicle routes
```

---

## 3. Environment Variables (Optional)

* `ML_PIPELINE_ROOT`: Root path to the repository (defaults to repository root).
* `ML_REGISTRY_DIR`: Path to the model registry directory (defaults to `ml_pipeline/registry`).
