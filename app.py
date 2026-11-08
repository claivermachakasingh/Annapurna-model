"""Annapurna ML Serving & Interactive Evaluation App for Hugging Face Spaces.

Provides interactive evaluation and API endpoints for:
1. Multimodal Food Safety & Quality Fusion (IoT + Vision + Shelf-life)
2. Institutional Kitchen Demand Forecasting (Quantile Ensemble + CQR + Newsvendor)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Setup environment paths so ml_pipeline modules and model registries resolve correctly
ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "ml_pipeline"
REGISTRY_DIR = PIPELINE_DIR / "registry"

sys.path.insert(0, str(PIPELINE_DIR))
os.environ["ML_PIPELINE_ROOT"] = str(ROOT_DIR)
os.environ["ML_REGISTRY_DIR"] = str(REGISTRY_DIR)

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Import Annapurna ML core inference functions
from ml_quality_inference import assess_safety, _load
from ml_demand_model import predict_demand
from ml_utils import get_paths


def evaluate_food_safety(
    category: str,
    hours_since_prepared: float,
    temperatures_str: str,
    step_minutes: float,
    humidity_mean: float,
    cv_unsafe_prob: float,
    hours_to_expiry: float,
):
    """Run multimodal safety fusion and remaining shelf-life estimation."""
    try:
        # Parse temperature series
        temp_strs = [t.strip() for t in temperatures_str.split(",") if t.strip()]
        if not temp_strs:
            temps = [65.0]
        else:
            temps = [float(t) for t in temp_strs]

        res = assess_safety(
            category=category,
            hours_since_prepared=float(hours_since_prepared),
            temps=temps,
            step_min=float(step_minutes),
            humidity_mean=float(humidity_mean),
            cv_unsafe_prob=float(cv_unsafe_prob),
            hours_to_expiry=float(hours_to_expiry) if hours_to_expiry > 0 else None,
        )

        status = res.get("status", "UNKNOWN")
        hours_left = res.get("hours_left", {})
        reasons = res.get("reasons", [])
        p_unsafe = res.get("p_unsafe", 0.0)

        # Build readable summary
        summary = (
            f"### Safety Status: **{status}**\n"
            f"- **Unsafe Probability:** {p_unsafe * 100:.2f}%\n"
            f"- **Conservative Lower Bound Safe Hours:** {hours_left.get('lower_bound', 0.0):.1f} hrs\n"
            f"- **Estimated Range (P10 - P90):** {hours_left.get('p10', 0.0):.1f} hrs – {hours_left.get('p90', 0.0):.1f} hrs\n"
            f"- **Flags / Reasons Triggered:** {', '.join(reasons) if reasons else 'None (Pass)'}\n"
            f"- **Human Approval Required:** {res.get('requires_human_approval', True)}"
        )
        return summary, res
    except Exception as e:
        err_msg = f"Error during safety assessment: {str(e)}"
        return err_msg, {"error": str(e)}


# Cache demand model bundle
_demand_bundle = None

def get_demand_bundle():
    global _demand_bundle
    if _demand_bundle is None:
        paths = get_paths()
        model_path = paths.latest_model_dir("demand") / "model.joblib"
        _demand_bundle = joblib.load(model_path)
    return _demand_bundle


def forecast_kitchen_demand(
    attendance: float,
    meal_type: str,
    day_of_week_str: str,
    service_level: float,
):
    """Run LightGBM quantile demand forecasting with conformal calibration."""
    try:
        dow_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        dow = dow_map.get(day_of_week_str, 0)

        bundle = get_demand_bundle()
        payload = {
            "attendance": float(attendance),
            "meal_type": meal_type.upper(),
            "day_of_week": dow,
            "service_level": float(service_level),
        }

        res = predict_demand(bundle, payload)
        quantiles = res.get("quantiles", {})
        plan = res.get("plan", {})

        summary = (
            f"### Production Recommendation: **{plan.get('recommended_units', 0):.0f} units**\n"
            f"- **Median Forecast (P50):** {quantiles.get('0.5', 0):.1f} units\n"
            f"- **80% Prediction Band (P10 - P90):** {quantiles.get('0.1', 0):.1f} – {quantiles.get('0.9', 0):.1f} units\n"
            f"- **Avoided Surplus Risk:** {plan.get('avoided_surplus_pct', 0):.1f}%\n"
            f"- **Decision Rule:** {res.get('decision_rule', 'Newsvendor optimum with service floor')}"
        )
        return summary, res
    except Exception as e:
        err_msg = f"Error in demand forecast: {str(e)}"
        return err_msg, {"error": str(e)}


# -----------------------------------------------------------------------------
# Gradio Interface Definition
# -----------------------------------------------------------------------------
CATEGORIES = [
    "rice_dal",
    "cooked_veg",
    "non_veg_curry",
    "dairy_based",
    "bread_roti",
    "fried_snack",
    "raw_salad",
    "sweets",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEALS = ["LUNCH", "DINNER", "BREAKFAST", "SNACK"]

theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
)

with gr.Blocks(theme=theme, title="Annapurna ML - AI Food Redistribution Engine") as demo:
    gr.Markdown(
        """
        # 🍲 Annapurna AI - Food Quality & Demand Engine
        ### Official SIH-26234 Machine Learning Inference Suite
        
        This suite provides decision support for institutional kitchens and food redistribution:
        - **Food Safety Fusion:** Merges computer vision spoilage confidence with IoT time-temperature histories and FSSAI standards.
        - **Demand Forecasting:** LightGBM Quantile Ensembles + Conformalized Quantile Regression (CQR) to minimize food waste.
        """
    )

    with gr.Tab("🛡️ Food Safety & Shelf-Life Assessment"):
        gr.Markdown("#### Multimodal Safety Fusion (IoT Sensors + Vision Risk)")
        with gr.Row():
            with gr.Column():
                in_category = gr.Dropdown(
                    label="Food Category",
                    choices=CATEGORIES,
                    value="rice_dal",
                )
                in_hours_prep = gr.Slider(
                    label="Hours Since Prepared",
                    minimum=0.0,
                    maximum=48.0,
                    value=2.0,
                    step=0.5,
                )
                in_temps = gr.Textbox(
                    label="Temperature Trace (°C, comma-separated)",
                    value="65, 62, 59, 57",
                    info="Trace of sensor readings over time",
                )
                in_step_min = gr.Number(
                    label="Sensor Step Interval (minutes)",
                    value=30.0,
                )
                in_humidity = gr.Slider(
                    label="Average Humidity (%)",
                    minimum=10.0,
                    maximum=100.0,
                    value=55.0,
                    step=1.0,
                )
                in_cv_prob = gr.Slider(
                    label="Vision Spoilage Probability (CV Model)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.05,
                    step=0.01,
                    info="Estimated visual deterioration score from camera",
                )
                in_expiry = gr.Number(
                    label="Label Expiry in Hours (0 if unlabelled)",
                    value=6.0,
                )
                btn_safety = gr.Button("Assess Batch Safety", variant="primary")

            with gr.Column():
                out_safety_summary = gr.Markdown(label="Assessment Summary")
                out_safety_json = gr.JSON(label="Detailed Model Output")

        btn_safety.click(
            fn=evaluate_food_safety,
            inputs=[
                in_category,
                in_hours_prep,
                in_temps,
                in_step_min,
                in_humidity,
                in_cv_prob,
                in_expiry,
            ],
            outputs=[out_safety_summary, out_safety_json],
        )

    with gr.Tab("📈 Kitchen Demand Forecasting"):
        gr.Markdown("#### Institutional Kitchen Demand & Production Planning")
        with gr.Row():
            with gr.Column():
                in_attendance = gr.Number(
                    label="Expected Attendance",
                    value=350,
                    info="Estimated patrons or hostel residents",
                )
                in_meal_type = gr.Dropdown(
                    label="Meal Type",
                    choices=MEALS,
                    value="LUNCH",
                )
                in_dow = gr.Dropdown(
                    label="Day of Week",
                    choices=DAYS,
                    value="Wednesday",
                )
                in_service_level = gr.Slider(
                    label="Target Service Level (Floor)",
                    minimum=0.70,
                    maximum=0.99,
                    value=0.85,
                    step=0.01,
                    info="Prevents kitchen stockouts while capping waste",
                )
                btn_demand = gr.Button("Forecast Demand Plan", variant="primary")

            with gr.Column():
                out_demand_summary = gr.Markdown(label="Forecast Summary")
                out_demand_json = gr.JSON(label="Detailed Model Output")

        btn_demand.click(
            fn=forecast_kitchen_demand,
            inputs=[in_attendance, in_meal_type, in_dow, in_service_level],
            outputs=[out_demand_summary, out_demand_json],
        )

    with gr.Tab("ℹ️ About the Models"):
        gr.Markdown(
            """
            ### System Subsystems & Guarantees
            
            1. **Visual Freshness Screening**:
               - Backbone: `tf_efficientnet_lite0` & DINOv2 linear probe.
               - Gated on **≥ 0.98 unsafe recall** on unseen-source splits.
            
            2. **Multimodal Safety Fusion**:
               - Monotonic LightGBM calibrated with isotonic regression.
               - Integrates FSSAI danger zone guidelines (5°C - 60°C).
            
            3. **Shelf-Life Prediction**:
               - Regresses remaining safe hours with split conformal intervals.
               - Guarantees finite-sample coverage on the conservative lower bound.
            
            4. **Demand Optimization**:
               - LightGBM quantile regression ensemble at P10, P50, P75, P90, P95.
               - Conformalized Quantile Regression (CQR) avoids data leakage.
               - Avoids ~45.6% surplus waste compared to historical baseline.
            """
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
