
"""Annapurna ML Serving & Interactive Evaluation App.

Demand forecasting is exposed through FastAPI for Go backend integration.
The existing Gradio interface is retained.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "ml_pipeline"
REGISTRY_DIR = PIPELINE_DIR / "registry"

sys.path.insert(0, str(PIPELINE_DIR))

os.environ["ML_PIPELINE_ROOT"] = str(ROOT_DIR)
os.environ["ML_REGISTRY_DIR"] = str(REGISTRY_DIR)

import joblib
import gradio as gr
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_quality_inference import assess_safety
from ml_demand_model import predict_demand
from ml_utils import get_paths


# --------------------------------------------------
# FOOD SAFETY INFERENCE
# Existing functionality retained for Gradio.
# Not integrated into the Annapurna application.
# --------------------------------------------------

def evaluate_food_safety(
    category: str,
    hours_since_prepared: float,
    temperatures_str: str,
    step_minutes: float,
    humidity_mean: float,
    cv_unsafe_prob: float,
    hours_to_expiry: float,
):
    try:
        temp_strs = [
            t.strip()
            for t in temperatures_str.split(",")
            if t.strip()
        ]

        temps = [float(t) for t in temp_strs] if temp_strs else [65.0]

        res = assess_safety(
            category=category,
            hours_since_prepared=float(hours_since_prepared),
            temps=temps,
            step_min=float(step_minutes),
            humidity_mean=float(humidity_mean),
            cv_unsafe_prob=float(cv_unsafe_prob),
            hours_to_expiry=(
                float(hours_to_expiry)
                if hours_to_expiry > 0
                else None
            ),
        )

        status = res.get("status", "UNKNOWN")
        hours_left = res.get("hours_left", {})
        reasons = res.get("reasons", [])
        p_unsafe = res.get("p_unsafe", 0.0)

        summary = (
            f"### Safety Status: **{status}**\n"
            f"- **Unsafe Probability:** {p_unsafe * 100:.2f}%\n"
            f"- **Conservative Lower Bound Safe Hours:** "
            f"{hours_left.get('lower_bound', 0.0):.1f} hrs\n"
            f"- **Estimated Range (P10 - P90):** "
            f"{hours_left.get('p10', 0.0):.1f} – "
            f"{hours_left.get('p90', 0.0):.1f} hrs\n"
            f"- **Flags / Reasons:** "
            f"{', '.join(reasons) if reasons else 'None (Pass)'}\n"
            f"- **Human Approval Required:** "
            f"{res.get('requires_human_approval', True)}"
        )

        return summary, res

    except Exception as e:
        print(f"SAFETY ERROR: {e}", flush=True)
        return f"Error during safety assessment: {e}", {"error": str(e)}


# --------------------------------------------------
# DEMAND MODEL LOADING
# --------------------------------------------------

_demand_bundle = None


def get_demand_bundle():
    """Load the existing trained demand model once."""

    global _demand_bundle

    if _demand_bundle is None:
        paths = get_paths()
        model_path = paths.latest_model_dir("demand") / "model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Demand model not found: {model_path}"
            )

        print(f"Loading demand model: {model_path}", flush=True)

        _demand_bundle = joblib.load(model_path)

        print("Demand model loaded successfully.", flush=True)

    return _demand_bundle


# --------------------------------------------------
# SHARED DEMAND PREDICTION
# --------------------------------------------------

DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

MEALS = ["LUNCH", "DINNER", "BREAKFAST", "SNACK"]


def run_demand_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """Call the original trained demand model without modification."""

    return predict_demand(get_demand_bundle(), payload)


# --------------------------------------------------
# GRADIO DEMAND FORECASTING
# --------------------------------------------------

def forecast_kitchen_demand(
    attendance: float,
    meal_type: str,
    day_of_week_str: str,
    service_level: float,
):
    """Interactive forecasting using the existing model."""

    try:
        dow = DAYS.index(day_of_week_str)

        payload = {
            "attendance": float(attendance),
            "meal_type": meal_type.upper(),
            "day_of_week": dow,
            "service_level": float(service_level),
        }

        res = run_demand_prediction(payload)

        interval = res.get("prediction_interval", {})

        recommended = float(res.get("recommended_production", 0))
        median = float(
            interval.get(
                "p50",
                res.get("predicted_consumption", 0),
            )
        )

        p10 = float(interval.get("p10", 0))
        p90 = float(interval.get("p90", 0))

        summary = (
            f"### Production Recommendation: **{recommended:.0f} units**\n"
            f"- **Median Forecast (P50):** {median:.1f} units\n"
            f"- **80% Prediction Band (P10 - P90):** "
            f"{p10:.1f} – {p90:.1f} units\n"
            f"- **Surplus Risk:** {res.get('surplus_risk', 'UNKNOWN')}\n"
            f"- **Expected Surplus:** {res.get('expected_surplus', 0)} units\n"
            f"- **Decision Rule:** "
            f"{res.get('operating_point_set_by', 'Not specified')}"
        )

        print(f"DEMAND SUMMARY: {summary}", flush=True)

        return summary, res

    except Exception as e:
        print(f"DEMAND FORECAST ERROR: {e}", flush=True)

        return (
            f"Error in demand forecast: {str(e)}",
            {"error": str(e)},
        )


# --------------------------------------------------
# FASTAPI REQUEST CONTRACT
# Matches the current Go ML client request.
# --------------------------------------------------

class PredictDemandRequest(BaseModel):
    kitchen_id: int
    food_category: str
    date_time: datetime | None = None
    features: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------
# GO REQUEST -> EXISTING MODEL PAYLOAD
# --------------------------------------------------

def build_model_payload(request: PredictDemandRequest) -> dict[str, Any]:

    features = request.features or {}

    # Attendance
    attendance = features.get(
        "attendance",
        features.get("expected_diners"),
    )

    if attendance is None:
        raise ValueError("Missing attendance in request features.")

    attendance = float(attendance)

    if attendance < 0:
        raise ValueError("Attendance cannot be negative.")

    # Meal type
    meal_type = (
        features.get("meal_type") or request.food_category or ""
    ).strip().upper()

    if meal_type not in MEALS:
        raise ValueError(
            f"Invalid meal_type: {meal_type}. "
            f"Allowed values: {', '.join(MEALS)}"
        )

    # Day of week
    day_value = features.get("day_of_week")

    if day_value is not None:
        if isinstance(day_value, str):
            day_names = {name.lower(): i for i, name in enumerate(DAYS)}
            normalized = day_value.strip().lower()

            if normalized in day_names:
                day_of_week = day_names[normalized]
            else:
                day_of_week = int(day_value)
        else:
            day_of_week = int(day_value)

    elif request.date_time is not None:
        day_of_week = request.date_time.weekday()

    else:
        raise ValueError(
            "Missing day_of_week or date_time."
        )

    if not 0 <= day_of_week <= 6:
        raise ValueError("day_of_week must be between 0 and 6.")

    # Base payload expected by predict_demand()
    payload: dict[str, Any] = {
        "attendance": attendance,
        "meal_type": meal_type,
        "day_of_week": day_of_week,
    }

    # Menu
    menu = features.get("menu")

    if menu is not None:
        if not isinstance(menu, list):
            raise ValueError("menu must be a list.")

        payload["menu"] = menu

    # Historical consumption
    historical_consumption = features.get("historical_consumption")

    if historical_consumption is not None:
        if not isinstance(historical_consumption, list):
            raise ValueError("historical_consumption must be a list.")

        payload["historical_consumption"] = historical_consumption

    # Historical surplus / waste
    historical_surplus = features.get("historical_surplus")

    if historical_surplus is not None:
        if not isinstance(historical_surplus, list):
            raise ValueError("historical_surplus must be a list.")

        payload["historical_surplus"] = historical_surplus

    # Optional model optimization settings
    for key in ("service_level", "cost_under", "cost_over"):
        if features.get(key) is not None:
            payload[key] = float(features[key])

    return payload


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

api = FastAPI(
    title="Annapurna Demand Forecasting API",
    description="Demand forecasting using the existing trained model.",
    version="1.0.0",
)


@api.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "annapurna-demand",
    }


@api.get("/health/demand")
def demand_health_check():
    try:
        get_demand_bundle()

        return {
            "status": "healthy",
            "model": "demand",
            "model_loaded": True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )


@api.post("/predict/demand")
def predict_demand_endpoint(request: PredictDemandRequest):
    try:
        payload = build_model_payload(request)

        print(
            "DEMAND REQUEST: "
            f"kitchen={request.kitchen_id}, "
            f"meal={payload['meal_type']}, "
            f"attendance={payload['attendance']}, "
            f"day={payload['day_of_week']}",
            flush=True,
        )

        result = run_demand_prediction(payload)

        print(
            "DEMAND SUCCESS: "
            f"prediction={result.get('predicted_consumption')}, "
            f"production={result.get('recommended_production')}",
            flush=True,
        )

        # Return the complete actual model result.
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    except Exception as e:
        print(f"DEMAND API ERROR: {e}", flush=True)

        raise HTTPException(
            status_code=500,
            detail={
                "code": getattr(
                    e,
                    "code",
                    "DEMAND_PREDICTION_FAILED",
                ),
                "message": getattr(e, "message", str(e)),
            },
        )


# --------------------------------------------------
# GRADIO INTERFACE
# Original interface retained.
# --------------------------------------------------

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

theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
)


with gr.Blocks(title="Annapurna ML - AI Food Redistribution Engine") as demo:

    gr.Markdown(
        """
        # 🍲 Annapurna AI - Food Quality & Demand Engine

        ### Official SIH-26234 Machine Learning Inference Suite

        Decision support for institutional kitchens and food redistribution.

        - **Food Safety Fusion:** Combines computer vision risk,
          IoT time-temperature histories and food safety standards.
        - **Demand Forecasting:** LightGBM quantile ensembles,
          conformal prediction and production optimization.
        """
    )

    # ----------------------------------------------
    # FOOD SAFETY TAB
    # ----------------------------------------------

    with gr.Tab("🛡️ Food Safety & Shelf-Life Assessment"):

        gr.Markdown("#### Multimodal Safety Fusion")

        with gr.Row():

            with gr.Column():

                in_category = gr.Dropdown(
                    label="Food Category",
                    choices=CATEGORIES,
                    value="rice_dal",
                )

                in_hours_prep = gr.Slider(
                    label="Hours Since Prepared",
                    minimum=0,
                    maximum=48,
                    value=2,
                    step=0.5,
                )

                in_temps = gr.Textbox(
                    label="Temperature Trace (°C)",
                    value="65, 62, 59, 57",
                )

                in_step_min = gr.Number(
                    label="Sensor Step Interval (minutes)",
                    value=30,
                )

                in_humidity = gr.Slider(
                    label="Average Humidity (%)",
                    minimum=10,
                    maximum=100,
                    value=55,
                )

                in_cv_prob = gr.Slider(
                    label="Vision Spoilage Probability",
                    minimum=0,
                    maximum=1,
                    value=0.05,
                    step=0.01,
                )

                in_expiry = gr.Number(
                    label="Label Expiry in Hours",
                    value=6,
                )

                btn_safety = gr.Button(
                    "Assess Batch Safety",
                    variant="primary",
                )

            with gr.Column():

                out_safety_summary = gr.Markdown()
                out_safety_json = gr.JSON(
                    label="Detailed Model Output"
                )

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
            outputs=[
                out_safety_summary,
                out_safety_json,
            ],
        )

    # ----------------------------------------------
    # DEMAND FORECASTING TAB
    # ----------------------------------------------

    with gr.Tab("📈 Kitchen Demand Forecasting"):

        gr.Markdown(
            "#### Institutional Kitchen Demand & Production Planning"
        )

        with gr.Row():

            with gr.Column():

                in_attendance = gr.Number(
                    label="Expected Attendance",
                    value=350,
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
                )

                btn_demand = gr.Button(
                    "Forecast Demand Plan",
                    variant="primary",
                )

            with gr.Column():

                out_demand_summary = gr.Markdown(
                    label="Forecast Summary"
                )

                out_demand_json = gr.JSON(
                    label="Detailed Model Output"
                )

        btn_demand.click(
            fn=forecast_kitchen_demand,
            inputs=[
                in_attendance,
                in_meal_type,
                in_dow,
                in_service_level,
            ],
            outputs=[
                out_demand_summary,
                out_demand_json,
            ],
        )

    # ----------------------------------------------
    # ABOUT TAB
    # ----------------------------------------------

    with gr.Tab("ℹ️ About the Models"):

        gr.Markdown(
            """
            ### System Subsystems

            1. **Visual Freshness Screening**
               - Computer vision-based spoilage screening.

            2. **Multimodal Safety Fusion**
               - Combines sensor data and vision risk.

            3. **Shelf-Life Prediction**
               - Estimates remaining safe hours.

            4. **Demand Optimization**
               - LightGBM quantile regression.
               - Conformal prediction intervals.
               - Newsvendor production optimization.

            **Note:** Synthetic simulator data is not evidence
            of performance in a real institutional kitchen.
            """
        )


# --------------------------------------------------
# MOUNT GRADIO AND START SERVER
# --------------------------------------------------

app = gr.mount_gradio_app(
    api,
    demo.queue(),
    path="/",
)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    print(
        f"Starting Annapurna demand ML service on port {port}",
        flush=True,
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
