
"""Annapurna ML Serving & Interactive Evaluation App."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

from ml_quality_inference import assess_safety
from ml_demand_model import predict_demand
from ml_utils import get_paths


# --------------------------------------------------
# FOOD SAFETY INFERENCE
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
    global _demand_bundle

    if _demand_bundle is None:
        paths = get_paths()
        model_path = paths.latest_model_dir("demand") / "model.joblib"

        print(f"Loading demand model: {model_path}", flush=True)

        _demand_bundle = joblib.load(model_path)

    return _demand_bundle


# --------------------------------------------------
# DEMAND FORECASTING
# --------------------------------------------------

def forecast_kitchen_demand(
    attendance: float,
    meal_type: str,
    day_of_week_str: str,
    service_level: float,
):
    """Run demand forecasting and display the actual model output."""

    print("RUNNING UPDATED DEMAND FORECAST FUNCTION", flush=True)

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

        # Actual model output structure
        interval = res.get("prediction_interval", {})

        recommended = float(res.get("recommended_production", 0))
        median = float(
            interval.get(
                "p50",
                res.get("predicted_consumption", 0)
            )
        )

        p10 = float(interval.get("p10", 0))
        p90 = float(interval.get("p90", 0))

        surplus_risk = res.get("surplus_risk", "UNKNOWN")
        expected_surplus = res.get("expected_surplus", 0)
        decision_rule = res.get(
            "operating_point_set_by",
            "Newsvendor optimum with service floor"
        )

        summary = (
            f"### Production Recommendation: **{recommended:.0f} units**\n"
            f"- **Median Forecast (P50):** {median:.1f} units\n"
            f"- **80% Prediction Band (P10 - P90):** "
            f"{p10:.1f} – {p90:.1f} units\n"
            f"- **Surplus Risk:** {surplus_risk}\n"
            f"- **Expected Surplus:** {expected_surplus} units\n"
            f"- **Decision Rule:** {decision_rule}"
        )

        print(f"DEMAND SUMMARY: {summary}", flush=True)

        return summary, res

    except Exception as e:
        print(f"DEMAND FORECAST ERROR: {e}", flush=True)

        return (
            f"Error in demand forecast: {str(e)}",
            {"error": str(e)}
        )


# --------------------------------------------------
# GRADIO INTERFACE
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

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

MEALS = ["LUNCH", "DINNER", "BREAKFAST", "SNACK"]

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
# RENDER SERVER STARTUP
# --------------------------------------------------

if __name__ == "__main__":

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
