import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.main import main as run_training_pipeline

OUTPUT_DIR = Path("outputs")
RESULT_CSV = OUTPUT_DIR / "result.csv"
CHART_FILES = {
    "Residual Distribution": OUTPUT_DIR / "residual_chart.png",
    "SHAP Feature Importance": OUTPUT_DIR / "Feature_Importance.png",
    "SHAP Summary": OUTPUT_DIR / "Shap_Summary.png",
    "SHAP Waterfall": OUTPUT_DIR / "Waterfall_Plot.png",
}


def _secret_to_json_string(secret_value: Any) -> str:
    if isinstance(secret_value, dict):
        return json.dumps(secret_value)
    if isinstance(secret_value, str):
        return secret_value
    raise TypeError("GCP_SERVICE_ACCOUNT_KEY must be a JSON string or TOML table.")


def _run_training_with_credentials() -> tuple[bool, str]:
    secret_value = st.secrets.get("GCP_SERVICE_ACCOUNT_KEY")
    if secret_value is None:
        return False, "Missing secret: GCP_SERVICE_ACCOUNT_KEY"

    cred_json = _secret_to_json_string(secret_value)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(cred_json)
            temp_path = temp_file.name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = cred_json
        run_training_pipeline()
        return True, "Training completed successfully."
    except Exception:
        return False, traceback.format_exc()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS_JSON", None)


def _render_results() -> None:
    st.subheader("Model Outputs")

    if RESULT_CSV.exists():
        results_df = pd.read_csv(RESULT_CSV)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(results_df))
        col2.metric("Mean Residual", f"{results_df['residual'].mean():.4f}")
        col3.metric("MAE", f"{results_df['residual'].abs().mean():.4f}")
        st.dataframe(results_df.head(20), use_container_width=True)
    else:
        st.warning("`outputs/result.csv` not found yet. Run training to generate results.")

    st.subheader("Visualisations")
    for title, image_path in CHART_FILES.items():
        if image_path.exists():
            st.image(str(image_path), caption=title, use_container_width=True)
        else:
            st.info(f"{title} not available yet: `{image_path}`")


def main() -> None:
    st.set_page_config(page_title="XGBoost Ecommerce Forecast", layout="wide")
    st.title("XGBoost Ecommerce Spend Forecast")
    st.write(
        "Run the end-to-end training pipeline and explore prediction outputs generated in `outputs/`."
    )

    if "last_run_ok" not in st.session_state:
        st.session_state.last_run_ok = None
        st.session_state.last_run_message = ""

    with st.sidebar:
        st.title("Training Controls")
        if st.button("Run Training", type="primary"):
            with st.spinner("Training model and generating outputs..."):
                ok, message = _run_training_with_credentials()
                st.session_state.last_run_ok = ok
                st.session_state.last_run_message = message

    st.title("Ecommerce Spend Forecast Dashboard")

    if st.session_state.last_run_ok is True:
        # st.success(st.session_state.last_run_message)
        st.success("Analysis Complete.")
    elif st.session_state.last_run_ok is False:
        st.error("Training failed. Check details below.")
        with st.expander("Show error details"):
            st.code(st.session_state.last_run_message)

    # st.subheader("Diagnostics")
    # st.write(f"Outputs directory: `{OUTPUT_DIR.resolve()}`")
    # st.write(f"Outputs exists: `{OUTPUT_DIR.exists()}`")
    # if OUTPUT_DIR.exists():
    #     files = [p.name for p in OUTPUT_DIR.iterdir() if p.is_file()]
    #     st.write("Generated files:", files if files else "No files yet.")

    _render_results()


if __name__ == "__main__":
    main()
