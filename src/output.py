from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

OUTPUT_DIR = Path("outputs")


def _ensure_outputs_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

def save_result(y_test: pd.Series,
                preds: np.ndarray,
                filename: str = 'result.csv') -> None:
    """Output y_test, preds, residual to csv file"""
    output_path = _ensure_outputs_dir() / filename
    results = pd.DataFrame({
                    'y_test': y_test,
                    'prediction': preds,
                    'residual': y_test - preds})
    results.to_csv(output_path, index=False)
    print(f'Results saved to {output_path}')

def plot_residuals(y_test:pd.Series,
                 preds:np.ndarray) -> None:
    """Plot residual chart"""
    output_path = _ensure_outputs_dir() / "residual_chart.png"
    residual = y_test - preds
    plt.figure(figsize=(8, 6))
    plt.hist(x=residual, bins=100, color="orange", edgecolor="black")
    plt.xlabel("Residual Historgram Chart")
    plt.ylabel("Residual Spend Count")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def shap_analysis(model: Any, x_test:pd.DataFrame) ->None:
    """Generate SHAP charts to explain feature importance and save as png format"""
    output_dir = _ensure_outputs_dir()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_test)

    # bar chart: which feature is more important?
    plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP Bar)")
    plt.savefig(output_dir / "Feature_Importance.png")
    plt.close()

    # scatter chart: how the features impact the prediction when feature is higher/lower
    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False)
    plt.title('Shap_Summary')
    plt.savefig(output_dir / "Shap_Summary.png")
    plt.close()

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title('Waterfall_Plot')
    plt.savefig(output_dir / "Waterfall_Plot.png")
    plt.close()

    # if "country" in X_test.columns:
    #     plt.figure()
    #     shap.dependence_plot(
    #         "country", shap_values.values, X_test, interaction_index=None, show=False
    #     )
    #     plt.title("Impact of Country Labels")
    #     plt.show()

    # return shap_values)
