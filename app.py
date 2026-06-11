from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lr_scratch.config import ExperimentConfig
from lr_scratch.experiments import (
    benchmark,
    run_learning_rate_experiment,
    run_scaling_comparison,
    train_from_config,
)
from lr_scratch.model import LinearRegressionGD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "linear_regression_model.json"

st.set_page_config(
    page_title="Linear Regression From Scratch",
    page_icon="LR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --app-bg: #f5f7fb;
        --panel-bg: #ffffff;
        --text-main: #111827;
        --text-muted: #4b5563;
        --border: #d7dde7;
        --accent: #2563eb;
        --chart-bg: #101418;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--app-bg);
        color: var(--text-main);
    }
    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.92);
        border-bottom: 1px solid var(--border);
    }
    [data-testid="stSidebar"] {
        background: var(--panel-bg);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--text-main);
    }
    .small-label {
        color: var(--text-muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .tight {
        margin-top: -10px;
        color: var(--text-muted);
    }
    div[data-testid="stMetric"] {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(17, 24, 39, 0.06);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-main) !important;
    }
    button[kind="primary"], div.stButton > button {
        background: var(--accent);
        color: #ffffff !important;
        border: 1px solid var(--accent);
        border-radius: 8px;
    }
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 14px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-color: var(--accent);
    }
    [data-testid="stDataFrame"],
    [data-testid="stJson"],
    [data-testid="stCodeBlock"] {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def use_dark_chart_theme(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        paper_bgcolor="#101418",
        plot_bgcolor="#101418",
        font={"color": "#f9fafb"},
        title_font={"color": "#ffffff", "size": 18},
        legend={
            "bgcolor": "rgba(16, 20, 24, 0.9)",
            "bordercolor": "#374151",
            "borderwidth": 1,
            "font": {"color": "#f9fafb"},
        },
        margin={"l": 50, "r": 25, "t": 60, "b": 45},
    )
    fig.update_xaxes(
        color="#f9fafb",
        gridcolor="#293241",
        zerolinecolor="#6b7280",
        linecolor="#4b5563",
        tickfont={"color": "#f9fafb"},
        title_font={"color": "#ffffff"},
    )
    fig.update_yaxes(
        color="#f9fafb",
        gridcolor="#293241",
        zerolinecolor="#6b7280",
        linecolor="#4b5563",
        tickfont={"color": "#f9fafb"},
        title_font={"color": "#ffffff"},
    )
    return fig


def current_config() -> ExperimentConfig:
    return ExperimentConfig(
        n_samples=int(st.session_state.n_samples),
        noise=float(st.session_state.noise),
        true_slope=float(st.session_state.true_slope),
        true_intercept=float(st.session_state.true_intercept),
        train_ratio=float(st.session_state.train_ratio),
        seed=int(st.session_state.seed),
        learning_rate=float(st.session_state.learning_rate),
        max_epochs=int(st.session_state.max_epochs),
        tolerance=float(st.session_state.tolerance),
        patience=int(st.session_state.patience),
        early_stopping=bool(st.session_state.early_stopping),
        scale_features=bool(st.session_state.scale_features),
    )


with st.sidebar:
    st.header("Experiment Config")
    st.slider("Samples", 50, 5_000, 300, step=50, key="n_samples")
    st.slider("Noise", 0.0, 10.0, 2.5, step=0.25, key="noise")
    st.slider("Train ratio", 0.5, 0.9, 0.8, step=0.05, key="train_ratio")
    st.number_input("Seed", value=42, step=1, key="seed")
    st.divider()
    st.slider("True slope", 0.5, 8.0, 2.5, step=0.25, key="true_slope")
    st.slider("True intercept", -10.0, 20.0, 5.0, step=0.5, key="true_intercept")
    st.divider()
    st.select_slider(
        "Learning rate",
        options=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
        value=0.05,
        key="learning_rate",
    )
    st.select_slider("Max epochs", options=[100, 250, 500, 1_000, 2_000, 5_000], value=2_000, key="max_epochs")
    st.number_input("Tolerance", min_value=0.0, value=1e-8, format="%.1e", key="tolerance")
    st.number_input("Patience", min_value=1, value=25, step=1, key="patience")
    st.checkbox("Early stopping", value=True, key="early_stopping")
    st.checkbox("Scale features", value=True, key="scale_features")
    train_clicked = st.button("Train", use_container_width=True)

st.title("Linear Regression From Scratch")
st.markdown(
    "<p class='tight'>Pure NumPy batch gradient descent, manual split, train-only scaling, diagnostics, persistence, and experiments.</p>",
    unsafe_allow_html=True,
)

if train_clicked or "bundle" not in st.session_state:
    cfg = current_config()
    with st.spinner("Training NumPy model..."):
        bundle = train_from_config(cfg)
    st.session_state.bundle = bundle
    st.session_state.config = cfg
    LOGGER.info("Trained model with config: %s", cfg)

bundle = st.session_state.bundle
config = st.session_state.config

tab_train, tab_diag, tab_experiments, tab_math, tab_artifacts = st.tabs(
    ["Training", "Diagnostics", "Experiments", "Mathematics", "Artifacts"]
)

with tab_train:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test R2", f"{bundle.test_metrics['r2']:.4f}")
    c2.metric("Test RMSE", f"{bundle.test_metrics['rmse']:.4f}")
    c3.metric("Test MAE", f"{bundle.test_metrics['mae']:.4f}")
    c4.metric("Epochs", bundle.model.epochs_run_)
    c5.metric("Stop reason", bundle.model.stopped_reason_)

    col_left, col_right = st.columns(2)
    with col_left:
        x_line_raw = np.linspace(bundle.x_train_raw.min(), bundle.x_test_raw.max(), 200).reshape(-1, 1)
        x_line_model = bundle.scaler.transform(x_line_raw) if bundle.scaler else x_line_raw
        y_line = bundle.model.predict(x_line_model)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bundle.x_train_raw.ravel(), y=bundle.y_train, mode="markers", name="Train"))
        fig.add_trace(go.Scatter(x=bundle.x_test_raw.ravel(), y=bundle.y_test, mode="markers", name="Test"))
        fig.add_trace(go.Scatter(x=x_line_raw.ravel(), y=y_line, mode="lines", name="Prediction"))
        use_dark_chart_theme(fig, "Data and Fitted Regression Line", "x", "y", 360)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bundle.model.cost_history_, mode="lines", name="Train cost"))
        if bundle.model.validation_cost_history_ is not None and len(bundle.model.validation_cost_history_) > 0:
            fig.add_trace(go.Scatter(y=bundle.model.validation_cost_history_, mode="lines", name="Validation cost"))
        use_dark_chart_theme(fig, "Cost Convergence", "Epoch", "J(theta)", 360)
        st.plotly_chart(fig, use_container_width=True)

    metrics_df = pd.DataFrame(
        [
            {"Split": "Train", **bundle.train_metrics},
            {"Split": "Test", **bundle.test_metrics},
        ]
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with tab_diag:
    residuals = bundle.y_test - bundle.y_test_pred
    col_left, col_right = st.columns(2)
    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bundle.y_test_pred, y=residuals, mode="markers", name="Residual"))
        fig.add_hline(y=0, line_dash="dash")
        use_dark_chart_theme(fig, "Residuals vs Predicted", "Predicted", "Residual", 340)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=residuals, nbinsx=20, name="Residual distribution"))
        use_dark_chart_theme(fig, "Residual Distribution", "Residual", "Count", 340)
        st.plotly_chart(fig, use_container_width=True)

    min_y = min(float(bundle.y_test.min()), float(bundle.y_test_pred.min()))
    max_y = max(float(bundle.y_test.max()), float(bundle.y_test_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bundle.y_test, y=bundle.y_test_pred, mode="markers", name="Prediction"))
    fig.add_trace(go.Scatter(x=[min_y, max_y], y=[min_y, max_y], mode="lines", name="Perfect"))
    use_dark_chart_theme(fig, "Predicted vs Actual", "Actual", "Predicted", 360)
    st.plotly_chart(fig, use_container_width=True)

with tab_experiments:
    st.subheader("Learning Rate Experiment")
    learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    lr_df = pd.DataFrame(run_learning_rate_experiment(config, learning_rates))
    st.dataframe(lr_df, use_container_width=True, hide_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr_df["learning_rate"], y=lr_df["test_rmse"], mode="lines+markers", name="Test RMSE"))
    fig.add_trace(go.Scatter(x=lr_df["learning_rate"], y=lr_df["test_r2"], mode="lines+markers", name="Test R2"))
    use_dark_chart_theme(fig, "Learning Rate Sensitivity", "Learning rate", "Metric value", 360)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Scaling Comparison")
    scale_df = pd.DataFrame(run_scaling_comparison(config))
    st.dataframe(scale_df, use_container_width=True, hide_index=True)

    st.subheader("Performance Benchmark")
    bench_df = pd.DataFrame(benchmark(config, [100, 500, 1_000, 2_500, 5_000]))
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

with tab_math:
    st.markdown(
        """
        ### Objective
        For a design matrix `X` with a bias column and target vector `y`, the model is:

        `y_hat = X theta`

        The optimized objective is the half-MSE cost:

        `J(theta) = (1 / 2m) * (X theta - y)^T (X theta - y)`

        The gradient follows from matrix calculus:

        `dJ / dtheta = (1 / m) * X^T (X theta - y)`

        Batch gradient descent updates every parameter simultaneously:

        `theta := theta - alpha * (1 / m) * X^T (X theta - y)`

        ### Numerical Stability
        Features are z-score scaled using train-set mean and standard deviation only. Constant columns are assigned
        a scale of `1.0` to avoid division by zero. The training loop stops on divergence, small cost improvement, or
        validation patience exhaustion.

        ### Evaluation
        The app reports MSE, RMSE, MAE, and R2 on both train and held-out test data. Residual plots check whether
        errors are centered around zero and whether a linear model is systematically missing structure.
        """
    )

with tab_artifacts:
    st.subheader("Configuration")
    st.json(config.to_dict())
    if st.button("Save model JSON", use_container_width=True):
        bundle.model.save_json(MODEL_PATH, bundle.scaler)
        st.success(f"Saved model to {MODEL_PATH}")

    if MODEL_PATH.exists():
        loaded_model, loaded_scaler = LinearRegressionGD.load_json(MODEL_PATH)
        x_loaded = loaded_scaler.transform(bundle.x_test_raw) if loaded_scaler else bundle.x_test_raw
        max_delta = float(np.max(np.abs(loaded_model.predict(x_loaded) - bundle.y_test_pred)))
        st.write(f"Loaded-model prediction max delta: `{max_delta:.12f}`")
        st.code(MODEL_PATH.read_text(encoding="utf-8"), language="json")

    st.download_button(
        "Download current config",
        data=json.dumps(config.to_dict(), indent=2),
        file_name="linear_regression_config.json",
        mime="application/json",
        use_container_width=True,
    )
