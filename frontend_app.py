"""Streamlit frontend for FLEX-Persona experiment control and report viewing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed

st.set_page_config(page_title="FLEX-Persona Lab", page_icon="F", layout="wide")

st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(circle at 10% 20%, #fef2d8 0%, #f9e1c8 35%, #e8f0e9 100%);
      }
      .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #123a5e 0%, #0c6e6e 45%, #f76c5e 100%);
        color: #fff7ec;
        box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        margin-bottom: 1.2rem;
      }
      .hero h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: 0.02em;
      }
      .hero p {
        margin-top: 0.6rem;
        font-size: 1.02rem;
      }
      .metric-card {
        padding: 0.9rem;
        border-radius: 12px;
        background: #fff7ee;
        border: 1px solid #e9d2bd;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>FLEX-Persona Control Desk</h1>
      <p>Run heterogeneous federated experiments, inspect robustness metrics, and verify communication efficiency.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Experiment Setup")
    dataset = st.selectbox("Dataset", ["femnist", "cifar100"], index=0)
    experiment_name = st.text_input("Experiment Name", value="flex_persona_frontend")
    rounds = st.slider("Rounds", min_value=1, max_value=10, value=1)
    num_clients = st.slider("Number of Clients", min_value=3, max_value=5, value=4)
    local_epochs = st.slider("Local Epochs", min_value=1, max_value=3, value=1)
    cluster_epochs = st.slider("Cluster-Aware Epochs", min_value=1, max_value=3, value=1)
    batch_size = st.number_input("Batch Size", min_value=16, max_value=4096, value=128, step=16)
    max_samples_per_client = st.number_input(
      "Max Samples Per Client (for quick testing)",
      min_value=64,
      max_value=10000,
      value=512,
      step=64,
    )

with right:
    st.subheader("Optimization")
    learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
    lambda_cluster = st.number_input(
        "Cluster Regularization Lambda", min_value=0.0, max_value=5.0, value=0.1, format="%.3f"
    )
    num_clusters = st.slider("Spectral Clusters", min_value=2, max_value=5, value=2)

run_button = st.button("Run FLEX-Persona Experiment", type="primary", use_container_width=True)

if run_button:
    try:
        config = ExperimentConfig(experiment_name=experiment_name, dataset_name=dataset)
        config.num_clients = num_clients
        config.training.rounds = rounds
        config.training.local_epochs = local_epochs
        config.training.cluster_aware_epochs = cluster_epochs
        config.training.batch_size = int(batch_size)
        config.training.max_samples_per_client = int(max_samples_per_client)
        config.training.learning_rate = learning_rate
        config.training.lambda_cluster = lambda_cluster
        config.clustering.num_clusters = num_clusters

        if dataset == "femnist":
            config.model.num_classes = 62
        else:
            config.model.num_classes = 100

        config.validate()
        set_global_seed(config.random_seed)

        with st.spinner("Running federated simulation. This may take a while on full FEMNIST."):
            simulator = FederatedSimulator(workspace_root=WORKSPACE_ROOT, config=config)
            history = simulator.run_experiment()
            report = simulator.build_report(history)

        report_dir = WORKSPACE_ROOT / config.output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{config.experiment_name}_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        st.success(f"Run complete. Report saved at: {report_path}")

        final_metrics = report.get("final_metrics", {})
        communication = report.get("communication", {})
        convergence = report.get("convergence", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Client Accuracy", f"{final_metrics.get('mean_client_accuracy', 0.0):.4f}")
        c2.metric("Worst Client Accuracy", f"{final_metrics.get('worst_client_accuracy', 0.0):.4f}")
        c3.metric("Worst Group Accuracy", f"{final_metrics.get('worst_group_accuracy', 0.0):.4f}")

        st.markdown("### Communication Overhead")
        st.json(communication)

        st.markdown("### Convergence")
        if convergence:
            st.line_chart(convergence)
        else:
            st.info("No convergence traces available yet.")

        with st.expander("Raw Report JSON", expanded=False):
            st.json(report)

    except Exception as exc:
        st.error(f"Experiment failed: {exc}")
