"""
Shared training utilities for MLOps_XOXO projects.
Provides standardized setup for:
  - MLflow experiment tracking
  - CodeCarbon emissions tracking
  - Device and seed setup
  - Config I/O and directory structure
  - Unified logging for metrics and params
"""

from __future__ import annotations
import os
import random
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import numpy as np
import mlflow
from codecarbon import EmissionsTracker
from dotenv import load_dotenv


# ============================================================
# 💾 CONFIG HANDLING
# ============================================================

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: str | Path):
    """Save configuration to YAML."""
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


# ============================================================
# ⚙️ DEVICE + SEED SETUP
# ============================================================

def get_device() -> torch.device:
    """
    Priority:
      1. FORCE_DEVICE env var (if set)
      2. CUDA if available
      3. MPS if built & available (macOS Metal)
      4. CPU
    """
    # 1) explicit override from CI / env
    force = os.getenv("FORCE_DEVICE")
    if force:
        print(f"Using device (forced): {force}")
        return torch.device(force)

    # 2) CUDA
    if torch.cuda.is_available():
        print("Using device: cuda")
        return torch.device("cuda")

    # 3) MPS (only if backend exists, is built AND available)
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            print("Using device: mps")
            return torch.device("mps")

    # 4) Fallback to CPU
    print("Using device: cpu")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 📁 OUTPUT DIRECTORY HANDLING
# ============================================================

def prepare_output_dirs(task_name: str) -> tuple[Path, Path]:
    """
    Create and return model and report directories for a given task.
    Returns:
        model_dir, report_dir
    """
    model_dir = Path(f"models/{task_name}")
    report_dir = Path(f"reports/{task_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, report_dir


# ============================================================
# 🧠 MLFLOW SETUP
# ============================================================

def init_mlflow(params: Dict[str, Any]) -> tuple[mlflow, str]:
    """
    Initialize MLflow tracking using environment variables and params.yaml.
    Returns MLflow client and experiment_id.
    """
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    experiment_name = params["mlflow"].get("experiment_name", "default_experiment")
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    return mlflow, experiment_id


def log_params_mlflow(params: Dict[str, Any]):
    """Log a dictionary of parameters to MLflow."""
    for key, val in params.items():
        try:
            mlflow.log_param(key, val)
        except Exception as e:
            print(f"⚠️ Could not log param {key}: {e}")


def log_metrics_mlflow(metrics: Dict[str, float], step: Optional[int] = None):
    """Log a dictionary of metrics to MLflow."""
    for key, val in metrics.items():
        try:
            mlflow.log_metric(key, float(val), step=step)
        except Exception as e:
            print(f"⚠️ Could not log metric {key}: {e}")


# ============================================================
# 🌍 CODECARBON SETUP
# ============================================================

def start_emissions_tracker(task_name: str, output_dir: Path) -> EmissionsTracker:
    """
    Start a CodeCarbon tracker context for emissions tracking.
    Usage:
        with start_emissions_tracker("task_name", Path("reports/task")):
            # training code
    """
    tracker = EmissionsTracker(
        project_name=task_name,
        output_dir=str(output_dir),
        output_file="emissions.csv",
        measure_power_secs=10,
        log_level="error"
    )
    return tracker


# ============================================================
# 🧾 UTILITY LOGGING HELPERS
# ============================================================

def log_run_metadata(model_name: str, task_name: str, extra_tags: Optional[Dict[str, str]] = None):
    """Add standard tags to current MLflow run."""
    tags = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project": "MLOps_XOXO",
        "task": task_name,
        "model": model_name,
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)