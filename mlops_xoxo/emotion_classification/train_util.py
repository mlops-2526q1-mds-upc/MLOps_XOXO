"""Training utilities for emotion classification.

Provides a small subset of helpers used by `train.py` and `eval.py`:
 - prepare_output_dirs
 - init_mlflow / log helpers
 - device + seed helpers
"""
from __future__ import annotations
import os
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import mlflow
from dotenv import load_dotenv

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


def prepare_output_dirs(task_name: str) -> tuple[Path, Path]:
    model_dir = Path(f"models/{task_name}")
    report_dir = Path(f"reports/{task_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, report_dir


def init_mlflow(params: Dict[str, Any]) -> tuple[mlflow, str]:
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    experiment_name = params.get("mlflow", {}).get("experiment_name", "emotion_classification")
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = experiment.experiment_id
    return mlflow, exp_id


def log_params_mlflow(params: Dict[str, Any]):
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            pass


def log_metrics_mlflow(metrics: Dict[str, float], step: Optional[int] = None):
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception:
            pass


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def start_emissions_tracker(task_name: str, output_dir: Path):
    if EmissionsTracker is None:
        class DummyCtx:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return DummyCtx()
    tracker = EmissionsTracker(project_name=task_name, output_dir=str(output_dir), output_file="emissions.csv")
    return tracker
