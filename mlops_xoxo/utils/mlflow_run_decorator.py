from functools import wraps
import mlflow
import yaml
import os

# Load MLFLOW_RUN_ID from env, strip extra whitespace
parent_run_id = os.getenv("MLFLOW_RUN_ID", "").strip()

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROJECT_EXPERIMENT_NAME = params["mlflow"]["experiment_name"]

def mlflow_run(wrapped_function):
    @wraps(wrapped_function)
    def wrapper(*args, **kwargs):
        mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)
        # Attach to parent if MLFLOW_RUN_ID exists
        with mlflow.start_run(run_name=wrapped_function.__name__,
                              nested=True,
                              run_id=parent_run_id if parent_run_id else None):
            return wrapped_function(*args, **kwargs)
    return wrapper