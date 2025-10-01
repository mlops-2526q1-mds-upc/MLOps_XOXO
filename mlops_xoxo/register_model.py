import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
PROJECT_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'default')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)

# Get inputs (could also be passed via params.yaml or command-line)
run_id = os.getenv("MLFLOW_RUN_ID")  # captured from pipeline
artifact_name = "model"               # or whatever you logged in train
model_registry_name = "FaceNetModel" # desired model registry name

if not run_id:
    raise ValueError("MLFLOW_RUN_ID environment variable is missing or empty. Please provide a valid run ID.")

model_uri = f"runs:/{run_id}/{artifact_name}"

# Register the model
model_details = mlflow.register_model(model_uri, model_registry_name)

print(f"Registered model '{model_details.name}', version {model_details.version}")