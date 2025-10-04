import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
PROJECT_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'default')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)

# Get MLflow client
client = MlflowClient()

# Get inputs (could also be passed via params.yaml or command-line)
run_id = os.getenv("MLFLOW_RUN_ID")  # captured from pipeline
artifact_name = "model"               # or whatever you logged in train
model_registry_name = "FaceNetModel" # desired model registry name

if not run_id:
    print("WARNING: MLFLOW_RUN_ID missing. Searching for the last run in the experiment...")
    
    # Get the current experiment
    experiment = client.get_experiment_by_name(PROJECT_EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"MLflow Experiment '{PROJECT_EXPERIMENT_NAME}' not found.")
        
    # Search for the most recent run (order by start time descending)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs:
        run_id = runs[0].info.run_id
        print(f"Using Run ID from the last run found: {run_id}")
    else:
        raise ValueError("Could not find any previous runs in the experiment to register the model.")

model_uri = f"runs:/{run_id}/{artifact_name}"

# Register the model
model_details = mlflow.register_model(model_uri, model_registry_name)

print(f"Registered model '{model_details.name}', version {model_details.version}")