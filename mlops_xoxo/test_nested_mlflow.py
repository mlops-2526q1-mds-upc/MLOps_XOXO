import os
from dotenv import load_dotenv
import mlflow

# --- Load your .env credentials ---
load_dotenv()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
if mlflow_username and mlflow_password:
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

# --- Create a test experiment ---
EXPERIMENT_NAME = "Nested_Run_Test"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Create nested runs ---
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {EXPERIMENT_NAME}")

with mlflow.start_run(run_name="parent_test_run") as parent:
    print(f"Started parent run: {parent.info.run_id}")

    with mlflow.start_run(nested=True, run_name="child_train") as child:
        print(f"Started child run: {child.info.run_id} (parent={child.data.tags.get('mlflow.parentRunId')})")

        for epoch in range(2):
            with mlflow.start_run(nested=True, run_name=f"epoch_{epoch}") as epoch_run:
                print(
                    f"  Epoch run: {epoch_run.info.run_id} (parent={epoch_run.data.tags.get('mlflow.parentRunId')})"
                )
                mlflow.log_metric("train_loss", 0.1 * (epoch + 1), step=epoch)
                mlflow.log_metric("val_loss", 0.2 * (epoch + 1), step=epoch)

print("\n Done. Check the Dagshub MLflow UI for experiment 'Nested_Run_Test'.")
print("Make sure to enable the 'Show nested runs' toggle in the UI to see the hierarchy.")
