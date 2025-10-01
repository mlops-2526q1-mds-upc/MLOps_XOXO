# mlops_xoxo/utils/start_pipeline.py
import mlflow
import typer
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent  # adjust to project root
PROJECT_EXPERIMENT_NAME = "face_embeddings"  # or read from params.yaml

def start_pipeline(run_name: str):
    mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact(PROJECT_ROOT_PATH / "dvc.yaml")
        print(run.info.run_id)  # ONLY print the run ID, no emojis or URLs

if __name__ == "__main__":
    typer.run(start_pipeline)