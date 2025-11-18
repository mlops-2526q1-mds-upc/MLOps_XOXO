"""
config.py - Load environment variables
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project paths
ROOT = Path(os.getenv('ROOT', '/home/mathys/Documents/MLOps_XOXO'))
PYTHONPATH = os.getenv('PYTHONPATH', str(ROOT))

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Hugging Face
HF_TOKEN = os.getenv('HF_TOKEN')
DISABLE_MLFLOW_INTEGRATION = os.getenv('DISABLE_MLFLOW_INTEGRATION', 'TRUE')


