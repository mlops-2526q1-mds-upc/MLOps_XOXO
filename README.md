# MLOps_XOXO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

---

# MLOps pipeline (DVC + MLflow)

This repo contains a complete pipelines for 4 face analysis task : Face Embedding, Fake Classification, Emotion Classification and Gender Age Prediction.

**Face Embedding** trains on the CasioFace dataset (**MobileNetV2 + ArcFace**)
**Fake Classification** trains on the Val dataset (**Mobilenetv3_small  + ArcFace**)
**Emotion Classification** trains on the CasioFace dataset (**Resnet18 + ArcFace**)
**Gender Age Prediction** trains on the UTKFace dataset (**MobileNetV2 + ArcFace**)

## Features

- **Data pipeline**: Ingestion, preprocessing, validation with DVC
- **Model training**: MobileNetV2 + ArcFace with MLflow tracking
- **Model evaluation**: Comprehensive metrics and reporting
- **Reproducibility**: DVC for data versioning and pipeline orchestration
- **Experiment tracking**: MLflow for model versioning and metrics
- **Containerization**: Docker and Docker Compose for easy deployment
- **REST API**: FastAPI for model inference
- **Web UI**: Streamlit interface for interactive predictions
- **Monitoring**: Prometheus + Grafana for metrics and visualization
- **CI/CD**: GitHub Actions for automated testing, training, and deployment
- **Testing**: Pytest with coverage reports and quality checks

The pipeline is reproducible with **DVC** and experiment tracking is handled with **MLflow**.

---

## Project Organization

```
    ├── LICENSE            <- Open-source license if one is chosen
    ├── Makefile           <- Makefile with convenience commands like `make run_pipeline`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default mkdocs project; see www.mkdocs.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── pyproject.toml     <- Project configuration file with package metadata for
    │                         mlops_xoxo and configuration for tools like black
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── param.yml       <- Config file for all prameter and hyperparameter
    │
    ├── dvc.yaml         <- DVC Configuration file
    │
    ├── mlruns       <- Log Mlflow runs
    │
    ├── powermetrics_log.txt       <- Log from Code carbon
    │
    └── mlops_xoxo   <- Source code for use in this project.
        │
        ├── data_ingest.py             <- Script to ingest data
        │
        ├── data_split.py               <- Script to properly split the data into train, validation and test set
        │
        ├── data_valifdate.py               <- Scripts to validate the dataset and generate a report
        │
        ├── train.py              <- Scripts to train the model
        │
        ├── eval.py             <- Code to evaluate the model
        │
        └── test.py                <- Code to perform quality testing
```

## 1. Setup environment

### 1) Clone the repository

```bash
git clone https://github.com/mlops-2526q1-mds-upc/MLOps_XOXO.git
cd <your-repo-directory>
```

### 2) Set up the environment

**Using `uv` (recommended)**

```bash
pip install uv
uv venv
uv sync
```

### 3) Configure credentials for DagsHub / MLflow

Create a `config.local` (for DVC) in side the .dvc folders:

```bash
[core]
    remote = origin

['remote "origin"']
    url = https://dagshub.com/pawarit.jamjod/MLOps_XOXO.dvc
    auth = basic
    user = pawarit.jamjod
    password = <Request to the repository's owner>
```

Create a `.env` (for mlflow) in side the project folders:

```bash
ROOT = /path/to/your/project/root
PYTHONPATH = ${ROOT}

HF_TOKEN = your_hugging_face_token

# Disable MLflow integration in Hugging Face
DISABLE_MLFLOW_INTEGRATION = TRUE

MLFLOW_TRACKING_URI=https://dagshub.com/pawarit.jamjod/MLOps_XOXO.mlflow
MLFLOW_TRACKING_USERNAME= pawarit.jamjod
MLFLOW_TRACKING_PASSWORD=<Request to the repository's owner>
```

### 4) Pull data and artifacts

```bash
dvc pull
```

### 5) (Optional) Adjust parameters

Edit `params.yaml` to tweak hyperparameters (batch size, lr, epochs, margin, device).

### 6) Run the pipeline

```bash
dvc repro
```

See MLflow experiments and runs here in these link:

https://dagshub.com/pawarit.jamjod/MLOps_XOXO.mlflow

---

## 2. Docker Deployment (Recommended for Production)

### Quick Start with Docker Compose

```bash
# Build and start all services (API + UI + Monitoring)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```


### Docker Services

The `docker-compose.yml` includes:

1. **API Service**: FastAPI server with model inference
2. **UI Service**: Streamlit web interface
3. **Prometheus**: Metrics collection and monitoring
4. **Grafana**: Visualization dashboards
5. **Node Exporter**: Hardware metrics (CPU/RAM)

### Model Download Options

Models are automatically downloaded when the container starts if DVC credentials are provided:

**Option 1: Using environment variables**
```bash
export DVC_USER=your_username
export DVC_PASSWORD=your_password
docker-compose up -d
```

**Option 2: Using DVC config file**
The `docker-compose.yml` already mounts `.dvc/config.local` if it exists.

**Option 3: Manual download before Docker**
```bash
# Download models locally first
dvc pull

# Models will be available in persistent volume
docker-compose up -d
```

### Persistent Volumes

Docker uses persistent volumes for:
- **xoface-models**: ML models (persist between restarts)
- **xoface-dvc-config**: DVC configuration
- **prometheus_data**: Prometheus metrics history
- **grafana_data**: Grafana dashboards


---

## 3. API Usage

### FastAPI Endpoints

The REST API provides the following endpoints:

- `POST /predict/face_embedding`: Extract face embeddings
- `POST /predict/emotion`: Classify emotions
- `POST /predict/age_gender`: Predict age and gender
- `POST /predict/fake_detection`: Detect fake/real faces
- `GET /health`: Health check
- `GET /models/status`: Check loaded models

### Example API Request

```python
import requests

# Face embedding extraction
url = "http://10.4.41.80:8000/predict/face_embedding"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
embedding = response.json()["embedding"]

# Emotion classification
url = "http://10.4.41.80:8000/predict/emotion"
response = requests.post(url, files=files)
emotion = response.json()["emotion"]
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://10.4.41.80:8000/docs
- ReDoc: http://10.4.41.80:8000/redoc

---

## 4. Monitoring and Observability

### Prometheus Metrics

The API exposes metrics at `http://10.4.41.80:8000/metrics`:

- Request count and latency
- Model inference time
- Error rates
- Hardware metrics (CPU, RAM)

### Grafana Dashboards

Access Grafana at http://10.4.41.80:3000 (default: admin/admin)

Pre-configured dashboards include:
- API performance metrics
- Model inference latency
- System resource usage
- Request rates and error tracking

### Configuration

Prometheus configuration is in `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'xoface-api'
    static_configs:
      - targets: ['api:8000']
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

---

## 5. CI/CD Pipelines

### GitHub Actions Workflows

The project includes three main workflows:

#### 1. CI Pipeline (`.github/workflows/CI-action.yaml`)

Runs on every push and pull request:

- **Code Quality Checks**:
  - Black formatting
  - Flake8 linting
  - Bandit security scan

- **DVC Validation**:
  - YAML syntax validation
  - DVC pipeline consistency
  - Remote connection tests

- **Unit Tests**:
  - Face embedding tests with pytest
  - Coverage reports
  - Model validation

#### 2. CD Training (`.github/workflows/CD-train.yaml`)

Automated model training and deployment:

- Downloads data from DVC
- Trains face embedding model (2 epochs for CI)
- Pushes trained model back to DVC
- Tracks experiments in MLflow

#### 3. CD Deployment (`.github/workflows/CD-deploy.yaml`)

Automated deployment to production:

- Builds Docker images
- Pushes to Docker Hub
- Deploys to self-hosted runner (VM)
- Starts monitoring stack (Prometheus + Grafana)
- Health checks and verification

### Self-Hosted Runner Setup

For the deployment workflow, configure a self-hosted GitHub Actions runner on your VM:

```bash
# On your VM
cd ~/actions-runner
./config.sh --url https://github.com/your-org/MLOps_XOXO --token YOUR_TOKEN
./run.sh
```

### Required GitHub Secrets

Configure these secrets in your repository settings:

- `DAGSHUB_DVC_USER`: DagsHub username
- `DAGSHUB_DVC_PASS`: DagsHub password
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password/token

---

## 6. Testing

### Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=mlops_xoxo --cov-report=html

# Run specific test suite
pytest test_face_embedding/ -v
```

### Test Structure

```
test_face_embedding/
├── conftest.py           # Test fixtures
├── test_api.py          # API endpoint tests
├── test_data_ingest.py  # Data ingestion tests
├── test_data_split.py   # Data splitting tests
├── test_data_utils.py   # Utility function tests
├── test_data_validate.py # Data validation tests
├── test_model.py        # Model architecture tests
└── test_train.py        # Training pipeline tests
```

---
