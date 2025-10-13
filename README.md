# MLOps_XOXO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

---

# CasioFace MLOps pipeline (DVC + MLflow)

This repo contains a complete **face embedding training** on the CasioFace dataset (**MobileNetV2 + ArcFace**):

- Data ingestion
- Data preprocessing
- Data validation
- Model training (MobileNetV2 + ArcFace)
- Model evaluation

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
make run_pipeline
```

See MLflow experiments and runs here in these link:

https://dagshub.com/pawarit.jamjod/MLOps_XOXO.mlflow
