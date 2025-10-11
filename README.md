# MLOps_XOXO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

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

---

# CasioFace MLOps pipeline (DVC + MLflow)

This repo contains a complete **face re-identification pipeline** on the CasioFace dataset:

- Data ingestion
- Face detection + preprocessing
- Data validation
- Model training
- Model evaluation

The pipeline is reproducible with **DVC** and experiment tracking is handled with **MLflow**.

---

## 1. Setup environment

```bash
# clone your repo if not already
git clone <your-repo>
cd <your-repo>

# create environment from conda.yaml
conda env create -f conda.yaml
conda activate casioface-env
```
