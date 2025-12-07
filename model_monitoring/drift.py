import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from pathlib import Path

# 1. Load Reference Data (Embeddings from Training)
# You would ideally save a sample of training embeddings to a CSV
reference_data = pd.read_csv("data/processed/reference_embeddings.csv") 

# 2. Load Current Data (Recent API Logs)
# In production, you'd log API requests to a file/DB. 
# Here we simulate it.
current_data = pd.read_csv("logs/current_embeddings.csv")

def generate_drift_report():
    # Create a report
    report = Report(metrics=[
        DataDriftPreset(), # Checks drift for all columns
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save as HTML for visual inspection
    report.save_html("docs/drift_report.html")
    
    # Return JSON for Prometheus
    return report.json()

if __name__ == "__main__":
    result = generate_drift_report()
    drift_score = json.loads(result)['metrics'][0]['result']['dataset_drift']
    print(f"Dataset Drift Detected: {drift_score}")
```

#### 3. Integrate with Your API
To make this real-time, you need to log data from your API.

**In `api/api.py`:**
Every time `/predict_embedding` is called, save the resulting embedding to a log file (`logs/current_embeddings.csv`).

```python
# In api/api.py predict_embedding function
import csv

# ... generate embedding ...

# Log it for monitoring
with open("logs/current_embeddings.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(embedding_list)
```

#### 4. Expose Metrics to Prometheus
You can create a new endpoint in your API, or a background task, that runs the Evidently report periodically and updates a Prometheus Gauge.

```python
from prometheus_client import Gauge

# Define a new metric
DRIFT_GAUGE = Gauge('model_data_drift', 'Current data drift score')

@app.get("/monitoring/drift")
def calculate_drift():
    # Run Evidently calculation
    # Update the Gauge
    # This is heavy, so run it rarely (e.g., once an hour)
    pass