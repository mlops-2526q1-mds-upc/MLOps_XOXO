import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path

# Paths
REFERENCE_FILE = Path("model_monitoring/reference_data.csv")
LOG_FILE = Path("model_monitoring/prediction_logs.csv")
REPORT_DIR = Path("model_monitoring/reports") 

def generate_report():
    if not LOG_FILE.exists():
        print("No current data to analyze.")
        return

    # Load data
    reference = pd.read_csv(REFERENCE_FILE)
    current = pd.read_csv(LOG_FILE)
    
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Authenticity Model (real/fake) Report
    ref_auth = reference[reference['model'] == 'authenticity_model']
    curr_auth = current[current['model'] == 'authenticity_model']

    if not curr_auth.empty:
        # Check drift for the 'prediction' column (Real vs Fake)
        auth_report = Report(metrics=[
            DataDriftPreset(columns=['prediction'])
        ])
        
        auth_report.run(reference_data=ref_auth, current_data=curr_auth)
        
        output_path = REPORT_DIR / "authenticity_drift_report.html"
        auth_report.get_html(str(output_path))
        print(f"Authenticity report generated: {output_path}")
    else:
        print("No authenticity predictions found in logs.")

    # Age/Gender Model Report
    
    # Filter for gender model logs (contains 'Male'/'Female' predictions)
    ref_gender = reference[reference['model'] == 'gender_model']
    curr_gender = current[current['model'] == 'gender_model']
    
    # Filter for age model logs (contains numeric age predictions)
    ref_age = reference[reference['model'] == 'age_model']
    curr_age = current[current['model'] == 'age_model']

    if not curr_gender.empty and not curr_age.empty:
        # Gender Report
        gender_report = Report(metrics=[
            DataDriftPreset(columns=['prediction']) # Checks 'Male' vs 'Female' distribution
        ])
        gender_report.run(reference_data=ref_gender, current_data=curr_gender)
        gender_output = REPORT_DIR / "gender_drift_report.html"
        gender_report.get_html(str(gender_output))
        print(f"Gender drift report generated: {gender_output}")

        # Age Report
        # Ensure 'prediction' column is treated as numeric for age
        ref_age_numeric = ref_age.copy()
        curr_age_numeric = curr_age.copy()
        ref_age_numeric['prediction'] = pd.to_numeric(ref_age_numeric['prediction'])
        curr_age_numeric['prediction'] = pd.to_numeric(curr_age_numeric['prediction'])

        age_report = Report(metrics=[
            DataDriftPreset(columns=['prediction']) # Checks numerical distribution of ages
        ])
        age_report.run(reference_data=ref_age_numeric, current_data=curr_age_numeric)
        age_output = REPORT_DIR / "age_drift_report.html"
        age_report.get_html(str(age_output))
        print(f"Age drift report generated: {age_output}")

    else:
        print("No Age/Gender predictions found in logs. Skipping.")

if __name__ == "__main__":
    generate_report()