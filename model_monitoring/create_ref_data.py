import pandas as pd

# Create a synthetic reference dataset representing the 'normal' distribution of predicted values
reference_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
    'model': ['gender_model'] * 500 + ['authenticity_model'] * 500,
    'prediction': ['Male'] * 250 + ['Female'] * 250 + ['Real'] * 350 + ['Fake'] * 150, # 50% gender split, 70% real
    'confidence': [0.6] * 1000
})

reference_data.to_csv("model_monitoring/reference_data.csv", index=False)