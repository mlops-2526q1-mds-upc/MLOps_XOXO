import pytest
from pathlib import Path
import sys
import os

# Path to 'eval.py' where we evaluate the model
EVAL_SCRIPT_PATH = 'mlops_xoxo.eval'

"""
We do not use 'test_ds' fixture as described in Pytest demo, as our test
dataset is loaded in 'eval.py' which we test in this 'test_model.py' file.
"""

@pytest.fixture
def pipe(mocker):
    """
    This fixture prepares the environment for the test, sets up the model and
    the data.
    """
    # Mock MLflow calls to avoid depending on a live server
    mocker.patch(f'{EVAL_SCRIPT_PATH}.mlflow')
    
    # Import the eval script. It will load the real model and manifest
    # as defined within the script itself.
    if EVAL_SCRIPT_PATH in sys.modules:
        del sys.modules[EVAL_SCRIPT_PATH]
    import mlops_xoxo.eval as eval_script
    
    return {"script": eval_script}

def test_model_accuracy(pipe):
    """
    Tests if the model's accuracy on the full test set is above a
    reasonable threshold.
    """
    eval_script = pipe["script"]
    
    # Run the model evaluation
    metrics = eval_script.calculate_metrics(
        eval_script.model,
        eval_script.manifest,
        eval_script.transform,
        eval_script.DEVICE
    )

    # Check if accuracy is above 50%
    accuracy_threshold = 0.50
    
    assert metrics is not None, "evaluate_model should return a metrics dictionary."
    assert 'acc_top1' in metrics, "Metrics should contain 'top1_accuracy'."
    assert metrics['acc_top1'] >= accuracy_threshold
    
    print(f"\n Model accuracy is {metrics['acc_top1']:.2f}, which is above the threshold of {accuracy_threshold}.")

@pytest.mark.parametrize("image_path, expected_identity", [
    # Test 1: The first photo of person '0000099'
    ("data/processed/test/0000204/6671.jpg", "0000204"),
    # Test 2: A different photo of the same person
    ("data/processed/test/0000204/6712.jpg", "0000204"),
    # Test 3: A photo of a different person to ensure the model can distinguish them
    ("data/processed/test/0000233/9004.jpg", "0000233"),
])
def test_model_predictions(pipe, image_path, expected_identity):
    """
    Tests the model's prediction for a few specific, known images.
    """
    eval_script = pipe["script"]

    results = eval_script.calculate_metrics(
        eval_script.model,
        eval_script.manifest,
        eval_script.transform,
        eval_script.DEVICE
    )
    
    # Find our specific image in the list of predictions 
    all_predictions = results["predictions"]

    print("\n\n--- 🕵️ DEBUGGING PREDICTION PATHS 🕵️ ---")
    print("Path being searched for in test:", image_path)
    print("\nPaths found in results from manifest.json:")
    if not all_predictions:
        print(" -> The prediction list is EMPTY.")
    else:
        for p in all_predictions:
            print(" ->", p['image_path'])
    print("--- END DEBUGGING ---\n")
    
    # Find the dictionary corresponding to the image_path for this test run
    prediction_for_image = next(
        (p for p in all_predictions if p['image_path'].endswith(image_path)), 
        None
    )

    assert prediction_for_image is not None, f"Prediction for {image_path} not found."
    assert prediction_for_image['predicted_id'] == expected_identity
    
    print(f"\n Model correctly predicted '{image_path}' as '{expected_identity}'.")