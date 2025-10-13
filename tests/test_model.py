import pytest
import sys
import yaml  
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Markers and constants
EVAL_SCRIPT_PATH = 'mlops_xoxo.eval'
REAL_MANIFEST_PATH = Path("data/processed/splits/manifest.json")
requires_real_data = pytest.mark.skipif(not REAL_MANIFEST_PATH.exists(), reason="Real data not found. Run 'dvc pull'.")

"""
We do not use 'test_ds' fixture as described in Pytest demo, as our test
dataset is loaded in 'eval.py' which we test in this 'test_model.py' file.
"""

### Fixtures

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


### Tests

@requires_real_data
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

@requires_real_data
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
    
    # Find the dictionary corresponding to the image_path for this test run
    prediction_for_image = next(
        (p for p in all_predictions if p['image_path'].endswith(image_path)), 
        None
    )

    assert prediction_for_image is not None, f"Prediction for {image_path} not found."
    assert prediction_for_image['predicted_id'] == expected_identity
    
    print(f"\n Model correctly predicted '{image_path}' as '{expected_identity}'.")

def test_save_results_logs_and_writes_files(mocker):
    """
    Unit test for the save_results function to ensure it calls the
    correct logging and file-writing functions.
    We mock this function not to clutter MLFlow with actual resilts,
    so we just check if the method 'save_results' from eval.py functions correctly.
    """
    from mlops_xoxo import eval as eval_script

    # Create a fake 'results' dictionary to pass to the function
    fake_results = {
        "acc_top1": 0.95,
        "acc_top5": 0.99,
        "total": 100,
        "predictions": [{"image_path": "path/to/img.jpg"}]
    }

    # Mock all the external functions that save_results calls
    mock_mlflow = mocker.patch.object(eval_script, 'mlflow')
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mock_path = mocker.patch.object(eval_script, 'Path')
    
    eval_script.save_results(fake_results)

    # Check that MLflow logging was called with the correct metrics
    mock_mlflow.log_metric.assert_any_call('top1_accuracy', 0.95)
    mock_mlflow.log_metric.assert_any_call('top5_accuracy', 0.99)
    mock_mlflow.log_param.assert_any_call('eval_total', 100)
    
    # Check that the script tried to create the reports directory
    mock_path.return_value.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    # Check that the script tried to write the summary and predictions files
    assert mock_open.call_count >= 2, "Expected at least two files to be opened for writing."
    mock_mlflow.log_artifact.assert_any_call(str(mock_path.return_value / 'eval_summary.txt'))


def test_calculate_metrics_handles_empty_test_set(mocker):
    """Tests that calculate_metrics handles an empty test set gracefully."""
    from mlops_xoxo.eval import calculate_metrics
    mock_model = MagicMock()
    mock_transform = MagicMock()
    fake_manifest = {"train": ["path/to/img.jpg"], "test": []}
    
    mock_model.return_value.detach.return_value.cpu.return_value.numpy.return_value = [np.random.rand(512)]
    mocker.patch("PIL.Image.open")
    
    results = calculate_metrics(mock_model, fake_manifest, mock_transform, "cpu")
    
    assert results["acc_top1"] == 0

def test_main_uses_existing_experiment_id(mocker):
    """Tests the 'else' branch in the evaluate_model function."""
    from mlops_xoxo import eval as eval_script
    mock_params = {
        'mlflow': {'experiment_id': 'id', 'run_id': 'id', 'experiment_name': 'name'},
        'dataset': {'processed_dir': 'data/processed'}
    }
    mocker.patch.object(eval_script, 'params', mock_params)
    mock_mlflow = mocker.patch.object(eval_script, 'mlflow')
    mocker.patch.object(eval_script, 'calculate_metrics', return_value={})
    mocker.patch.object(eval_script, 'save_results')
    mocker.patch('yaml.safe_dump')

    eval_script.evaluate_model()

    mock_mlflow.get_experiment_by_name.assert_not_called()