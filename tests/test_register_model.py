import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Since the target script executes logic immediately upon import, 
# we need to mock out the critical components before we import it.

REGISTER_MODEL_SCRIPT_PATH = 'mlops_xoxo.register_model'

# --- Utility Fixtures ---

@pytest.fixture
def mock_mlflow_components():
    """Mock out all MLflow client methods needed for the script."""
    with patch('mlflow.tracking.MlflowClient', autospec=True) as MockClient, \
         patch('mlflow.register_model', autospec=True) as mock_register_model, \
         patch('mlflow.set_tracking_uri', autospec=True), \
         patch('mlflow.set_experiment', autospec=True):
        
        # Configure the MockClient instance
        mock_client_instance = MockClient.return_value
        
        # Mock experiment details for get_experiment_by_name
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "12345"
        mock_client_instance.get_experiment_by_name.return_value = mock_experiment

        # Mock search_runs result (a list containing one mock run)
        mock_run_info = MagicMock()
        mock_run_info.info.run_id = "MOCK_SEARCHED_RUN_ID"
        mock_client_instance.search_runs.return_value = [mock_run_info]

        # Mock the return value of mlflow.register_model
        mock_register_model.return_value.name = "FaceNetModel"
        mock_register_model.return_value.version = 1
        
        yield {
            'MockClient': MockClient,
            'client_instance': mock_client_instance,
            'mock_register_model': mock_register_model,
        }

@pytest.fixture
def run_script(monkeypatch):
    """
    Utility function to run the script under test.
    We import the script inside the test to allow patching of environment variables beforehand.
    """
    def _run(env_vars):
        # Apply environment variables
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
            
        # Clear environment variable if value is None
        for key, value in env_vars.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
                
        # Patch the print function to capture output for assertions
        with patch('builtins.print') as mock_print:
            # We must remove the module from sys.modules to force a fresh import
            # so that the code inside the `if not run_id:` block executes
            if REGISTER_MODEL_SCRIPT_PATH in sys.modules:
                del sys.modules[REGISTER_MODEL_SCRIPT_PATH]
                
            # Execute the script
            import mlops_xoxo.register_model 
            
            # Return the print mock to check output
            return mock_print
            
    yield _run
    
# --- Tests ---

def test_direct_registration_with_run_id(run_script, mock_mlflow_components):
    """Test the script registers immediately when MLFLOW_RUN_ID is provided."""
    env_vars = {
        "MLFLOW_TRACKING_URI": "http://mock-uri",
        "MLFLOW_RUN_ID": "DIRECT_PROVIDED_RUN_ID",
    }
    
    run_script(env_vars)
    
    # Check that it called the register function
    mock_mlflow_components['mock_register_model'].assert_called_once()
    
    # Check that the correct URI was used (DIRECT_PROVIDED_RUN_ID)
    expected_uri = "runs:/DIRECT_PROVIDED_RUN_ID/model"
    actual_uri, registry_name = mock_mlflow_components['mock_register_model'].call_args[0]
    
    assert actual_uri == expected_uri, "Did not use the run ID provided in environment."
    
    # Check that search_runs was NOT called
    mock_mlflow_components['client_instance'].search_runs.assert_not_called()
    

def test_search_and_register_when_run_id_missing(run_script, mock_mlflow_components):
    """Test the script searches and uses the found run ID when MLFLOW_RUN_ID is missing."""
    env_vars = {
        "MLFLOW_TRACKING_URI": "http://mock-uri",
        "MLFLOW_RUN_ID": None, # Explicitly remove the env variable
    }
    
    mock_print = run_script(env_vars)
    
    # Check that search_runs WAS called
    mock_mlflow_components['client_instance'].search_runs.assert_called_once()
    
    # Check that the register function was called
    mock_mlflow_components['mock_register_model'].assert_called_once()

    # Check that the correct URI was used (MOCK_SEARCHED_RUN_ID)
    expected_uri = "runs:/MOCK_SEARCHED_RUN_ID/model"
    actual_uri, registry_name = mock_mlflow_components['mock_register_model'].call_args[0]
    
    assert actual_uri == expected_uri, "Did not use the run ID found by searching."
    
    # Check warning/info was printed
    output = " ".join(call[0][0] for call in mock_print.call_args_list)
    assert "Using Run ID from the last run found: MOCK_SEARCHED_RUN_ID" in output


def test_error_on_missing_experiment(run_script, mock_mlflow_components):
    """Test the script raises an error if the MLflow experiment cannot be found."""
    env_vars = {
        "MLFLOW_TRACKING_URI": "http://mock-uri",
        "MLFLOW_RUN_ID": None,
        "MLFLOW_EXPERIMENT_NAME": "non_existent_experiment"
    }
    
    # Simulate a missing experiment
    mock_mlflow_components['client_instance'].get_experiment_by_name.return_value = None
    
    # Assert that a ValueError is raised
    with pytest.raises(ValueError, match="MLflow Experiment 'non_existent_experiment' not found."):
        run_script(env_vars)


def test_error_on_no_runs_found(run_script, mock_mlflow_components):
    """Test the script raises an error if MLFLOW_RUN_ID is missing and search_runs returns nothing."""
    env_vars = {
        "MLFLOW_TRACKING_URI": "http://mock-uri",
        "MLFLOW_RUN_ID": None,
    }
    
    # Simulate search_runs returning an empty list
    mock_mlflow_components['client_instance'].search_runs.return_value = []
    
    # Assert that a ValueError is raised
    with pytest.raises(ValueError, match="Could not find any previous runs in the experiment to register the model."):
        run_script(env_vars)


def test_default_experiment_name_used(run_script, mock_mlflow_components):
    """Test that the default experiment name 'default' is used if the env var is missing."""
    env_vars = {
        "MLFLOW_TRACKING_URI": "http://mock-uri",
        "MLFLOW_RUN_ID": None,
        "MLFLOW_EXPERIMENT_NAME": None # Explicitly test the default fallback
    }
    
    # The script should fall back to 'default' and search_runs should be called successfully
    run_script(env_vars)
    
    # Check that search_runs was called (i.e., the default experiment was found)
    mock_mlflow_components['client_instance'].search_runs.assert_called_once()
