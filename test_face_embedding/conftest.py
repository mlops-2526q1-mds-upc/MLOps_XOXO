import pytest
import yaml
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def setup_global_params(session_mocker):
    """
    This autouse, session-scoped fixture mocks yaml.safe_load for the
    entire test session, preventing import-time errors in any test file.
    """
    mock_params = {
        'mlflow': {'run_name': 'test_run', 'run_id': 'fake_run_id'},
        'dataset': {'processed_dir': 'data/processed'},
        'training': {
            'device': 'cpu', 'batch_size': 32, 'epochs': 1, 
            'lr': 0.001, 'margin': 0.5, 'weight_decay': 0.0
        }
    }
    session_mocker.patch('yaml.safe_load', return_value=mock_params)
