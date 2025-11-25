# tests/test_train.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from unittest import mock
import torch

TRAIN_SCRIPT_PATH = "mlops_xoxo.train"

@pytest.fixture
def setup_training_mocks(mocker):
    """
    This central fixture mocks all major components of the train.py script
    to allow for a fast, isolated unit test of the training loop logic.
    """
    # Mock the data loaders to return a single mocked batch
    fake_image_batch = torch.randn(4, 3, 112, 112) # Batch of 4 images
    fake_labels_batch = torch.randint(0, 10, (4,)) # Batch of 4 labels

    mocker.patch(f"{TRAIN_SCRIPT_PATH}.DEVICE", torch.device('cpu'))
    
    mock_train_loader = MagicMock()
    mock_train_loader.__iter__.return_value = [(fake_image_batch, fake_labels_batch)]
    mock_train_loader.__len__.return_value = 1
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.train_loader", mock_train_loader)
    
    mock_val_loader = MagicMock()
    mock_val_loader.__iter__.return_value = [(fake_image_batch, fake_labels_batch)]
    mock_val_loader.__len__.return_value = 1
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.val_loader", mock_val_loader)

    # Mock the models and Optimizer
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.return_value = torch.randn(4, 512) # Mocked embeddings
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.model", mock_model)
    
    mock_arcface = MagicMock(spec=torch.nn.Module)
    mock_arcface.return_value = torch.randn(4, 10).requires_grad_() 
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.arcface", mock_arcface)

    mock_optimizer = MagicMock()
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.optimizer", mock_optimizer)

    # Mock External Services and file input output
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.mlflow")
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.EmissionsTracker")
    mocker.patch("torch.save")
    mocker.patch("pandas.read_csv")

    # Mock constants to speed up the test
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.EPOCHS", 1) 

    # Import the script after all the mocks are in place
    from mlops_xoxo import train
    return train

def test_train_model_workflow(setup_training_mocks):
    """
    Tests the main `train_model` function to ensure the core training
    and validation loop executes and calls the correct functions.
    """
    train_script = setup_training_mocks

    train_script.train_model(run_id="fake_run_id")

    # Check that key functions were called
    train_script.optimizer.step.assert_called_once()
    torch.save.assert_called_once()
    train_script.EmissionsTracker.assert_called_once()

    # Get the names of all metrics that were logged
    logged_metrics = [
        call.args[0] for call in train_script.mlflow.log_metric.call_args_list
    ]
    
    # Assert that the metrics we expect are in the list of logged metrics
    assert 'train_loss' in logged_metrics
    assert 'val_accuracy' in logged_metrics
    assert 'val_top1_nn_accuracy' in logged_metrics