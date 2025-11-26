# tests/test_train.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from unittest import mock
import torch

TRAIN_SCRIPT_PATH = "mlops_xoxo.face_embedding.train"

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

    # Mock the model classes to return simple mock objects
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.MobileFace")
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.ArcFaceHead")
    
    # Mock torch components
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.torch.optim.Adam")
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.torch.nn.CrossEntropyLoss")

    # Mock External Services and file input output
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.mlflow")
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.start_emissions_tracker")
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.log_metrics_mlflow")
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.log_params_mlflow")
    mocker.patch("torch.save")
    mocker.patch("pandas.read_csv")

    # Mock constants to speed up the test
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.EPOCHS", 1)
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.BATCH", 4)
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.LR", 0.001)
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.MARGIN", 0.5)
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.WEIGHT_DECAY", 0.0001)

    # Mock the dataset to return a fixed number of classes
    mock_train_ds = MagicMock()
    mock_train_ds.classes = list(range(10))  # 10 classes
    mocker.patch(f"{TRAIN_SCRIPT_PATH}.train_ds", mock_train_ds)

    # Import the script after all the mocks are in place
    from mlops_xoxo.face_embedding import train
    return train

def test_train_model_workflow(setup_training_mocks):
    """
    Tests the main `train_model` function to ensure the core training
    and validation loop executes and calls the correct functions.
    """
    train_script = setup_training_mocks

    # Mock the context manager for emissions tracker
    mock_emissions_context = MagicMock()
    train_script.start_emissions_tracker.return_value.__enter__.return_value = mock_emissions_context

    # Create tensors that require gradients for proper backprop
    mock_embeddings = torch.randn(4, 512, requires_grad=True)
    mock_logits = torch.randn(4, 10, requires_grad=True)
    
    # Mock model instances with proper gradient setup
    mock_model_instance = MagicMock()
    mock_model_instance.return_value = mock_embeddings
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.train.return_value = None
    mock_model_instance.eval.return_value = None
    mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    train_script.MobileFace.return_value = mock_model_instance

    mock_arcface_instance = MagicMock()
    mock_arcface_instance.return_value = mock_logits
    mock_arcface_instance.to.return_value = mock_arcface_instance
    mock_arcface_instance.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    train_script.ArcFaceHead.return_value = mock_arcface_instance

    # Mock optimizer
    mock_optimizer_instance = MagicMock()
    train_script.torch.optim.Adam.return_value = mock_optimizer_instance

    # Mock criterion to return a loss tensor that requires grad
    mock_loss_tensor = torch.tensor(0.5, requires_grad=True)
    mock_criterion_instance = MagicMock()
    mock_criterion_instance.return_value = mock_loss_tensor
    train_script.torch.nn.CrossEntropyLoss.return_value = mock_criterion_instance

    train_script.train_model(run_id="fake_run_id")

    # Check that key functions were called
    train_script.MobileFace.assert_called_once()
    train_script.ArcFaceHead.assert_called_once()
    train_script.torch.optim.Adam.assert_called_once()
    mock_optimizer_instance.step.assert_called()
    torch.save.assert_called_once()
    train_script.start_emissions_tracker.assert_called_once()

    # Check that metrics were logged via log_metrics_mlflow
    train_script.log_metrics_mlflow.assert_called()

    # Get the metrics that were passed to log_metrics_mlflow
    logged_metrics_calls = train_script.log_metrics_mlflow.call_args_list
    
    # Check that we have training metrics
    assert len(logged_metrics_calls) > 0
    
    # Check the first call to log_metrics_mlflow contains expected metrics
    first_metrics_call = logged_metrics_calls[0][0][0]  # First call, first arg (metrics dict)
    expected_metrics = ['train_loss', 'val_loss', 'val_accuracy']
    
    for metric in expected_metrics:
        assert metric in first_metrics_call, f"Metric {metric} not found in logged metrics"