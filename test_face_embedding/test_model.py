import pytest
import torch
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops_xoxo.face_embedding.eval import calculate_metrics, save_results, evaluate_model


class TestEvalModel:
    """Test cases for model evaluation functions"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = Mock()
        model.eval.return_value = None
        
        # Mock the forward pass to return random embeddings
        def mock_forward(x):
            batch_size = x.shape[0]
            # Return random embeddings of size 512
            return torch.randn(batch_size, 512)
        
        model.side_effect = mock_forward
        return model
    
    @pytest.fixture
    def mock_manifest(self):
        """Create a mock manifest for testing"""
        return {
            'train': [
                'data/train/person1/img1.jpg',
                'data/train/person1/img2.jpg',
                'data/train/person2/img1.jpg',
                'data/train/person2/img2.jpg'
            ],
            'test': [
                'data/test/person1/img3.jpg',
                'data/test/person2/img3.jpg',
                'data/test/person1/img4.jpg'
            ]
        }
    
    @pytest.fixture
    def mock_transform(self):
        """Create a mock transform"""
        return Mock(return_value=torch.randn(1, 3, 112, 112))
    
    @pytest.fixture
    def mock_device(self):
        """Mock device"""
        return torch.device('cpu')
    
    def test_calculate_metrics_structure(self, mock_model, mock_manifest, mock_transform, mock_device):
        """Test that calculate_metrics returns the expected structure"""
        # Mock PIL Image.open
        with patch('PIL.Image.open') as mock_image_open:
            # Mock image
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            mock_image_open.return_value = mock_image
            
            results = calculate_metrics(mock_model, mock_manifest, mock_transform, mock_device)
        
        # Check structure
        assert 'acc_top1' in results
        assert 'acc_top5' in results
        assert 'total' in results
        assert 'predictions' in results
        
        # Check types
        assert isinstance(results['acc_top1'], float)
        assert isinstance(results['acc_top5'], float)
        assert isinstance(results['total'], int)
        assert isinstance(results['predictions'], list)
        
        # Check values are within expected ranges
        assert 0 <= results['acc_top1'] <= 1
        assert 0 <= results['acc_top5'] <= 1
        assert results['total'] == len(mock_manifest['test'])
    
    def test_calculate_metrics_predictions_format(self, mock_model, mock_manifest, mock_transform, mock_device):
        """Test that predictions have the correct format"""
        with patch('PIL.Image.open') as mock_image_open:
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            mock_image_open.return_value = mock_image
            
            results = calculate_metrics(mock_model, mock_manifest, mock_transform, mock_device)
        
        # Check each prediction has required fields
        for prediction in results['predictions']:
            assert 'image_path' in prediction
            assert 'true_id' in prediction
            assert 'predicted_id' in prediction
            assert 'top5_ids' in prediction
            
            # Check types
            assert isinstance(prediction['image_path'], str)
            assert isinstance(prediction['true_id'], str)
            assert isinstance(prediction['predicted_id'], str)
            assert isinstance(prediction['top5_ids'], list)
            
            # Check top5_ids has at most 5 elements (can be fewer if fewer classes)
            # The test has only 2 classes, so top5 should have 2 elements
            assert len(prediction['top5_ids']) <= 5
            assert len(prediction['top5_ids']) >= 1
    
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('mlflow.log_artifact')
    @patch('mlflow.set_tags')
    def test_save_results(self, mock_set_tags, mock_log_artifact, mock_log_param, mock_log_metric):
        """Test save_results function with MLflow integration"""
        # Create test results
        test_results = {
            'acc_top1': 0.85,
            'acc_top5': 0.95,
            'total': 100,
            'predictions': [
                {
                    'image_path': 'test.jpg',
                    'true_id': 'person1',
                    'predicted_id': 'person1',
                    'top5_ids': ['person1', 'person2', 'person3', 'person4', 'person5']
                }
            ]
        }
        
        # Mock Path and file operations
        with patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'):
            
            save_results(test_results)
        
        # Verify MLflow metrics were logged
        mock_log_metric.assert_any_call('top1_accuracy', 0.85)
        mock_log_metric.assert_any_call('top5_accuracy', 0.95)
        mock_log_param.assert_any_call('eval_total', 100)
        
        # Verify tags were set
        mock_set_tags.assert_called_once()
    
    @patch('mlflow.start_run')
    @patch('mlflow.set_experiment')
    @patch('mlflow.get_experiment_by_name')
    def test_evaluate_model_missing_run_id(self, mock_get_experiment, mock_set_experiment, mock_start_run):
        """Test evaluate_model raises error when run_id is missing"""
        # Mock params to have no run_id
        with patch('mlops_xoxo.face_embedding.eval.params', {'mlflow': {}}):
            with pytest.raises(ValueError, match="Missing parent run_id in params.yaml"):
                evaluate_model()
    
    @patch('mlflow.start_run')
    @patch('mlflow.set_experiment')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlops_xoxo.face_embedding.eval.calculate_metrics')
    @patch('mlops_xoxo.face_embedding.eval.save_results')
    def test_evaluate_model_success(self, mock_save_results, mock_calculate_metrics, 
                                  mock_get_experiment, mock_set_experiment, mock_start_run):
        """Test successful execution of evaluate_model"""
        # Mock experiment
        mock_experiment = Mock()
        mock_experiment.experiment_id = 'test-exp-123'
        mock_get_experiment.return_value = mock_experiment
        
        # Mock params with run_id
        with patch('mlops_xoxo.face_embedding.eval.params', 
                  {'mlflow': {'run_id': 'test-run-123', 'experiment_name': 'test-exp'}}):
            
            # Mock MLflow run context
            mock_run = Mock()
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            
            # Mock results
            mock_results = {'acc_top1': 0.8, 'acc_top5': 0.9, 'total': 50, 'predictions': []}
            mock_calculate_metrics.return_value = mock_results
            
            evaluate_model()
            
            # Verify functions were called
            mock_calculate_metrics.assert_called_once()
            mock_save_results.assert_called_once_with(mock_results)