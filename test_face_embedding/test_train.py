import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops_xoxo.face_embedding.train import (
    MobileFace, 
    ArcFaceHead, 
    FaceDataset, 
    train_model
)


class TestTrainComponents:
    """Test cases for training components"""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings (already flattened)"""
        return torch.randn(2, 512)  # batch_size=2, embedding_size=512
    
    @pytest.fixture
    def mock_labels(self):
        """Create mock labels"""
        return torch.tensor([0, 1])
    
    def test_mobileface_forward(self):
        """Test MobileFace model forward pass"""
        model = MobileFace(emb_size=512)
        
        # Create proper input for MobileNetV2 (batch_size=2, channels=3, height=112, width=112)
        input_tensor = torch.randn(2, 3, 112, 112)
        output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (2, 512)  # batch_size=2, embedding_size=512
        # Check output is normalized (L2 norm ≈ 1)
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_arcface_forward(self, mock_embeddings, mock_labels):
        """Test ArcFaceHead forward pass"""
        num_classes = 10
        arcface = ArcFaceHead(emb_size=512, num_classes=num_classes, margin=0.5)
        
        logits = arcface(mock_embeddings, mock_labels)
        
        # Check output shape
        assert logits.shape == (2, num_classes)  # batch_size=2, num_classes=10
    
    def test_arcface_parameter_initialization(self):
        """Test ArcFace weight initialization"""
        num_classes = 5
        emb_size = 128
        arcface = ArcFaceHead(emb_size=emb_size, num_classes=num_classes)
        
        # Check weight shape
        assert arcface.weight.shape == (num_classes, emb_size)
        # Check weight is a parameter
        assert isinstance(arcface.weight, torch.nn.Parameter)
    
    @pytest.fixture
    def mock_dataset_dir(self):
        """Create a temporary directory with mock dataset structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create class directories and dummy images
            for class_name in ['person1', 'person2', 'person3']:
                class_dir = Path(temp_dir) / class_name
                class_dir.mkdir()
                
                # Create dummy image files
                for i in range(3):
                    img_path = class_dir / f'img{i}.jpg'
                    # Create empty file (in real scenario, these would be valid images)
                    img_path.touch()
            
            yield temp_dir
    
    def test_facedataset_initialization(self, mock_dataset_dir):
        """Test FaceDataset initialization"""
        dataset = FaceDataset(mock_dataset_dir)
        
        # Check dataset properties
        assert len(dataset) == 9  # 3 classes * 3 images each
        assert len(dataset.classes) == 3
        assert 'person1' in dataset.class_to_idx
        assert 'person2' in dataset.class_to_idx
        assert 'person3' in dataset.class_to_idx
    
    def test_facedataset_getitem(self, mock_dataset_dir):
        """Test FaceDataset __getitem__ method"""
        # Mock transform
        mock_transform = Mock(side_effect=lambda x: x)
        
        dataset = FaceDataset(mock_dataset_dir, transform=mock_transform)
        
        # Test getting an item
        with patch('PIL.Image.open') as mock_image_open:
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            mock_image_open.return_value = mock_image
            
            image, label = dataset[0]
        
        # Check types
        assert isinstance(label, int)
        assert 0 <= label < len(dataset.classes)
    
    def test_facedataset_class_indices(self, mock_dataset_dir):
        """Test that class_indices is properly populated"""
        dataset = FaceDataset(mock_dataset_dir)
        
        # Check class_indices structure
        assert len(dataset.class_indices) == 3
        for class_idx in dataset.class_indices:
            indices = dataset.class_indices[class_idx]
            assert len(indices) == 3  # 3 images per class
            assert all(0 <= idx < len(dataset) for idx in indices)


class TestTrainIntegration:
    """Integration tests for training process"""
    
    @patch('mlops_xoxo.face_embedding.train.prepare_output_dirs')
    @patch('mlops_xoxo.face_embedding.train.start_emissions_tracker')
    @patch('mlops_xoxo.face_embedding.train.log_metrics_mlflow')
    @patch('mlops_xoxo.face_embedding.train.log_params_mlflow')
    @patch('pandas.read_csv')
    @patch('mlops_xoxo.face_embedding.train.DEVICE', torch.device('cpu'))
    def test_train_model_integration(self, mock_read_csv, mock_log_params, mock_log_metrics,
                                mock_emissions_tracker, mock_prepare_dirs):
        """Test the main train_model function with mocked dependencies - simpler version"""
        # Mock dependencies
        mock_prepare_dirs.return_value = (Path('/mock/model/dir'), Path('/mock/report/dir'))
        
        # Mock emissions data
        mock_emissions_df = Mock()
        mock_emissions_data = Mock()
        mock_emissions_data.emissions = 0.1
        mock_emissions_data.energy_consumed = 0.5
        mock_emissions_df.tail.return_value = Mock(**{'iloc': [mock_emissions_data]})
        mock_read_csv.return_value = mock_emissions_df
        
        # Mock emissions tracker context manager
        mock_emissions_tracker.return_value.__enter__ = Mock(return_value=None)
        mock_emissions_tracker.return_value.__exit__ = Mock(return_value=None)
        
        # Mock everything at a higher level
        with patch('mlops_xoxo.face_embedding.train.MobileFace') as mock_mobileface, \
            patch('mlops_xoxo.face_embedding.train.ArcFaceHead') as mock_arcface, \
            patch('mlops_xoxo.face_embedding.train.torch.optim.Adam') as mock_optimizer, \
            patch('mlops_xoxo.face_embedding.train.torch.nn.CrossEntropyLoss') as mock_criterion, \
            patch('mlops_xoxo.face_embedding.train.train_ds') as mock_train_ds, \
            patch('mlops_xoxo.face_embedding.train.train_loader') as mock_train_loader, \
            patch('mlops_xoxo.face_embedding.train.val_loader') as mock_val_loader, \
            patch('torch.save'), \
            patch('mlflow.log_artifact'), \
            patch('mlflow.pytorch.log_model'), \
            patch('mlflow.log_metrics'), \
            patch('mlflow.log_param'):
            
            # Mock the training loop execution
            with patch.object(mock_train_loader, '__iter__') as mock_train_iter, \
                patch.object(mock_val_loader, '__iter__') as mock_val_iter:
                
                # Set up mock dataset
                mock_train_ds.classes = ['person1', 'person2', 'person3']
                
                # Create simple tensor batches
                train_batch = [(torch.randn(2, 3, 112, 112), torch.tensor([0, 1]))]
                val_batch = [(torch.randn(2, 3, 112, 112), torch.tensor([0, 1]))]
                
                mock_train_iter.return_value = iter(train_batch)
                mock_val_iter.return_value = iter(val_batch)
                mock_train_loader.__len__ = Mock(return_value=1)
                mock_val_loader.__len__ = Mock(return_value=1)
                
                # Mock models to return proper tensors
                mock_model = Mock()
                mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
                mock_model.to.return_value = mock_model
                mock_model.train.return_value = None
                mock_model.eval.return_value = None
                mock_model.return_value = torch.randn(2, 512)  # embeddings
                mock_mobileface.return_value = mock_model
                
                mock_arcface_instance = Mock()
                mock_arcface_instance.parameters.return_value = [torch.randn(10, requires_grad=True)]
                mock_arcface_instance.to.return_value = mock_arcface_instance
                mock_arcface_instance.return_value = torch.randn(2, 3)  # logits for 3 classes
                mock_arcface.return_value = mock_arcface_instance
                
                # Mock optimizer and criterion
                mock_optim_instance = Mock()
                mock_optimizer.return_value = mock_optim_instance
                mock_criterion.return_value = Mock(return_value=torch.tensor(0.5))
                
                # Mock tqdm to avoid progress bar issues
                with patch('mlops_xoxo.face_embedding.train.tqdm') as mock_tqdm:
                    mock_tqdm.return_value.__enter__ = Mock(return_value=iter(train_batch))
                    mock_tqdm.return_value.__exit__ = Mock(return_value=None)
                    
                    # Call train_model with minimal epochs
                    with patch('mlops_xoxo.face_embedding.train.EPOCHS', 1):
                        train_model()
        
        # Verify MLflow functions were called
        mock_log_params.assert_called_once()
        mock_log_metrics.assert_called()