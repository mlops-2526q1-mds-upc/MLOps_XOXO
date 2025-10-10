import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from mlops_xoxo import data_split

# --- Fixtures ---

@pytest.fixture(scope="function")
def setup_split_environment():
    """Creates temporary directories and a mock manifest file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Setup Mock Raw Data Structure
        raw_dir = temp_dir_path / 'data' / 'raw'
        raw_dir.mkdir(parents=True)
        
        # Create mock person folders and files that will be referenced in the manifest
        person_a_dir = raw_dir / 'person_a'
        person_b_dir = raw_dir / 'person_b'
        person_a_dir.mkdir(parents=True)
        person_b_dir.mkdir(parents=True)
        
        (person_a_dir / 'img1.jpg').touch()
        (person_a_dir / 'img2.jpg').touch()
        (person_b_dir / 'img3.jpg').touch()
        
        # Setup Mock Output Directory and Manifest Path
        output_dir = temp_dir_path / 'data' / 'processed'
        output_dir.mkdir(parents=True)
        
        manifest_path = temp_dir_path / 'manifest.json'
        
        # Create the mock manifest content
        manifest_content = {
            "train": [
                str(person_a_dir / 'img1.jpg'),
                str(person_b_dir / 'img3.jpg')
            ],
            "val": [
                str(person_a_dir / 'img2.jpg')
            ],
            "test": []
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_content, f)

        # Yield all necessary paths (as Path objects for convenience in the test)
        yield raw_dir, output_dir, manifest_path

# --- Tests ---

@patch('mlops_xoxo.data_split.shutil.copy')
def test_split_data_copies_correctly(mock_copy, setup_split_environment):
    """Tests if images are copied the correct number of times and to the right targets."""
    raw_dir, output_dir, manifest_path = setup_split_environment
    
    data_split.split_data(raw_dir, manifest_path, output_dir)
    
    # Total calls to shutil.copy
    # We expect 3 calls (2 for train, 1 for val, 0 for test)
    assert mock_copy.call_count == 3, "Expected 3 total image copy operations."

    # Verify the destination paths are correct for all splits
    # Expected Train Destinations
    train_dest_1 = output_dir / 'train' / 'person_a' / 'img1.jpg'
    train_dest_2 = output_dir / 'train' / 'person_b' / 'img3.jpg'
    
    # Expected Val Destination
    val_dest = output_dir / 'val' / 'person_a' / 'img2.jpg'
    
    # Extract all destination arguments from the mock calls
    copied_destinations = [call_args[0][1] for call_args in mock_copy.call_args_list]

    assert str(train_dest_1) in copied_destinations
    assert str(train_dest_2) in copied_destinations
    assert str(val_dest) in copied_destinations
    
    
def test_split_data_creates_directories(setup_split_environment):
    """Tests if the necessary split and person subdirectories are created."""
    raw_dir, output_dir, manifest_path = setup_split_environment
    
    # Run the splitting function
    data_split.split_data(raw_dir, manifest_path, output_dir)

    # Check for the existence of the expected directories
    assert (output_dir / 'train').is_dir()
    assert (output_dir / 'val').is_dir()
    assert (output_dir / 'test').is_dir()
    assert (output_dir / 'train' / 'person_a').is_dir()
    assert (output_dir / 'train' / 'person_b').is_dir()
    assert (output_dir / 'val' / 'person_a').is_dir()
    
    # Since person_b had no 'val' images, its directory shouldn't exist in 'val'
    assert not (output_dir / 'val' / 'person_b').is_dir()


@patch('mlops_xoxo.data_split.create_manifest')
def test_manifest_function_calls_create_manifest(mock_create_manifest, setup_split_environment):
    """Tests if the convenience function 'manifest' correctly calls the utility function."""
    raw_dir, _, manifest_path = setup_split_environment
    
    # The default arguments in data_split.py are used, but we pass the fixture paths for clarity
    data_split.manifest(raw_dir, manifest_path)
    
    # Check that create_manifest was called once with the correct arguments
    mock_create_manifest.assert_called_once_with(raw_dir, manifest_path)
