import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from mlops_xoxo import data_validate

# --- Fixtures for Setup and Teardown ---

@pytest.fixture(scope="function")
def setup_dirs_and_files():
    """Sets up temporary directories and mock files for testing."""
    # Use TemporaryDirectory to handle cleanup automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_raw_dir = Path(temp_dir) / 'raw_data'
        temp_raw_dir.mkdir()

        # Create mock person folders and files
        person_a = temp_raw_dir / 'person_a'
        person_b = temp_raw_dir / 'person_b'
        person_c_empty = temp_raw_dir / 'person_c_empty'

        person_a.mkdir()
        person_b.mkdir()
        person_c_empty.mkdir()

        # Create image files
        (person_a / 'img1.jpg').touch()
        (person_a / 'img2.jpg').touch()
        (person_b / 'img3.jpg').touch()

        # Create a mock manifest file for testing
        temp_manifest_path = Path(temp_dir) / 'manifest.json'
        
        manifest_content = {
            "train": [str(person_a / 'img1.jpg'), str(person_b / 'img3.jpg')],
            "val": [str(person_a / 'img2.jpg')],
            "test": []
        }
        with open(temp_manifest_path, 'w') as f:
            json.dump(manifest_content, f)

        # Yield paths as strings, matching how your script uses them
        yield str(temp_raw_dir), str(temp_manifest_path)

# --- Test check_person_folders ---

def test_check_person_folders_summary(setup_dirs_and_files):
    """Tests if the function correctly counts persons and images."""
    raw_dir, _ = setup_dirs_and_files
    
    summary = data_validate.check_person_folders(raw_dir)
    
    assert summary[0] == "Found 3 persons"
    assert summary[1] == "Total images: 3"
    assert summary[2] == "Average images per person: 1.00"

def test_check_person_folders_warnings(setup_dirs_and_files):
    """Tests if the function correctly raises a warning for empty folders."""
    raw_dir, _ = setup_dirs_and_files
    
    summary = data_validate.check_person_folders(raw_dir)
    
    # Check that the specific warning for the empty folder is present
    assert "Warning: Person person_c_empty has no images" in summary

# --- Test validate_images ---

@patch('mlops_xoxo.data_validate.cv2.imread')
def test_validate_images_corrupted(mock_imread, setup_dirs_and_files):
    """Tests that corrupted images are correctly identified and listed."""
    raw_dir, _ = setup_dirs_and_files
    raw_dir_path = Path(raw_dir) # Use Path object internally for easier sorting

    mock_imread.side_effect = [
        MagicMock(),  # 1st file (person_a/img1.jpg) is valid
        None,         # 2nd file (person_a/img2.jpg) is corrupted (returns None)
        MagicMock(),  # 3rd file (person_b/img3.jpg) is valid
    ]
    
    corrupted = data_validate.validate_images(raw_dir)

    assert len(corrupted) == 1, "Expected 1 corrupted image to be reported."

    expected_corrupted_path = str(raw_dir_path / 'person_a' / 'img2.jpg')

    assert any("person_a/img2.jpg" in p for p in corrupted), "Did not find the expected corrupted file path."

# --- Test check_manifest ---

def test_check_manifest_success(setup_dirs_and_files):
    """Tests manifest with all splits present."""
    _, manifest_path = setup_dirs_and_files
    
    lines = data_validate.check_manifest(manifest_path)
    
    assert lines[0] == "train: 2 images"
    assert lines[1] == "val: 1 images"
    assert lines[2] == "test: 0 images"

def test_check_manifest_not_found():
    """Tests the case where the manifest file does not exist."""
    missing_path = "/nonexistent/path/to/manifest.json"
    lines = data_validate.check_manifest(missing_path)
    
    assert lines == ["Manifest not found"]