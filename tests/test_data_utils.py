import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from mlops_xoxo.utils.data_utils import create_manifest

# --- Fixtures ---

@pytest.fixture(scope="function")
def mock_data_directory():
    """
    Creates a temporary directory simulating the RAW_DIR structure:
    person_a: 10 images (7/2/1 split)
    person_b: 5 images (3/1/1 split)
    total: 15 images
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "raw_data"
        data_dir.mkdir()

        # Person A (10 files)
        person_a = data_dir / "person_a"
        person_a.mkdir()
        for i in range(10):
            (person_a / f"img_{i:02}.jpg").touch()

        # Person B (5 files)
        person_b = data_dir / "person_b"
        person_b.mkdir()
        for i in range(5):
            (person_b / f"img_{i:02}.jpg").touch()

        # Extra file that should be ignored
        (data_dir / "ignore_me.txt").touch()

        yield data_dir

@pytest.fixture(scope="function")
def output_path():
    """Creates a temporary path for the manifest output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "manifest.json"

# --- Tests ---

def test_manifest_total_image_count(mock_data_directory, output_path):
    """Test that all 15 images are included in the manifest splits."""
    create_manifest(mock_data_directory, output_path)
    
    with open(output_path, 'r') as f:
        manifest = json.load(f)
    
    total_in_splits = len(manifest['train']) + len(manifest['val']) + len(manifest['test'])
    assert total_in_splits == 15, f"Expected 15 total images, found {total_in_splits}"


def test_manifest_split_ratios(mock_data_directory, output_path):
    """Test if the default 70/20/10 split is correctly applied."""
    create_manifest(mock_data_directory, output_path)
    
    with open(output_path, 'r') as f:
        manifest = json.load(f)

    # Person A (10 images): 7 train, 2 val, 1 test
    # Person B (5 images): 3 train, 1 val, 1 test
    # TOTAL: 10 train, 3 val, 2 test (10+3+2 = 15)
    
    assert len(manifest['train']) == 10, f"Expected 10 train images, got {len(manifest['train'])}"
    assert len(manifest['val']) == 3, f"Expected 3 val images, got {len(manifest['val'])}"
    assert len(manifest['test']) == 2, f"Expected 2 test images, got {len(manifest['test'])}"


def test_manifest_no_data_leakage_by_person(mock_data_directory, output_path):
    """Test that all images for a given person are in only ONE split (no cross-split leakage)."""
    create_manifest(mock_data_directory, output_path)
    
    with open(output_path, 'r') as f:
        manifest = json.load(f)

    # Collect all unique person IDs found in the train split
    train_person_ids = set(Path(p).parent.name for p in manifest['train'])
    
    # Assert that no person ID from train exists in the val or test splits
    for val_path in manifest['val']:
        assert Path(val_path).parent.name not in train_person_ids, "Data leakage found between TRAIN and VAL."
    
    for test_path in manifest['test']:
        assert Path(test_path).parent.name not in train_person_ids, "Data leakage found between TRAIN and TEST."


def test_manifest_no_shuffle_is_deterministic(mock_data_directory, output_path):
    """Test that when shuffle=False, the split is deterministic and uses file name order."""
    
    # Run 1
    create_manifest(mock_data_directory, output_path, shuffle=False)
    with open(output_path, 'r') as f:
        manifest1 = json.load(f)
        
    # Run 2
    create_manifest(mock_data_directory, output_path, shuffle=False)
    with open(output_path, 'r') as f:
        manifest2 = json.load(f)
        
    # Check that the two unshuffled manifests are identical
    assert manifest1 == manifest2, "Unshuffled split is not deterministic."

    # Explicitly check that the first file is always the same (img_00)
    person_a_train = [p for p in manifest1['train'] if Path(p).parent.name == 'person_a']
    assert Path(person_a_train[0]).name == 'img_00.jpg', "First image is not img_00.jpg when unshuffled."


def test_manifest_custom_splits(mock_data_directory, output_path):
    """Test a custom split ratio (e.g., 50/50)."""
    custom_splits = {'train': 0.5, 'val': 0.5, 'test': 0.0}
    create_manifest(mock_data_directory, output_path, splits=custom_splits)
    
    with open(output_path, 'r') as f:
        manifest = json.load(f)
        
    # Person A (10 images): 5 train, 5 val, 0 test
    # Person B (5 images): 2 train, 3 val, 0 test (integer math: 5*0.5=2, 5*0.5=2.5->3 for remainder)
    # TOTAL: 7 train, 8 val, 0 test
    
    assert len(manifest['train']) == 7, f"Expected 7 train images with 50/50 split, got {len(manifest['train'])}"
    assert len(manifest['val']) == 8, f"Expected 8 val images with 50/50 split, got {len(manifest['val'])}"
    assert len(manifest['test']) == 0, "Expected 0 test images with 50/50 split."
