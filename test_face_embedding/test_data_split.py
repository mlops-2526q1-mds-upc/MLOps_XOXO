import pytest
from pathlib import Path
import json

DATA_SPLIT_SCRIPT_PATH = "mlops_xoxo.face_embedding.data_split"

@pytest.fixture
def fake_raw_data(tmp_path):
    """Creates a temporary raw data directory with a few images for testing."""
    raw_dir = tmp_path / "raw"
    # Create two people with 10 images each
    for person_id in ["person_a", "person_b"]:
        person_dir = raw_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            (person_dir / f"{i}.jpg").touch()
    return raw_dir

def test_data_split_workflow(mocker, tmp_path, fake_raw_data):
    """
    Tests the entire main workflow of data_split.py.
    """
    # Define temporary output paths for our test
    output_dir = tmp_path / "processed"
    manifest_path = output_dir / "splits" / "manifest.json"

    # Mock the global constants in data_split.py to use our temp paths.
    # This redirects the script's I/O to our temporary test directories.
    mocker.patch(f"{DATA_SPLIT_SCRIPT_PATH}.RAW_DIR", fake_raw_data)
    mocker.patch(f"{DATA_SPLIT_SCRIPT_PATH}.OUTPUT_DIR", output_dir)
    mocker.patch(f"{DATA_SPLIT_SCRIPT_PATH}.MANIFEST_PATH", manifest_path)
    
    from mlops_xoxo.face_embedding import data_split

    # Run the main function of the script
    data_split.split_data()

    # Check that the outputs were created correctly
    assert manifest_path.exists(), "Manifest file was not created."

    # Load the manifest and check its contents
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Check if the splits have the correct number of images (70/20/10 split of 20 total images)
    assert len(manifest["train"]) == 14
    assert len(manifest["val"]) == 4
    assert len(manifest["test"]) == 2
    
    # Check if a path in the manifest points to the new 'processed' directory
    first_train_image = Path(manifest["train"][0])
    assert "processed" in first_train_image.parts
    
    # Check if the actual output image files were created
    assert first_train_image.exists()
    assert (output_dir / "test" / "person_a").exists()