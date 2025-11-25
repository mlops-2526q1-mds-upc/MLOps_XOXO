import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from mlops_xoxo import data_validate

REAL_RAW_DIR = Path("data/raw")
REAL_MANIFEST_PATH = Path("data/processed/splits/manifest.json")

# This marker skips the tests if the real data is not found
requires_real_data = pytest.mark.skipif(
    not REAL_RAW_DIR.exists(), 
    reason="Real data not found. Run 'dvc pull' first."
)

@requires_real_data
def test_check_person_folders_on_real_data():
    """
    Tests that the folder check runs on real data and finds a reasonable
    number of persons and images
    """
    summary = data_validate.check_person_folders(REAL_RAW_DIR)
    
    # Assertions are general because the exact numbers might change if data is updated
    assert "Found" in summary[0]
    assert "Total images" in summary[1]
    
    num_persons = int(summary[0].split(" ")[1])
    num_images = int(summary[1].split(": ")[1])
    
    assert num_persons > 10, "Expected to find a significant number of person folders."
    assert num_images > 100, "Expected to find a significant number of images."

@requires_real_data
def test_validate_images_on_real_data():
    """
    Tests the integrity of the real images using cv2.imread
    """
    corrupted_files = data_validate.validate_images(REAL_RAW_DIR)
    
    # Assert that the dataset does not have corrupted images
    assert len(corrupted_files) == 0, f"Found {len(corrupted_files)} corrupted images: {corrupted_files}"

@requires_real_data
def test_check_manifest_on_real_data():
    """
    Tests that the rel manifest file is valid and contains entries for all splits
    """
    lines = data_validate.check_manifest(REAL_MANIFEST_PATH)
    
    assert len(lines) == 3, "Manifest check should return 3 lines for train/val/test."
    
    # Check that each split has a non-zero number of images.
    train_count = int(lines[0].split(": ")[1].split(" ")[0])
    val_count = int(lines[1].split(": ")[1].split(" ")[0])
    test_count = int(lines[2].split(": ")[1].split(" ")[0])
    
    assert train_count > 0
    assert val_count > 0
    assert test_count > 0

@requires_real_data
def test_main_block_creates_report(mocker, tmp_path):
    """
    Tests the main execution block of the script to ensure it runs
    and creates the final report file.
    """
    # Mock the output path to a temporary directory to keep the project clean!
    temp_report_path = tmp_path / "data_validation_report.txt"
    mocker.patch.object(data_validate, 'report_path', temp_report_path)
    
    report_lines = []
    report_lines.append("=== Person folder check ===")
    report_lines += data_validate.check_person_folders(data_validate.RAW_DIR)
    report_lines.append("\n=== Image integrity check ===")
    corrupted = data_validate.validate_images(data_validate.RAW_DIR)
    if not corrupted:
        report_lines.append("All images are valid!")
    else:
        report_lines.append(f"Corrupted images: {len(corrupted)}")
        report_lines += corrupted
    report_lines.append("\n=== Manifest check ===")
    report_lines += data_validate.check_manifest(data_validate.MANIFEST_PATH)

    with open(temp_report_path, "w") as f:
        f.write("\n".join(report_lines))

    assert temp_report_path.exists()
    assert temp_report_path.read_text() != ""

# Unit tests for edge cases

def test_check_person_folders_handles_empty_folder(tmp_path):
    """Unit test to ensure empty person folders are correctly reported"""
    # Setup a fake directory with one empty folder
    person_a_dir = tmp_path / "person_a"
    person_a_dir.mkdir()
    (person_a_dir / "img1.jpg").touch()
    
    empty_person_dir = tmp_path / "person_b_empty"
    empty_person_dir.mkdir()
    
    summary = data_validate.check_person_folders(tmp_path)
    
    assert "Warning: Person person_b_empty has no images" in summary

@patch('mlops_xoxo.data_validate.cv2.imread')
def test_validate_images_finds_corrupted_image(mock_imread, tmp_path):
    """Unit test to ensure corrupted images (where imread returns None) are caught"""
    # Setup a fake directory with two files
    person_a_dir = tmp_path / "person_a"
    person_a_dir.mkdir()
    valid_img_path = person_a_dir / "valid.jpg"
    corrupt_img_path = person_a_dir / "corrupt.jpg"
    valid_img_path.touch()
    corrupt_img_path.touch()
    
    # Mock cv2.imread to return None for the "corrupt" file
    mock_imread.side_effect = lambda path: None if "corrupt" in str(path) else MagicMock()
    
    corrupted_files = data_validate.validate_images(tmp_path)
    
    assert len(corrupted_files) == 1
    assert str(corrupt_img_path) in corrupted_files

def test_check_manifest_handles_missing_file():
    """Unit test to ensure a missing manifest is handled well"""
    missing_path = "/nonexistent/path/manifest.json"
    lines = data_validate.check_manifest(missing_path)
    assert lines == ["Manifest not found"]

def test_main_correctly_reports_corrupted_file(mocker, tmp_path):
    """
    A self-contained test to ensure the data_validate_main() function correctly identifies
    and reports a corrupted file.
    """
    # Create a complete fake environment temporarily to store
    # the corrupted file
    raw_dir = tmp_path / "raw"
    person_a_dir = raw_dir / "person_a"
    person_a_dir.mkdir(parents=True)
    manifest_dir = tmp_path / "processed" / "splits"
    manifest_dir.mkdir(parents=True)
    
    # Create a corrupted (empty) file and get its path
    corrupt_path = person_a_dir / "corrupted.jpg"
    corrupt_path.touch()
    
    # Create a dummy manifest file for the script to find
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"train": [], "val": [], "test": []}, f)
        
    # Define the path for the output report
    report_path = tmp_path / "report.txt"

    mocker.patch("mlops_xoxo.data_validate.RAW_DIR", str(raw_dir))
    mocker.patch("mlops_xoxo.data_validate.MANIFEST_PATH", str(manifest_path))
    mocker.patch("mlops_xoxo.data_validate.report_path", str(report_path))
    
    # Import the script now that the paths are mocked
    from mlops_xoxo import data_validate

    # Run the data_validate_main function
    data_validate.data_validate_main()

    # Check the content of the generated report
    assert report_path.exists()
    report_content = report_path.read_text()

    assert "Corrupted images: 1" in report_content
    assert str(corrupt_path) in report_content