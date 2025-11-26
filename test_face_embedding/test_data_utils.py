import pytest
import base64
from pathlib import Path
from unittest.mock import patch

from mlops_xoxo.face_embedding.utils.data_utils import rec_to_images

REAL_REC_PATH = Path("data/external/casioface/train.rec")
REAL_IDX_PATH = Path("data/external/casioface/train.idx")
REAL_LST_PATH = Path("data/external/casioface/train.lst")

# Mark the test as 'slow' and skip it if the data is not present
@pytest.mark.slow
@pytest.mark.skipif(not REAL_REC_PATH.exists(), reason="Real data not found. Run 'dvc pull' first.")
def test_rec_to_images_with_real_data(tmp_path):
    """
    This is an integration test that runs rec_to_images on a small
    subset of the real data to ensure it doesn't crash.
    """
    output_dir = tmp_path / "real_data_output"
    
    # Run the function on the real data, but with a small limit
    # to keep the test from taking too long.
    rec_to_images(
        rec_path=REAL_REC_PATH,
        idx_path=REAL_IDX_PATH,
        lst_path=REAL_LST_PATH,
        output_dir=output_dir,
        limit=20,  # Process only the first 20 records
        show_progress=False
    )
    
    # Check for general outcomes rather than exact files, as the
    # real data might change over time.
    assert output_dir.exists()
    
    # Check that at least one person's directory was created
    person_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    assert len(person_dirs) > 0, "No person directories were created."
    
    # Check that at least one image was extracted
    image_files = list(output_dir.glob("**/*.jpg"))
    assert len(image_files) > 0, "No JPG files were extracted."
    assert len(image_files) <= 20, "More images were extracted than the limit."

def test_rec_to_images_handles_no_jpeg_header(tmp_path):
    """
    Tests that the function skips records with no JPEG header.
    """
    rec_path = tmp_path / "test.rec"
    idx_path = tmp_path / "test.idx"
    lst_path = tmp_path / "test.lst"
    output_dir = tmp_path / "output"
    
    # Create a .rec file with just junk data
    with open(rec_path, 'wb') as f:
        f.write(b"this is not a jpeg")
        
    with open(idx_path, 'w') as f:
        f.write("101 0\n")
        
    with open(lst_path, 'w') as f:
        f.write("0\t101\tdata/raw/person_d/img_01.jpg\n")
        
    rec_to_images(rec_path, idx_path, lst_path, output_dir, show_progress=False)
    
    # Function should run without crashing and produce nothing
    assert not list(output_dir.glob("**/*.jpg"))

@pytest.fixture
def fake_rec_files(tmp_path):
    """
    Creates a set of fake, but valid, .rec, .idx, and .lst files for testing.
    """
    # A tiny, valid 1x1 pixel black JPEG, encoded in base64
    tiny_jpeg_b64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAABwn/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCUQoKKAAA="
    jpeg_bytes = base64.b64decode(tiny_jpeg_b64)
    
    rec_path = tmp_path / "test.rec"
    idx_path = tmp_path / "test.idx"
    lst_path = tmp_path / "test.lst"
    
    with open(rec_path, 'wb') as f:
        f.write(b"JUNK_DATA_PREFIX")
        f.write(jpeg_bytes)
        
    with open(idx_path, 'w') as f:
        f.write("101 0\n")
        
    with open(lst_path, 'w') as f:
        f.write("0\t101\tdata/raw/person_c/img_01.jpg\n")
        
    return rec_path, idx_path, lst_path

@patch('mlops_xoxo.face_embedding.utils.data_utils.cv2.imdecode', return_value=None)
def test_rec_to_images_handles_corrupt_jpeg(mock_imdecode, fake_rec_files, tmp_path):
    """
    Tests that the function correctly skips records where cv2 cannot decode the image
    """
    rec_path, idx_path, lst_path = fake_rec_files
    output_dir = tmp_path / "output"
    
    rec_to_images(rec_path, idx_path, lst_path, output_dir, show_progress=False)
    
    # The function should call imdecode but still produce nothing
    mock_imdecode.assert_called()
    assert not list(output_dir.glob("**/*.jpg"))