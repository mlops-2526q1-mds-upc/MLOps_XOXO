
from unittest.mock import patch

DATA_INGEST_SCRIPT_PATH = "mlops_xoxo.face_embedding.data_ingest"

def test_main_calls_rec_to_images_correctly():
    """
    Tests that the main() function in data_ingest.py calls the
    rec_to_images utility with the  parameters.
    """
    # We mock the utility function that the script calls
    # This isolates our test to only the logic within data_ingest.py
    mock_target = 'mlops_xoxo.face_embedding.utils.data_utils.rec_to_images'
    
    with patch(mock_target) as mock_rec_to_images:
        
        from mlops_xoxo.face_embedding import data_ingest
        
        data_ingest.main()
        
        # Check that the mocked utility was called correctly 
        mock_rec_to_images.assert_called_once()
        
        # Check that it was called with the exact arguments defined in the script
        mock_rec_to_images.assert_called_once_with(
            "data/external/casioface/train.rec",
            "data/external/casioface/train.idx",
            "data/external/casioface/train.lst",
            "data/raw/face_embedding",
            limit=10000,
            show_progress=True
        )