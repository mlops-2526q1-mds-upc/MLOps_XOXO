import pytest
from unittest.mock import patch, MagicMock
import sys
DATA_INGEST_SCRIPT_PATH = 'mlops_xoxo.data_ingest'

# The main function in data_ingest.py is called implicitly upon execution due 
# to the standard Python __name__ == "__main__" block calling main().
# Therefore, importing the module executes the script's logic for the test.

def test_main_calls_rec_to_images_with_correct_params():
    """
    Tests that the data_ingest script's main function correctly calls
    rec_to_images with all the defined constants (paths and limit).
    """
    
    # We mock the function from the UTILS module that data_ingest imports
    # The full path is mlops_xoxo.utils.data_utils.rec_to_images
    mock_target = 'mlops_xoxo.utils.data_utils.rec_to_images'
    
    with patch(mock_target) as mock_rec_to_images:
        
        # We must delete the module from sys.modules to force a fresh import
        # and execute the main function logic when we import it.
        if DATA_INGEST_SCRIPT_PATH in sys.modules:
            del sys.modules[DATA_INGEST_SCRIPT_PATH]
            
        # Execute the script's entry point
        import mlops_xoxo.data_ingest 
        
        # Define the expected call arguments from the data_ingest.py file
        # REC_PATH, IDX_PATH, LST_PATH, OUTPUT_DIR are constants in the file.
        expected_rec_path = "data/external/casioface/train.rec"
        expected_idx_path = "data/external/casioface/train.idx"
        expected_lst_path = "data/external/casioface/train.lst"
        expected_output_dir = "data/raw"
        expected_limit = 10000 

        # Assert the function was called exactly once
        mock_rec_to_images.assert_called_once()
        
        # Assert the function was called with the correct keyword arguments
        mock_rec_to_images.assert_called_with(
            expected_rec_path,
            expected_idx_path,
            expected_lst_path,
            expected_output_dir,
            limit=expected_limit,
            show_progress=True
        )

