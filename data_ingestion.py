import os
import shutil
import random
import gdown
import json

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# Google Drive folder ID
folder_id = '13rTd4Do3jacWSFW6m8y7LpirNxYli-YW'

# Downloaded dataset
final_dataset_dir = "final_dataset"

# ----------------------------------------------------
# Download the dataset from Google Drive
# ----------------------------------------------------

def download_dataset(folder_id, download_dir):
    """
    Downloads a folder from Google Drive into the specified directory.
    
    Args:
        folder_id (str): The Google Drive folder ID.
        download_dir (str): The local path where the folder will be downloaded.
    
    Returns:
        str: The path to the downloaded content, or None if the download fails.
    """
    print("Starting the dataset download...")
    
    # Ensure the output directory is clean
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    os.makedirs(download_dir)
    
    try:
        # Downloads the entire folder to the temporary directory
        gdown.download_folder(id=folder_id, output=download_dir, quiet=False)
        print("Download complete.")
        
        # gdown downloads the folder's contents, and we need to find the correct path.
        # We assume the content is in a subfolder named like the folder ID.
        downloaded_content_path = os.path.join(download_dir, folder_id)

        if not os.path.exists(downloaded_content_path):
            # Fallback in case gdown names the folder differently
            subdirs = [d for d in os.listdir(download_dir) if os.path.isdir(os.path.join(download_dir, d))]
            if subdirs:
                downloaded_content_path = os.path.join(download_dir, subdirs[0])
            else:
                print("Error: Downloaded content not found.")
                return None

        return downloaded_content_path
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None
    
# ----------------------------------------------------
# Clean the dataset
# ----------------------------------------------------

def cleanup_dataset(base_dir):
    """
    Restructures the dataset by moving files from 'v2.0' subfolders up one level
    and removing unnecessary files.
    
    Args:
        base_dir (str): The path to the root dataset directory to clean.
    """
    print("\nStarting dataset cleanup and restructuring...")
    
    splits = ["training", "validation"]

    for split in splits:
        print(f"Processing '{split}' data...")
        split_path = os.path.join(base_dir, split)
        v2_0_path = os.path.join(split_path, "v2.0")

        if not os.path.exists(v2_0_path):
            print(f"Warning: 'v2.0' folder not found for '{split}'. Skipping.")
            continue

        subfolders = ["instances", "labels", "panoptic", "polygons"]
        for subfolder in subfolders:
            source = os.path.join(v2_0_path, subfolder)
            destination = os.path.join(split_path, subfolder)

            if os.path.exists(source):
                os.makedirs(destination, exist_ok=True)
                for item in os.listdir(source):
                    shutil.move(os.path.join(source, item), destination)

        panoptic_path = os.path.join(split_path, "panoptic")
        if os.path.exists(panoptic_path):
            files_to_delete = [f for f in os.listdir(panoptic_path) if f != "panoptic_2020.json"]
            for filename in files_to_delete:
                os.remove(os.path.join(panoptic_path, filename))
            print(f"Cleaned out all but 'panoptic_2020.json' from {panoptic_path}")
        
        shutil.rmtree(v2_0_path)
        print(f"Deleted: {v2_0_path}")

    print("\nDataset structure cleanup complete.")


if __name__ == "__main__":
    # Download the raw dataset ---
    downloaded_path = download_dataset(folder_id, final_dataset_dir)

    if downloaded_path:
        # Clean and restructure the raw dataset from unnecessary files
        cleanup_dataset(downloaded_path)
        
    print("\nProcessing finished. Your cleaned dataset is in the 'final_dataset' folder.")