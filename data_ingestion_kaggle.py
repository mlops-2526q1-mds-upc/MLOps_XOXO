import os
import shutil
import time
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# The temporary directory to download and extract the full dataset
temp_dir = "temp_dataset"

# New directory for processed images dataset
dataset_dir = "images_dataset"

# Percentage of images to sample
sample_pct = 0.30

# Mapillary Vistas v2.0 dataset splits
dataset_splits = ["training", "validation"]

# ----------------------------------------------------
# Image Dataset API Download and Cleanup
# ----------------------------------------------------

    
def download_dataset():
    """
    Automates the download of the Mapillary Vistas dataset from Kaggle
    and displays a progress bar.
    """
    print("Downloading dataset...")
    
    dataset_slug = 'kaggleprollc/mapillary-vistas-image-data-collection'
    download_path = './temp_dataset'
    
    # Ensure the download path is clean before starting
    if os.path.exists(download_path):
        shutil.rmtree(download_path)
    os.makedirs(download_path)

    api = KaggleApi()
    api.authenticate()

    # Define the path for the downloaded zip file
    zip_filename = 'mapillary-vistas-image-data-collection.zip'
    zip_filepath = os.path.join(download_path, zip_filename)

    estimated_total_size = 32.26 * 1024 * 1024 * 1024  # 32.26 GB in bytes

    try:
        # Start the download and get the response object
        response = api.dataset_download_files(dataset_slug, path=download_path, quiet=True)
        
        # Monitor the download progress manually
        with tqdm(total=estimated_total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            last_size = 0
            while not os.path.exists(zip_filepath) or os.path.getsize(zip_filepath) < estimated_total_size:
                if os.path.exists(zip_filepath):
                    current_size = os.path.getsize(zip_filepath)
                    pbar.update(current_size - last_size)
                    last_size = current_size
                    
                    if current_size >= estimated_total_size:
                        break
                time.sleep(5) # Check progress every 5 seconds
        
        print("Download complete.")
        
        # After download, unzip the files
        print("Unzipping files...")
        shutil.unpack_archive(zip_filepath, download_path)
        os.remove(zip_filepath)  # Clean up the zip file
        print("Unzipping complete.")
        
        # Return the path to the extracted dataset
        extracted_path = os.path.join(download_path, 'mapillary-vistas-image-data-collection')
        return extracted_path

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure the Kaggle CLI is installed and configured correctly with your API key.")
        return None

def cleanup_files(dataset_splits, base_path):
    """
    Performs file deletions on the downloaded dataset to remove unnecessary files.
    """
    print(f"Cleaning up {split_name} data...")

    # Delete the v1.2 folder
    v1_2_dir = os.path.join(base_path, split_name, "v1.2")
    if os.path.exists(v1_2_dir):
        shutil.rmtree(v1_2_dir)
        print(f"Deleted: {v1_2_dir}")

    # Delete all files in v2.0/panoptic except panoptic_2020.json
    panoptic_dir = os.path.join(base_path, split_name, "v2.0", "panoptic")
    if os.path.exists(panoptic_dir):
        files_to_delete = [f for f in os.listdir(panoptic_dir) if f != "panoptic_2020.json"]
        for filename in files_to_delete:
            os.remove(os.path.join(panoptic_dir, filename))
        print(f"Cleaned out all but 'panoptic_2020.json' from {panoptic_dir}")
    else:
        print(f"Warning: 'panoptic' folder not found in '{split_name}'.")

# ----------------------------------------------------
# Creating smaller dataset
# ----------------------------------------------------

def create_dataset_subset(dataset_path, dataset_dir, sample_pct, dataset_splits):
    """
    Creates a smaller dataset and copies files to the new directory structure.
    """
    print("\nStarting the data ingestion and sampling process.")

    # Define the source and destination paths based on the new structure
    for split_name in dataset_splits:
        print(f"Processing the '{split_name}' split...")

        # Define the source paths within the temporary directory
        source_images_dir = os.path.join(dataset_path, split_name, "images")
        source_labels_dir = os.path.join(dataset_path, split_name, "v2.0", "labels")
        source_instances_dir = os.path.join(dataset_path, split_name, "v2.0", "instances")
        source_polygons_dir = os.path.join(dataset_path, split_name, "v2.0", "polygons")

        # Define the destination paths of the new directory structure
        dest_split_dir = os.path.join(dataset_dir, split_name)
        dest_images_dir = os.path.join(dest_split_dir, "images")
        dest_labels_dir = os.path.join(dest_split_dir, "labels")
        dest_instances_dir = os.path.join(dest_split_dir, "instances")
        dest_polygons_dir = os.path.join(dest_split_dir, "polygons")

        # Create the destination folders
        os.makedirs(dest_images_dir, exist_ok=True)
        os.makedirs(dest_labels_dir, exist_ok=True)
        os.makedirs(dest_instances_dir, exist_ok=True)
        os.makedirs(dest_polygons_dir, exist_ok=True)

        # Get a list of all image filenames for this split
        try:
            image_filenames = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg')]
        except FileNotFoundError:
            print(f"Warning: The images folder for '{split_name}' was not found. Skipping.")
            continue

        # Get the number of images for the smaller dataset
        total_images = len(image_filenames)
        num_to_sample = int(total_images * sample_pct)

        print(f"Total images found: {total_images}")
        print(f"Sampling {num_to_sample} images...")

        if num_to_sample == 0 and total_images > 0:
            num_to_sample = 1
            print("Note: Sample size is less than 1. Sampling 1 image.")

        if total_images == 0:
            print("No images to sample. Skipping.")
            continue

        # Get the new images from the original dataset
        sampled_images = random.sample(image_filenames, num_to_sample)
        copied_count = 0

        # Copy the panoptic and config files to the destination
        source_panoptic_path = os.path.join(dataset_path, split_name, "v2.0", "panoptic", "panoptic_2020.json")
        if os.path.exists(source_panoptic_path):
            shutil.copy(source_panoptic_path, dest_split_dir)
            print("Copied panoptic_2020.json")

        source_config_path = os.path.join(dataset_path, split_name, "v2.0", "config_v2.0.json")
        if os.path.exists(source_config_path):
            shutil.copy(source_config_path, dest_split_dir)
            print("Copied config_v2.0.json")

        # Keep the sampled dataset files
        for image_filename in sampled_images:
            base_name = os.path.splitext(image_filename)[0]

            source_paths = {
                "image": os.path.join(source_images_dir, image_filename),
                "label": os.path.join(source_labels_dir, f"{base_name}.png"),
                "instance": os.path.join(source_instances_dir, f"{base_name}.png"),
                "polygon": os.path.join(source_polygons_dir, f"{base_name}.json")
            }

            dest_paths = {
                "image": os.path.join(dest_images_dir, image_filename),
                "label": os.path.join(dest_labels_dir, f"{base_name}.png"),
                "instance": os.path.join(dest_instances_dir, f"{base_name}.png"),
                "polygon": os.path.join(dest_polygons_dir, f"{base_name}.json")
            }

            # Check if all source files exist before copying
            if all(os.path.exists(p) for p in source_paths.values()):
                for file_type, source_path in source_paths.items():
                    shutil.copy(source_path, dest_paths[file_type])
                copied_count += 1
            else:
                print(f"Warning: Missing a file for '{image_filename}'. Skipping.")

        print(f"Completed '{split_name}' split. Copied {copied_count} images.")

# ----------------------------------------------------
# Final cleanup
# ----------------------------------------------------

def final_cleanup(temp_dir):
    """ Removes the temporary dataset directory. """
    print("\nFinal cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Successfully removed temporary directory: {temp_dir}")


# MAIN
if __name__ == "__main__":
    print("Starting the data ingestion and sampling process.")

    # Download the full dataset via Kaggle API
    dataset_path = download_dataset()

    # Perform initial cleanup of the downloaded data
    for split_name in dataset_splits:
        cleanup_files(split_name, dataset_path)

    # Create a smaller, processed dataset
    create_dataset_subset(dataset_path, dataset_dir, sample_pct, dataset_splits)

    # Remove the temporary full dataset
    final_cleanup(temp_dir)

    print(f"\nAll steps completed. The processed dataset is located in the '{dataset_dir}' folder.")