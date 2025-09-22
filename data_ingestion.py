import requests
import shutil
import os
import random
import tarfile
import time

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# The temporary directory to download and extract the full dataset
temp_dir = "temp_dataset"

# New directory for processed images dataset
dataset_dir = "images_dataset"

# Percentage of images to sample
sample_pct = 0.30 

# A list of the sub-folders to process
dataset_urls = {
    "training": "https://www.mapillary.com/dataset/vistas/v2.0/training.tar.gz",
    "validation": "https://www.mapillary.com/dataset/vistas/v2.0/validation.tar.gz"
}

# ----------------------------------------------------
# Image Dataset API Download and Removal of unnecessary files
# ----------------------------------------------------

def download_and_extract(temp_dir, dataset_urls):
    """
    Downloads specified datasets.
    """
    print("\nStarting data download and extraction...")
    os.makedirs(temp_dir, exist_ok=True)
    
    for split_name, url in dataset_urls.items():
        archive_path = os.path.join(temp_dir, f"{split_name}.tar.gz")
        
        print(f"Downloading {split_name} data from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download successful.")
            
            print(f"Extracting to {temp_dir}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)
            print("Extraction successful.")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the dataset: {e}")
            print("Please check your internet connection or the URL.")
            exit()
        except tarfile.TarError as e:
            print(f"Error extracting the dataset: {e}")
            exit()
        
        # Clean up unnecessary files immediately after extraction
        cleanup_files(split_name, temp_dir)
        time.sleep(1)

def cleanup_files(split_name, temp_dir):
    """
    Performs file deletions - deleting unnecessary duplicate v1.2 version files
    and panoptic .png files
    """
    print(f"Cleaning up {split_name} data...")

    # Delete the entire v1.2 folder
    v1_2_dir = os.path.join(temp_dir, split_name, "v1.2")
    if os.path.exists(v1_2_dir):
        shutil.rmtree(v1_2_dir)
        print(f"Deleted: {v1_2_dir}")

    # Delete all files in v2.0/panoptic except panoptic_2020.json
    panoptic_dir = os.path.join(temp_dir, split_name, "v2.0", "panoptic")
    if os.path.exists(panoptic_dir):
        files_to_delete = [f for f in os.listdir(panoptic_dir) if f != "panoptic_2020.json"]
        for filename in files_to_delete:
            os.remove(os.path.join(panoptic_dir, filename))
        print(f"Cleaned out all but 'panoptic_2020.json' from {panoptic_dir}")
    else:
        print(f"Warning: 'panoptic' folder not found in '{split_name}'.")

    print(f"Cleaning up '{split_name}' temporary data...")

# ----------------------------------------------------
# Creating smaller dataset
# ----------------------------------------------------

def create_dataset_subset():
    """
    Samples a smaller dataset and copies files to the new directory structure.
    """
    print("\nStarting the data ingestion and sampling process.")

    # Download and extract the full dataset 
    print("\nDownloading and extracting the full dataset")
    os.makedirs(temp_dir, exist_ok=True)
    for split_name, url in dataset_urls.items():
        archive_path = os.path.join(temp_dir, f"{split_name}.tar.gz")
        download_and_extract(url, archive_path, temp_dir)
        time.sleep(1)
        cleanup_files(split_name)

    # Define the source and destination paths based on the new structure (for training and validation)
    for split_name in dataset_urls.keys():
        print(f"Processing the '{split_name}' split...")

        # Define the source paths within the temporary directory
        source_images_dir = os.path.join(temp_dir, split_name, "images")
        source_labels_dir = os.path.join(temp_dir, split_name, "v2.0", "labels")
        source_instances_dir = os.path.join(temp_dir, split_name, "v2.0", "instances")
        source_polygons_dir = os.path.join(temp_dir, split_name, "v2.0", "polygons")
        
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

        # Get the number of image for smaller dataset
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
        
        # Get the new images from original dataset
        sampled_images = random.sample(image_filenames, num_to_sample)
        copied_count = 0

        # Copy the panoptic and config files to the destination
        source_panoptic_path = os.path.join(temp_dir, split_name, "v2.0", "panoptic", "panoptic_2020.json")
        if os.path.exists(source_panoptic_path):
            shutil.copy(source_panoptic_path, dest_split_dir)
            print("Copied panoptic_2020.json")

        source_config_path = os.path.join(temp_dir, split_name, "v2.0", "config_v2.0.json")
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
    """ Main function to orchestrate the data ingestion pipeline. """
    print("Starting the data ingestion and sampling process.")
    
    download_and_extract(temp_dir, dataset_urls)
    create_dataset_subset(temp_dir, dataset_dir, sample_pct, dataset_urls)
    final_cleanup(temp_dir)
    
    print(f"\nAll steps completed. The processed dataset is located in the '{dataset_dir}' folder.")