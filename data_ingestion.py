import mapillary.interface as mly
import requests
import shutil
import os
import random
import tarfile
from dotenv import load_dotenv
import random_land_points as rlp

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# The temporary directory to download and extract the full dataset
dataset_dir = "images_dataset"

# This folder will contain the processed dataset
subset_dir = "output"

# Percentage of images to sample
sample_pct = 0.30 

# A list of the sub-folders to process
dataset_urls = {
    "training": "https://www.mapillary.com/dataset/vistas/v2.0/training.tar.gz",
    "validation": "https://www.mapillary.com/dataset/vistas/v2.0/validation.tar.gz"
}

# The file types to look for in each folder
file_types = {
    "images": ".jpg",
    "labels": ".png",
    "instances": ".png",
    "polygons": ".json"
}

# ----------------------------------------------------
# Image Dataset API Download
# ----------------------------------------------------

# Get the access token from the .env file
load_dotenv() 
ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")

# Set the access token for the Mapillary API
mly.set_access_token(ACCESS_TOKEN)

def download_and_extract(url, path):
    """Downloads a .tar.gz file from a URL and extracts it."""
    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download successful.")
        
        print(f"Extracting to {dataset_dir}...")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=dataset_dir)
        print("Extraction successful.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        print("Please check your internet connection or the URL.")
        exit()
    except tarfile.TarError as e:
        print(f"Error extracting the dataset: {e}")
        exit

# ----------------------------------------------------
# Creating smaller dataset
# ----------------------------------------------------

print(f"Starting the data sampling process...")

for split_name in splits:
    print(f"\nProcessing the '{split_name}' split...")

    # Define the source paths for the current split
    source_dirs = {
        "images": os.path.join(dataset_dir, split_name, "images"),
        "labels": os.path.join(dataset_dir, split_name, "labels"),
        "instances": os.path.join(dataset_dir, split_name, "instances"),
        "polygons": os.path.join(dataset_dir, split_name, "polygons")
    }

    # Define the destination paths for the subset
    dest_dirs = {
        "images": os.path.join(subset_dir, split_name, "images"),
        "labels": os.path.join(subset_dir, split_name, "labels"),
        "instances": os.path.join(subset_dir, split_name, "instances"),
        "polygons": os.path.join(subset_dir, split_name, "polygons")
    }

    # Create the destination folders
    for dir_name in dest_dirs.values():
        os.makedirs(dir_name, exist_ok=True)

    # Get a list of all image filenames for this split
    try:
        image_filenames = [f for f in os.listdir(source_dirs["images"]) if f.endswith(file_types["images"])]
    except FileNotFoundError:
        print(f"Warning: The images folder for '{split_name}' was not found. Skipping.")
        continue

    total_images = len(image_filenames)

    # Calculate the number of images to sample
    num_to_sample = int(total_images * sample_pct)

    print(f"Total images found: {total_images}")
    print(f"Sampling {num_to_sample} images (30%)...")

    # Randomly select the image filenames
    if num_to_sample == 0 and total_images > 0:
        # Handle the case where the sample size is less than 1
        num_to_sample = 1
        print("Note: Sample size is less than 1. Sampling 1 image.")
        
    if total_images == 0:
        print("No images to sample. Skipping.")
        continue
        
    sampled_images = random.sample(image_filenames, num_to_sample)
    copied_count = 0

    for image_filename in sampled_images:
        # Get the base filename without the extension
        base_name = os.path.splitext(image_filename)[0]

        # Define the full source and destination paths for all files
        source_paths = {
            "image": os.path.join(source_dirs["images"], image_filename),
            "label": os.path.join(source_dirs["labels"], f"{base_name}.png"),
            "instance": os.path.join(source_dirs["instances"], f"{base_name}.png"),
            "polygon": os.path.join(source_dirs["polygons"], f"{base_name}.json")
        }
        
        dest_paths = {
            "image": os.path.join(dest_dirs["images"], image_filename),
            "label": os.path.join(dest_dirs["labels"], f"{base_name}.png"),
            "instance": os.path.join(dest_dirs["instances"], f"{base_name}.png"),
            "polygon": os.path.join(dest_dirs["polygons"], f"{base_name}.json")
        }
        
        # Check if all files exist and then copy them
        if all(os.path.exists(p) for p in source_paths.values()):
            for file_type, source_path in source_paths.items():
                shutil.copy(source_path, dest_paths[file_type])
            copied_count += 1
            
        else:
            print(f"Warning: Missing a file for '{image_filename}'. Skipping.")

    print(f"Completed '{split_name}' split. Copied {copied_count} images.")

print(f"\nAll splits processed. Subset created in the '{subset_dir}' folder.")