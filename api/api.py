import io
import sys
from pathlib import Path
import torch
import yaml
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import uvicorn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mlops_xoxo.face_embedding.train import MobileFace
from mlops_xoxo.age_gender.model import GenderAgeModel

# --- Configuration & Model Loading ---

# Load parameters
try:
    with open("../params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    print("Warning: params.yaml not found. Using default settings.")
    params = {'training': {'device': 'cpu'}}

# Device setup 
device_param = params['training'].get('device', 'cpu').lower()
if device_param == 'cuda' and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif device_param == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"API using device: {DEVICE}")

# Load face embedding model
EMBEDDING_MODEL_PATH = project_root / "models/face_embedding/mobilenetv2_arcface_epoch_last.pt" 
embedding_model = MobileFace(emb_size=512).to(DEVICE) 

if not EMBEDDING_MODEL_PATH.exists():
    print(f"ERROR: Model file not found at {EMBEDDING_MODEL_PATH}")
    sys.exit(1)

try:
    state_dict = torch.load(EMBEDDING_MODEL_PATH, map_location=DEVICE)
    embedding_model.load_state_dict(state_dict, strict=False)
    embedding_model.eval()
    print(f"Model loaded successfully from {EMBEDDING_MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model state_dict: {e}")
    sys.exit(1)

# Image transformations
embedding_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load age/gender model

age_gender_model = None  
age_gender_transform = None

try:
    AGE_GENDER_MODEL_PATH = project_root / "models/age_gender_model.pth"
    if not AGE_GENDER_MODEL_PATH.exists():
        print(f"WARNING: Age/Gender model file not found at {AGE_GENDER_MODEL_PATH}. This endpoint will be disabled.")
    else:
        # If file exists, load the model
        age_gender_model = GenderAgeModel().to(DEVICE)
        state_dict = torch.load(AGE_GENDER_MODEL_PATH, map_location=DEVICE)
        age_gender_model.load_state_dict(state_dict, strict=False)
        age_gender_model.eval()
        
        # Image transformations
        age_gender_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"Age/Gender model loaded successfully from {AGE_GENDER_MODEL_PATH}")

except ImportError:
    print("INFO: Age/Gender model code (mlops_xoxo/age_gender/model.py) not found. This endpoint will be disabled.")
except Exception as e:
    print(f"WARNING: Failed to load Age/Gender model. Endpoint will be disabled. Error: {e}")

# --- FastAPI Application ---

# Main application instance that handles web requests
app = FastAPI(title="XOFace Multi-Model API", description="API to generate face embeddings and age/gender estimation.")

@app.post("/predict_embedding", summary="Generate face embedding")
async def predict_embedding(file: UploadFile = File(..., description="Image file of a face")):
    """
    Accepts an image file, processes it, and returns the 512-dimensional face embedding.
    """
    # Read image file
    contents = await file.read()
    
    try:
        # Open image using PIL
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Preprocess the image
    try:
        img_tensor = embedding_transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image transformation: {e}")

    # Perform inference
    with torch.no_grad():
        try:
            embedding = embedding_model(img_tensor)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Convert embedding tensor to a list for JSON response
    embedding_list = embedding.cpu().numpy().flatten().tolist()

    return {"embedding": embedding_list}

@app.post("/predict_age_gender", summary="Predict Age and Gender")
async def predict_age_gender(file: UploadFile = File(..., description="Image file of a face")):
    """
    Accepts an image and returns predicted age and gender indices.
    """
    if age_gender_model is None:
        raise HTTPException(
            status_code=503,
            detail="Age/Gender model is not loaded. Please check server configuration."
        )

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    img_tensor = age_gender_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        age_logits, gender_logits = age_gender_model(img_tensor)

    age_pred_index = torch.argmax(age_logits, dim=1).item()
    gender_pred_index = torch.argmax(gender_logits, dim=1).item()

    return {
        "age_index": age_pred_index,
        "gender_index": gender_pred_index
    }

@app.get("/info", summary="Get model information")
def get_model_info():
    """Returns metadata about the currently loaded models."""
    info = {
        "embedding_model": {
            "model_name": "MobileNetV2 + ArcFace",
            "model_file_path": str(EMBEDDING_MODEL_PATH),
            "device": str(DEVICE),
            "status": "Loaded"
        },
        "age_gender_model": {
            "model_name": "EfficientNet-B0",
            "model_file_path": str(AGE_GENDER_MODEL_PATH),
            "device": str(DEVICE),
            "status": "Loaded" if age_gender_model is not None else "Not available"
        }
    }
    return info

@app.get("/", summary="Root endpoint")
def read_root():
    """Provides basic information about the API."""
    return {"message": "Welcome to the XOFace Multi-Model API. Use the /predict endpoint to generate embeddings, /info to get"
    "model metadata."}

# Running the app
if __name__ == "__main__":
    # Access at http://127.0.0.1:8000 or http://127.0.0.1:8000/docs
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True, app_dir="api")