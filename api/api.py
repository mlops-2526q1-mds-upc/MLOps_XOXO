import io
import sys
from pathlib import Path
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import uvicorn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mlops_xoxo.train import MobileFace

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

# Load the trained model state
MODEL_PATH = project_root / "models/face_embedding/mobilenetv2_arcface_epoch_last.pt" 
model = MobileFace(emb_size=512).to(DEVICE) 

if not MODEL_PATH.exists():
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model state_dict: {e}")
    sys.exit(1)

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- FastAPI Application ---

# Main application instance that handles web requests
app = FastAPI(title="Face Embedding API", description="API to generate face embeddings using MobileNetV2+ArcFace.")

@app.post("/predict", summary="Generate face embedding")
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
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image transformation: {e}")

    # Perform inference
    with torch.no_grad():
        try:
            embedding = model(img_tensor)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Convert embedding tensor to a list for JSON response
    embedding_list = embedding.cpu().numpy().flatten().tolist()

    return {"embedding": embedding_list}

@app.get("/info", summary="Get model information")
def get_model_info():
    """
    Returns metadata about the currently loaded face embedding model.
    """
    model_info = {
        "model_name": "MobileNetV2 + ArcFace", 
        "model_file_path": str(MODEL_PATH),
        "embedding_size": 512, 
        "device": str(DEVICE),
        "input_normalization": { 
            "type": "ToTensor & Normalize",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        }
    }
    return model_info

@app.get("/", summary="Root endpoint")
def read_root():
    """Provides basic information about the API."""
    return {"message": "Welcome to the Face Embedding API. Use the /predict endpoint to generate embeddings, /info to get"
    "model metadata."}

# Running the app
if __name__ == "__main__":
    # Access at http://127.0.0.1:8000 or http://127.0.0.1:8000/docs
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True, app_dir="api")