# Run 'python api/api.py'
# Access the API UI (Swagger UI) at http://127.0.0.1:8000 or http://127.0.0.1:8000/docs

import io, os
import sys
from pathlib import Path
import torch
import yaml
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
from prometheus_fastapi_instrumentator import Instrumentator
import torch.nn.functional as F
import uvicorn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
# repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# correct directory name (underscore)
FACE_EMB = os.path.join(ROOT, "mlops_xoxo", "face_embedding")

sys.path.append(ROOT)
sys.path.append(FACE_EMB)
from mlops_xoxo.utils.models import MobileFace

EMBEDDING_MODEL_PATH = project_root / "models/face_embedding/mobilenetv2_arcface_model.pth" 
embedding_model = MobileFace(emb_size=512).to(DEVICE) 

if not EMBEDDING_MODEL_PATH.exists():
    print(f"ERROR: Model file not found at {EMBEDDING_MODEL_PATH}")
    sys.exit(1)

try:
    state_dict = torch.load(EMBEDDING_MODEL_PATH, map_location=DEVICE)
    embedding_model.load_state_dict(state_dict, strict=False)
    embedding_model.eval()
    print(f"Embedding face model loaded.")
except Exception as e:
    print(f"ERROR loading model state_dict: {e}")
    sys.exit(1)

# Image transformations
embedding_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load age and gender estimation models

AGE_MODEL_PATH =  project_root / "models/age_gender/age_model.pt"
GENDER_MODEL_PATH =  project_root / "models/age_gender/gender_model.pt"
age_model = None
gender_model = None

age_gender_transform = transforms.Compose([
    transforms.Resize((200, 200)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    from mlops_xoxo.utils.models import SimpleModel

    if AGE_MODEL_PATH.exists():
        age_model = SimpleModel(num_outputs=1).to(DEVICE) 
        age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
        age_model.eval()
        print("Age model loaded.")
    else: 
        print(f"Age model file not found at {AGE_MODEL_PATH}")

    if GENDER_MODEL_PATH.exists():
        gender_model = SimpleModel(num_outputs=2).to(DEVICE)
        gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
        gender_model.eval()
        print("Gender model loaded.")
    else:
        print(f"Gender model file not found at {GENDER_MODEL_PATH}")

except ImportError:
    print("Could not import SimpleModel class from mlops_xoxo.age_gender.train")
except Exception as e:
    print(f"Error loading Age/Gender models: {e}")

# Load emotion recognition model

emotion_model = None
emotion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((48, 48)),                 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

try:
    from mlops_xoxo.utils.models import build_resnet18
    
    EMOTION_PATH = project_root / "models/emotion_classification/best_model.pth"
    
    if EMOTION_PATH.exists():
        # Rebuild the exact model architecture
        emotion_model = build_resnet18(num_classes=len(EMOTION_CLASSES), pretrained=False, in_channels=1)
        
        # Load weights
        checkpoint = torch.load(EMOTION_PATH, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            emotion_model.load_state_dict(checkpoint['state_dict'])
        else:
            emotion_model.load_state_dict(checkpoint)
            
        emotion_model.to(DEVICE)
        emotion_model.eval()
        print("Emotion model loaded.")
    else:
        print(f"⚠️ Emotion model file not found at {EMOTION_PATH}")

except ImportError:
    print("Could not import build_resnet18 from mlops_xoxo.emotion.train")
except Exception as e:
    print(f"Error loading Emotion model: {e}")

# Load authenticity (real/fake) model

auth_model = None
auth_classes = ['Fake', 'Real'] 
auth_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    from mlops_xoxo.utils.models import build_model
    
    AUTH_PATH = project_root / "models/fake_classification/model_best.pth"
    
    if AUTH_PATH.exists():
        # Load checkpoint to get metadata 
        checkpoint = torch.load(AUTH_PATH, map_location=DEVICE)
        
        backbone_name = checkpoint.get("backbone", "mobilenet_v3_small")
        auth_classes = checkpoint.get("classes", ['Fake', 'Real'])
        
        auth_model = build_model(backbone=backbone_name, num_classes=len(auth_classes), pretrained=False)
        
        # Load weights
        auth_model.load_state_dict(checkpoint["state_dict"])
        auth_model.to(DEVICE)
        auth_model.eval()
        print(f"Authenticity model loaded ({backbone_name}).")
    else:
        print(f"Authenticity model file not found at {AUTH_PATH}")
except Exception as e:
    print(f"Error loading Authenticity model: {e}")
    
# --- FastAPI Application ---

# Main application instance that handles web requests
app = FastAPI(title="XOFace Multi-Model API", description="Unified API for facial analysis.")

async def read_imagefile(file) -> Image.Image:
    contents = await file.read()
    try:
        return Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@app.get("/info", summary="Get model information")
def get_model_info():
    """
    Returns metadata about all currently loaded models.
    """
    info = {
        "service_name": "XOFace Multi-Model API",
        "device": str(DEVICE),
        "models": {}
    }

    # Face Embedding Info
    if embedding_model is not None:
        info["models"]["face_embedding"] = {
            "status": "loaded",
            "model_type": "MobileNetV2 + ArcFace",
            "embedding_size": 512,
            "input_shape": "(3, 112, 112)"
        }
    else:
        info["models"]["face_embedding"] = {"status": "not_loaded"}

    # Age/Gender Info
    if age_model is not None and gender_model is not None:
        info["models"]["age_gender"] = {
            "status": "loaded",
            "age_model_type": "SimpleModel (Regression)",
            "gender_model_type": "SimpleModel (Classification)",
            "input_shape": "(3, 200, 200)"
        }
    else:
        info["models"]["age_gender"] = {"status": "not_loaded"}

    # Emotion Info
    if emotion_model is not None:
        info["models"]["emotion"] = {
            "status": "loaded",
            "model_type": "ResNet18 (Grayscale)",
            "classes": EMOTION_CLASSES,
            "input_shape": "(1, 48, 48)"
        }
    else:
        info["models"]["emotion"] = {"status": "not_loaded"}

    # Authenticity Info
    if auth_model is not None:
        info["models"]["authenticity"] = {
            "status": "loaded",
            "model_type": "FakeDetectionModel",
            "classes": auth_classes if 'AUTH_CLASSES' in globals() else ['Fake', 'Real'],
            "input_shape": "(3, 224, 224)"
        }
    else:
        info["models"]["authenticity"] = {"status": "not_loaded"}

    return info

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/", summary="Root endpoint")
def read_root():
    """Provides basic information about the API."""
    return {"message": "Welcome to the XOFace Multi-Model API. Use the /predict endpoint to generate embeddings, /info to get"
    "model metadata."}   

@app.post("/predict_embedding", summary="Generate face embedding")
async def predict_embedding(file: UploadFile = File(..., description="Image file of a face")):
    """
    Accepts an image file, processes it, and returns the 512-dimensional face embedding.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded.")
    
    img = await read_imagefile(file)
    tensor = embedding_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = embedding_model(tensor)
    return {"embedding": embedding.cpu().numpy().flatten().tolist()}

@app.post("/predict_age_gender", summary="Predict Age and Gender")
async def predict_age_gender(file: UploadFile = File(..., description="Image file of a face")):
    """
    Accepts an image and returns predicted age and gender indices.
    """
    if age_model is None or gender_model is None:
        raise HTTPException(status_code=503, detail="Age/Gender models not loaded.")
    
    # Read and preprocess image
    img = await read_imagefile(file)
    tensor = age_gender_transform(img).unsqueeze(0).to(DEVICE)   # [1, C, H, W]

    with torch.no_grad():
        age_out = age_model(tensor)
        gender_out = gender_model(tensor)
    
    gender_idx = torch.argmax(gender_out.flatten()).item()
    gender_label = "Male" if gender_idx == 0 else "Female" 

    gender_probs = torch.softmax(gender_out.flatten(), dim=0)
    gender_confidence = gender_probs[gender_idx].item()
    
    return {
        "predicted_age": round(age_out.item(), 1),   # if age_out is scalar
        "gender_label": gender_label,
        "gender_confidence": round(gender_confidence, 4)
    }

@app.post("/predict_emotion", summary="Predict Emotion")
async def predict_emotion(file: UploadFile = File(...)):
    if emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded.")
    
    img = await read_imagefile(file)
    img_gray = img.convert('L')
    tensor = emotion_transform(img_gray).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = emotion_model(tensor)
        probs = torch.softmax(logits, dim=1)
    
    pred_idx = torch.argmax(probs, dim=1).item()
    return {
        "emotion": EMOTION_CLASSES[pred_idx],
        "confidence": probs[0][pred_idx].item()
    }

@app.post("/predict_authenticity", summary="Predict if the image is real or fake")
async def predict_authenticity(file: UploadFile = File(...)):
    if auth_model is None:
        raise HTTPException(status_code=503, detail="Authenticity model not loaded.")
    
    img = await read_imagefile(file)
    tensor = auth_transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = auth_model(tensor)
        probs = torch.softmax(logits, dim=1)
        
    pred_idx = torch.argmax(probs, dim=1).item()
    predicted_class = auth_classes[pred_idx]
    
    return {
        "prediction": predicted_class,
        "confidence": probs[0][pred_idx].item(),
        "is_fake": predicted_class.lower() == "fake"
    }

# Prometheus Instrumentator - API exporter
Instrumentator().instrument(app).expose(app)

# Running the app
if __name__ == "__main__":
    # Access at http://127.0.0.1:8000 or http://127.0.0.1:8000/docs
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, app_dir="api")