import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# ----------------------
# Test API functionality
# ----------------------

# Import the FastAPI app object
from api.api import app

client = TestClient(app)

# Fixtures

@pytest.fixture
def example_face_image():
    """
    Creates a simple 100x100 RGB image in memory to simulate an uploaded file (example image).
    """
    # Create a red square image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

# General API tests

def test_root():
    """Test that the root endpoint returns a 200 status and welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test that the health check endpoint returns 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_model_info():
    """
    Test that the /info endpoint returns metadata for all four models.
    """
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    
    # Check top-level keys
    assert "service_name" in data
    assert "models" in data
    
    # Check that all 4 models are listed
    models = data["models"]
    assert "face_embedding" in models
    assert "age_gender" in models
    assert "emotion" in models
    assert "authenticity" in models

# Prediction Endpoint Tests

def test_predict_embedding(example_face_image):
    """Test the face embedding endpoint with a valid image."""
    files = {'file': ('test.jpg', example_face_image, 'image/jpeg')}
    response = client.post("/predict_embedding", files=files)
    
    if response.status_code == 503:
        pytest.skip("Embedding model not loaded, skipping test.")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    # MobileFace output size
    assert len(data["embedding"]) == 512

def test_predict_age_gender(example_face_image):
    """Test the age/gender prediction endpoint."""
    files = {'file': ('test.jpg', example_face_image, 'image/jpeg')}
    response = client.post("/predict_age_gender", files=files)
    
    if response.status_code == 503:
        pytest.skip("Age/Gender model not loaded, skipping test.")
        
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_age" in data
    assert "gender_label" in data
    assert "gender_confidence" in data

    assert isinstance(data["predicted_age"], float)
    assert data["gender_label"] in ["Male", "Female"]

def test_predict_emotion(example_face_image):
    """Test the emotion prediction endpoint."""
    files = {'file': ('test.jpg', example_face_image, 'image/jpeg')}
    response = client.post("/predict_emotion", files=files)
    
    if response.status_code == 503:
        pytest.skip("Emotion model not loaded, skipping test.")

    assert response.status_code == 200
    data = response.json()
    
    assert "emotion" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_predict_authenticity(example_face_image):
    """Test the authenticity prediction endpoint."""
    files = {'file': ('test.jpg', example_face_image, 'image/jpeg')}
    response = client.post("/predict_authenticity", files=files)
    
    if response.status_code == 503:
        pytest.skip("Authenticity model not loaded, skipping test.")

    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "is_fake" in data
    assert "confidence" in data
    assert isinstance(data["is_fake"], bool)

# Test to predict invalid image file in API calls
def test_predict_invalid_file_type():
    """Test that sending a non-image file results in a 400 error (or handled gracefully)."""

    files = {'file': ('test.txt', b"this is not an image", 'text/plain')}
    
    # Try calling endpoints
    response_emb = client.post("/predict_embedding", files=files)
    response_age_gender = client.post("/predict_age_gender", files=files)
    response_emotion = client.post("/predict_emotion", files=files)
    response_auth = client.post("/predict_authenticity", files=files)
    
    # Expecting a 400 Bad Request or 503 if model is missing 
    assert response_emb.status_code in [400, 503]
    assert response_age_gender.status_code in [400, 503]
    assert response_emotion.status_code in [400, 503]
    assert response_auth.status_code in [400, 503]