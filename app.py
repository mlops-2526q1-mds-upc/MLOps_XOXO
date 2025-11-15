import streamlit as st
import requests
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="XOFace Analysis UI",
    page_icon="🤖",
    layout="centered"
)

# Base URL of the running FastAPI
FAST_API_URL = "http://127.0.0.1:8000"

# Page Title 
st.title("XOFace - Face Analysis")
st.text("Upload an image to get the face embedding and age/gender prediction.")

# --- File Uploading ---
uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Prepare the file for the API request
    # We need to send the file's raw bytes
    file_bytes = uploaded_file.getvalue()
    files_for_api = {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}

    # Show a spinner while processing
    with st.spinner("Analyzing image... Please wait."):
        
        # Call API Endpoints
        try:
            # Call the embedding endpoint
            response_embed = requests.post(f"{FAST_API_URL}/predict_embedding", files=files_for_api.copy())
            
            # Call the age/gender endpoint
            response_age_gender = requests.post(f"{FAST_API_URL}/predict_age_gender", files=files_for_api.copy())
            
            # Display Results
            st.subheader("Analysis Results")
            
            # Display Age/Gender Results
            if response_age_gender.status_code == 200:
                data = response_age_gender.json()
                st.success("Age & Gender Prediction Successful!")
                # You can map these indices to real labels
                # e.g. gender_map = {0: "Male", 1: "Female"}
                st.metric(label="Predicted Gender (Index)", value=data['gender_index'])
                st.metric(label="Predicted Age (Index)", value=data['age_index'])
            else:
                st.error(f"Age/Gender Error: {response_age_gender.json().get('detail')}")

            # Display Embedding Results
            if response_embed.status_code == 200:
                data = response_embed.json()
                st.success("Face Embedding Successful!")
                st.metric(label="Embedding Dimensions", value=len(data['embedding']))
                # Show a preview of the embedding (not all 512 values)
                st.write(f"**Embedding Preview:**")
                st.code(f"{data['embedding'][:5]}...")
            else:
                st.error(f"Embedding Error: {response_embed.json().get('detail')}")

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API. Is it running?")