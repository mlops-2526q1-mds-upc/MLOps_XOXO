# Run the API from api.py
# And then run 'streamlit run app.py'

import streamlit as st
import requests
from PIL import Image
import io
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="XOFace Analysis UI",
    page_icon="🤖",
    layout="centered"
)

# Base URL of the running FastAPI
# FAST_API_URL = "http://127.0.0.1:8000"
#FAST_API_URL = os.getenv("FAST_API_URL", "http://127.0.0.1:8000")
FAST_API_URL = os.getenv("FAST_API_URL", "http://10.4.41.80:8000")


# Page Title 
st.title("XOFace - Face Analysis")
st.markdown("""
Upload an image to get a full facial analysis:
* **Face Embedding:** A 512-dimensional vector representation.
* **Demographics:** Estimated Age and Gender.
* **Emotion:** Predicted emotional state.
* **Authenticity:** Detection of Real/Fake (AI-generated) faces.
""")

# --- File Uploading ---
uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Prepare the file for the API request
    file_bytes = uploaded_file.getvalue()
    
    def get_file_payload():
        return {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}

    # Show a spinner while processing
    if st.button("Analyze Image"):
        with st.spinner("Analyzing image... Please wait."):
            
            # Call API Endpoints
            try:
                # Embedding
                resp_embed = requests.post(f"{FAST_API_URL}/predict_embedding", files=get_file_payload())
                
                # Age/Gender
                resp_age_gender = requests.post(f"{FAST_API_URL}/predict_age_gender", files=get_file_payload())

                # Emotion
                resp_emotion = requests.post(f"{FAST_API_URL}/predict_emotion", files=get_file_payload())
                
                # Authenticity (real/fake)
                resp_auth = requests.post(f"{FAST_API_URL}/predict_authenticity", files=get_file_payload())

                # Display Results
                st.subheader("Analysis Results")
                
                # Create columns for a grid layout
                col1, col2 = st.columns(2)

                # Column 1: Age/Gender & Emotion
                with col1:
                    st.info("**Age/Gender**")
                    if resp_age_gender.status_code == 200:
                        data = resp_age_gender.json()
                        st.write(f"**Gender:** {data['gender_label']}")
                        st.write(f"**Age:** {data['predicted_age']} years")
                        st.progress(data.get('gender_confidence', 0.0), text="Gender Confidence")
                    else:
                        st.error("Age/Gender model unavailable.")

                    st.info("**Emotion**")
                    if resp_emotion.status_code == 200:
                        data = resp_emotion.json()
                        st.metric(label="Emotion", value=data['emotion'])
                        st.progress(data['confidence'], text=f"Confidence: {data['confidence']:.2f}")
                    else:
                        st.error("Emotion model unavailable.")

                # Authenticity & Embedding 
                with col2:
                    st.info("**Authenticity**")
                    if resp_auth.status_code == 200:
                        data = resp_auth.json()
                        label = data['prediction']
                        
                        if data['is_fake']:
                            st.error(f"Prediction: {label}")
                        else:
                            st.success(f"Prediction: {label}")
                        st.write(f"**Confidence:** {data['confidence']:.4f}")
                    else:
                        st.error("Authenticity model unavailable.")

                    st.info("**Embedding**")
                    if resp_embed.status_code == 200:
                        data = resp_embed.json()
                        emb_len = len(data['embedding'])
                        st.write(f"**Vector Size:** {emb_len}")
                        with st.expander("View Embedding Vector"):
                            st.code(str(data['embedding']))
                    else:
                        st.error("Embedding model unavailable.")

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the API.")