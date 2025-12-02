FROM python:3.13-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install everything in ONE pip layer (much smaller)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        fastapi>=0.117.1 \
        uvicorn[standard]>=0.36.0 \
        python-multipart>=0.0.6 \
        pillow>=10.0.0 \
        numpy>=2.3.3 \
        pyyaml>=6.0 \
        pandas>=2.3.2 \
        scikit-learn>=1.5.0 \
        streamlit>=1.51.0 \
        requests>=2.32.5 \
        dvc>=3.63.0 \
        dvc-s3>=3.0.0 \
        opencv-python-headless>=4.8.0 \
        tqdm>=4.66.0 \
        python-dotenv>=1.0.0 \
        "mlflow<3.0" \
        "codecarbon>=3.0.5"

# Copy project files
COPY . .

# Make entrypoint executable
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create model folders but keep empty
RUN mkdir -p \
    models/face_embedding \
    models/age_gender \
    models/emotion_classification \
    models/fake_classification \
    reports \
    .dvc

ENV PYTHONPATH=/app
ENV DEVICE=cpu

EXPOSE 8000 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "api/api.py"]