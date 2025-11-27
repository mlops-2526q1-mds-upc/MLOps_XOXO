FROM python:3.13-slim

# Instalar dependencias del sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Instalar solo dependencias esenciales para producción (API + UI)
# Usar las mismas versiones que el proyecto (torch 2.8.0, torchvision 0.23.0)
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0

# Dependencias core para la API
RUN pip install --no-cache-dir \
    fastapi>=0.117.1 \
    uvicorn[standard]>=0.36.0 \
    python-multipart>=0.0.6 \
    pillow>=10.0.0 \
    numpy>=2.3.3 \
    pyyaml>=6.0 \
    pandas>=2.3.2 \
    scikit-learn>=1.5.0

# Dependencias para UI (Streamlit)
RUN pip install --no-cache-dir \
    streamlit>=1.51.0 \
    requests>=2.32.5

# DVC para descargar modelos (necesario en runtime)
RUN pip install --no-cache-dir \
    dvc>=3.63.0 \
    dvc-s3>=3.0.0

# Dependencias opcionales pero útiles
RUN pip install --no-cache-dir \
    opencv-python-headless>=4.8.0 \
    tqdm>=4.66.0 \
    python-dotenv>=1.0.0

# MLflow (necesario para importar módulos de mlops_xoxo)
RUN pip install --no-cache-dir "mlflow<3.0"

# Codecarbon (necesario para importar módulos, aunque solo se usa en entrenamiento)
RUN pip install --no-cache-dir "codecarbon>=3.0.5"

# Limpiar cache de pip y archivos temporales
RUN pip cache purge || true && \
    find /usr/local/lib/python3.13 -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.13 -name "*.pyc" -delete 2>/dev/null || true

# Crear directorio .dvc (config se maneja en entrypoint.sh)
RUN mkdir -p .dvc

# Copiar código del proyecto (incluye .git para que DVC funcione)
COPY . .

# Copiar script entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# config.local se maneja en entrypoint.sh (montado como volumen o creado desde env vars)

# Crear directorios vacíos para modelos (los modelos se descargan en entrypoint.sh desde los pipelines)
# Estos directorios se crearán en el volumen persistente cuando se monte
RUN mkdir -p models/face_embedding \
             models/age_gender \
             models/emotion_classification \
             models/fake_classification \
             reports \
             .dvc && \
    # Asegurar que las carpetas de modelos estén vacías (no copiar modelos en la imagen)
    rm -rf models/*/* 2>/dev/null || true

# Variables de entorno
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV DEVICE=cpu

# Exponer puertos
EXPOSE 8000 8501

# Usar entrypoint para descargar modelos si es necesario
ENTRYPOINT ["/entrypoint.sh"]

# Comando por defecto (puede ser sobrescrito por docker-compose)
CMD ["python", "api/api.py"]

