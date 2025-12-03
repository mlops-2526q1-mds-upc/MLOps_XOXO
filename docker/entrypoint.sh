#!/bin/bash
set -e

echo "🚀 XOFace API - Starting up..."

# Configurar DVC: prioridad: config.local > variables de entorno > config base
mkdir -p .dvc

if [ -f .dvc/config.local ]; then
    echo "📋 Using config.local as config (contains credentials)"
    cp .dvc/config.local .dvc/config
    # Verificar que se copió correctamente
    if [ -f .dvc/config ]; then
        echo "✅ Config file created successfully"
    else
        echo "❌ ERROR: Failed to create .dvc/config from config.local"
    fi
elif [ -n "$DVC_USER" ] && [ -n "$DVC_PASSWORD" ]; then
    echo "🔑 Creating config from environment variables (DVC_USER, DVC_PASSWORD)"
    cat > .dvc/config <<EOF
[core]
    remote = origin
['remote "origin"']
    url = https://dagshub.com/pawarit.jamjod/MLOps_XOXO.dvc
    auth = basic
    user = $DVC_USER
    password = $DVC_PASSWORD
EOF
elif [ -f .dvc/config ]; then
    echo "✅ Using existing .dvc/config"
else
    echo "⚠️  No DVC credentials found (config.local or DVC_USER/DVC_PASSWORD)"
    echo "⚠️  Models should be pre-downloaded in ./models or provide credentials"
fi

# Verificar si los modelos existen
# Los modelos se descargan en /app/models (volumen persistente montado)
MODELS_DIR="/app/models"

# Asegurar que las carpetas de modelos existan (creadas vacías en Dockerfile)
mkdir -p "$MODELS_DIR/face_embedding" \
         "$MODELS_DIR/age_gender" \
         "$MODELS_DIR/emotion_classification" \
         "$MODELS_DIR/fake_classification"

MODEL_FILES=(
    "face_embedding/mobilenetv2_arcface_model.pth"
    "age_gender/age_model.pt"
    "age_gender/gender_model.pt"
    "emotion_classification/best_model.pth"
    "fake_classification/model_best.pth"
)

check_models() {
    local missing=0
    for model_file in "${MODEL_FILES[@]}"; do
        if [ ! -f "$MODELS_DIR/$model_file" ]; then
            echo "⚠️  Model not found: $model_file"
            missing=1
        fi
    done
    return $missing
}

# Si los modelos no existen y hay credenciales DVC, intentar descargarlos
if ! check_models; then
    echo "📦 Some models are missing. Attempting to download with DVC..."
    
    # Verificar si tenemos configuración DVC (config.local o variables de entorno)
    HAS_DVC_CONFIG=false
    
    # Debug: verificar qué archivos existen
    echo "🔍 Checking for DVC config files..."
    ls -la .dvc/ 2>/dev/null || echo "   .dvc directory contents not accessible"
    
    if [ -f .dvc/config ]; then
        echo "🔑 DVC config found at .dvc/config. Attempting to download models..."
        HAS_DVC_CONFIG=true
    elif [ -n "$DVC_USER" ] && [ -n "$DVC_PASSWORD" ]; then
        echo "🔑 DVC credentials found in environment. Creating config and downloading models..."
        HAS_DVC_CONFIG=true
        # Crear config desde variables de entorno si no existe
        if [ ! -f .dvc/config ]; then
            cat > .dvc/config <<EOF
[core]
    remote = origin
['remote "origin"']
    url = https://dagshub.com/pawarit.jamjod/MLOps_XOXO.dvc
    auth = basic
    user = $DVC_USER
    password = $DVC_PASSWORD
EOF
        fi
    fi
    
    if [ "$HAS_DVC_CONFIG" = true ]; then
        cd /app
        
        # Verificar configuración DVC
        echo "🔍 Checking DVC configuration..."
        if dvc remote list 2>&1 | grep -q origin; then
            echo "✅ DVC remote 'origin' configured"
        else
            echo "⚠️  DVC remote 'origin' not found, but config exists"
        fi
        
        # Intentar descargar modelos con DVC desde cada pipeline
        echo "📥 Downloading models from DVC pipelines to $MODELS_DIR..."
        echo "   This may take a few minutes depending on model sizes..."
        echo "   Models will be downloaded to the persistent volume (empty folders created in Dockerfile)"
        
        # Descargar desde cada pipeline (solo el stage train que tiene los modelos)
        # Los pipelines tienen wdir: ../.. así que necesitamos cambiar al directorio del pipeline
        # y hacer pull desde ahí, o hacer pull desde la raíz especificando el archivo
        PULL_SUCCESS=false
        for pipeline_dir in pipelines/face_embedding pipelines/age_gender pipelines/emotion_classification pipelines/fake_classification; do
            if [ -f "$pipeline_dir/dvc.yaml" ]; then
                echo "   📦 Pulling models from $pipeline_dir..."
                echo "      → Will download to: $MODELS_DIR/$(basename $pipeline_dir)/"
                # Cambiar al directorio del pipeline y hacer pull desde ahí
                # DVC respetará el wdir: ../.. especificado en el dvc.yaml
                cd "$pipeline_dir"
                if dvc pull train 2>&1; then
                    echo "   ✅ Models from $pipeline_dir downloaded to $MODELS_DIR/$(basename $pipeline_dir)/"
                    PULL_SUCCESS=true
                else
                    echo "   ⚠️  Failed to pull train from $pipeline_dir, trying full pull..."
                    if dvc pull 2>&1; then
                        echo "   ✅ Models from $pipeline_dir downloaded (full pull) to $MODELS_DIR/$(basename $pipeline_dir)/"
                        PULL_SUCCESS=true
                    else
                        echo "   ❌ Failed to pull from $pipeline_dir"
                    fi
                fi
                cd - > /dev/null
            fi
        done
        
        # Verificar que los modelos se descargaron
        sleep 1
        if check_models; then
            echo "✅ All models downloaded successfully!"
        elif [ "$PULL_SUCCESS" = true ]; then
            echo "⚠️  Pull completed but some models may still be missing"
            check_models || true
        else
            echo "❌ Failed to download models with DVC."
            echo "⚠️  Check DVC credentials and remote configuration."
        fi
    else
        echo "⚠️  No DVC credentials found (config.local or DVC_USER/DVC_PASSWORD)."
        echo "⚠️  Please either:"
        echo "   1. Mount .dvc/config.local as volume (contains credentials)"
        echo "   2. Provide DVC credentials as environment variables (DVC_USER, DVC_PASSWORD)"
        echo "   3. Pre-download models to ./models directory"
        echo "   4. Models should be in: $MODELS_DIR"
    fi
else
    echo "✅ All required models found!"
fi

# Verificar una última vez después de intentar descargar
if ! check_models; then
    echo "⚠️  WARNING: Some models are still missing after download attempt."
    echo "⚠️  The API may have limited functionality."
    echo ""
    echo "📋 Missing models:"
    for model_file in "${MODEL_FILES[@]}"; do
        if [ ! -f "$MODELS_DIR/$model_file" ]; then
            echo "   - $model_file"
        fi
    done
    echo ""
    echo "💡 Solutions:"
    echo "   1. Check DVC credentials in .dvc/config.local"
    echo "   2. Verify models exist in DVC remote: dvc list models/"
    echo "   3. Pre-download models locally: dvc pull models/"
    echo ""
    echo "⚠️  Continuing anyway - API will start but some endpoints may fail..."
else
    echo "✅ All required models are available!"
fi

echo ""
echo "🌐 Starting API server..."

# Ejecutar el comando original (API o UI)
exec "$@"

