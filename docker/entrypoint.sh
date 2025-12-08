#!/bin/bash
set -e

echo "🚀 XOFace API - Starting up..."

# ---------------------------------------------------------
# 🔧 Ensure DVC config exists (config.local or env vars)
# ---------------------------------------------------------

mkdir -p .dvc

if [ -f .dvc/config.local ]; then
    echo "📋 Using .dvc/config.local"
    cp .dvc/config.local .dvc/config
elif [ -n "$DVC_USER" ] && [ -n "$DVC_PASSWORD" ]; then
    echo "🔑 Writing .dvc/config from env credentials"
    cat > .dvc/config <<EOF
[core]
    remote = origin
['remote "origin"']
    url = https://dagshub.com/pawarit.jamjod/MLOps_XOXO.dvc
    auth = basic
    user = $DVC_USER
    password = $DVC_PASSWORD
EOF
else
    echo "⚠️ No DVC credentials found — will only run if models exist"
fi

# ---------------------------------------------------------
# 📁 Model file tracking
# ---------------------------------------------------------

MODELS_DIR="/app/models"

MODEL_FILES=(
    "face_embedding/mobilenetv2_arcface_model.pth"
    "age_gender/age_model.pt"
    "age_gender/gender_model.pt"
    "emotion_classification/best_model.pth"
    "fake_classification/model_best.pth"
)

mkdir -p "$MODELS_DIR"/{face_embedding,age_gender,emotion_classification,fake_classification}

# ---------------------------------------------------------
# 🔍 Check model availability
# ---------------------------------------------------------

check_models() {
    local missing=0
    for model_file in "${MODEL_FILES[@]}"; do
        if [ ! -f "$MODELS_DIR/$model_file" ]; then
            echo "⚠️ Missing: $model_file"
            missing=1
        fi
    done
    return $missing
}

# ---------------------------------------------------------
# 📥 Retry-based DVC pull for single artifact
# ---------------------------------------------------------

retry_pull() {
    local target=$1
    echo "➡️ Pulling $target"
    for attempt in 1 2 3; do
        echo "   Try $attempt: dvc pull $target"
        if dvc pull "$target" 2>/dev/null; then
            echo "   ✔ Pulled successfully"
            return 0
        fi
        echo "   ❌ Failed, retrying..."
        sleep 1
    done
    echo "   ❌ Giving up"
}

# ---------------------------------------------------------
# 📦 Pull each missing model individually
# ---------------------------------------------------------

if ! check_models; then
    echo "📦 Some models missing — checking if DVC auth exists..."

    if [ -f .dvc/config ] || { [ -n "$DVC_USER" ] && [ -n "$DVC_PASSWORD" ]; }; then
        echo "🔑 DVC config found — downloading artifacts..."

        cd /app

        for model_file in "${MODEL_FILES[@]}"; do
            if [ ! -f "$MODELS_DIR/$model_file" ]; then
                retry_pull "models/$model_file"
            fi
        done

        cd - > /dev/null
    else
        echo "⚠️ No DVC credentials — cannot pull missing models"
    fi
fi

echo ""
if check_models; then
    echo "✅ All required models available!"
else
    echo "⚠️ WARNING — Some models are still missing."
    echo "⚠️ The API can still start but functionality may be limited."
fi

echo ""
echo "🌐 Starting API server..."
exec "$@"