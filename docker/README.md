# Docker Setup

## Quick Start

```bash
# Build and start services (API + UI)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Access

- **API**: http://localhost:8000/docs
- **UI**: http://localhost:8501

## Models

Models are automatically downloaded when the container starts if DVC credentials are provided:

### Option 1: Environment Variables

```bash
export DVC_USER=your_username
export DVC_PASSWORD=your_password
docker-compose up -d
```

### Option 2: Mount DVC Config File

If you have a `.dvc/config.local` file with credentials, add to `docker-compose.yml`:

```yaml
services:
  api:
    volumes:
      - ./dvc/config.local:/app/.dvc/config.local:ro
```

Models are stored in a **persistent volume** (`xoface-models`) that persists between container restarts.

### Manual Model Download

If you prefer to download models before running:

```bash
# Download models locally
dvc pull

# Copy to Docker volume
docker volume create xoface-models
docker run --rm -v xoface-models:/models -v $(pwd)/models:/source alpine sh -c "cp -r /source/* /models/"
```

## Persistent Volumes

The project uses Docker persistent volumes:
- **`xoface-models`**: ML models (persist between restarts)
- **`xoface-dvc-config`**: DVC configuration (optional, for credentials)

### Access Models

```bash
# View volume contents
docker run --rm -v xoface-models:/models alpine ls -la /models

# Backup models from volume
docker run --rm -v xoface-models:/models -v $(pwd):/backup alpine tar czf /backup/models_backup.tar.gz -C /models .
```

## Configuration

### Switch to GPU

Edit `docker-compose.yml` and change:
```yaml
services:
  api:
    build:
      dockerfile: docker/Dockerfile.cuda
    environment:
      - DEVICE=cuda
```

## Notes

- **All code is copied into the image** (self-contained image)
- **Models are stored in persistent volumes** (not lost on restart)
- Docker Compose automatically builds images if needed
- Device is configured via `DEVICE` environment variable (cpu/cuda)
