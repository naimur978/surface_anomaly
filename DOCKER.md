# Docker Setup Guide

## Quick Start

### Build Image
```bash
docker build -t patchcore:latest .
```

### Training
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config:/app/config \
  patchcore:latest python scripts/train.py config/config.yaml
```

### Inference
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config:/app/config \
  patchcore:latest python scripts/inference.py \
    --model results/models/patchcore_surface.pkl \
    --folder data/surface/test \
    --output ./results/inference_latest \
    --visualize
```

### Using Docker Compose

**Train and infer:**
```bash
docker-compose up
```

**Train only:**
```bash
docker-compose run patchcore python scripts/train.py config/config.yaml
```

**Inference only (requires trained model):**
```bash
docker-compose run inference
```

## GPU Support

### NVIDIA GPU (CUDA)

**1. Install nvidia-docker:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**2. Build CUDA image:**
```bash
docker build -t patchcore:cuda --build-arg BASE_IMAGE=nvidia/cuda:12.1-runtime-ubuntu22.04 .
```

**3. Run with GPU:**
```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  patchcore:cuda python scripts/train.py config/config.yaml
```

**4. Or use docker-compose with GPU:**
```bash
# Uncomment GPU section in docker-compose.yml, then:
docker-compose up
```

## Image Size Optimization

- **Multi-stage build**: Builder stage is discarded, final image is ~2GB
- **slim base image**: Uses python:3.10-slim (150MB) instead of python:3.10 (900MB)
- **No caching of pip packages**: Uses `--no-cache-dir` flag

## Troubleshooting

**Container exits immediately:**
```bash
docker run -it patchcore:latest bash
# Then test imports manually
```

**Permission denied on volumes:**
```bash
# Run as user (Linux)
docker run -u $(id -u):$(id -g) -v $(pwd)/data:/app/data patchcore:latest ...
```

**Out of memory:**
```bash
docker run --memory=16g patchcore:latest ...
```

**Check logs:**
```bash
docker logs <container-id>
```

## CI/CD

Docker image is automatically built and tested on every push to `main` via GitHub Actions (`.github/workflows/docker.yml`).
