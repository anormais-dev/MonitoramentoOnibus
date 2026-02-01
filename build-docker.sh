#!/bin/bash
set -e

REGISTRY="${REGISTRY:-localhost:5000}"
IMAGE_NAME="${IMAGE_NAME:-bus-monitor}"
TAG="${TAG:-latest}"
BUILDKIT_INLINE_CACHE="${BUILDKIT_INLINE_CACHE:-1}"

echo "Docker Build Optimization Script"
echo "=================================="
echo "Image: $REGISTRY/$IMAGE_NAME:$TAG"
echo ""

export DOCKER_BUILDKIT=1
export BUILDKIT_INLINE_CACHE=1

build_image() {
    local dockerfile=$1
    local tag=$2
    
    echo "Building $dockerfile as $tag..."
    docker build \
        --file "$dockerfile" \
        --tag "$REGISTRY/$IMAGE_NAME:$tag" \
        --progress=plain \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --compress \
        .
    
    echo ""
    echo "Image size:"
    docker images "$REGISTRY/$IMAGE_NAME:$tag" --format "table {{.Repository}}\t{{.Size}}"
}

echo "Step 1: Building development image (Dockerfile)..."
build_image "Dockerfile" "dev"

echo ""
echo "Step 2: Building production image (Dockerfile.prod)..."

if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, building with NVIDIA CUDA base..."
    build_image "Dockerfile.prod" "prod"
else
    echo "GPU not detected, skipping prod build (requires CUDA)"
    echo "  To build anyway: docker build -f Dockerfile.prod -t $REGISTRY/$IMAGE_NAME:prod ."
fi

echo ""
echo "Build complete!"
echo ""
echo "To run:"
echo "  docker run -it --rm \\\\
    --env-file .env \\\\
    -v \$(pwd)/classified:/app/classified \\\\
    -v \$(pwd)/detections:/app/detections \\\\
    $REGISTRY/$IMAGE_NAME:dev"
echo ""
echo "Or with docker-compose:"
echo "  docker-compose up -d"
