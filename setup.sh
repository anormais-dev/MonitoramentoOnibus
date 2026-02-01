#!/bin/bash
set -e

RESET='\033[0m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'

echo -e "${BLUE}🚌 Bus Monitoring Setup Helper${RESET}\n"

info() { echo -e "${BLUE}$1${RESET}"; }
success() { echo -e "${GREEN}$1${RESET}"; }
warn() { echo -e "${YELLOW}$1${RESET}"; }
error() { echo -e "${RED}$1${RESET}"; exit 1; }

info "Checking Python version..."
if ! command -v python3.11 &> /dev/null; then
    warn "Python 3.11 not found, trying python3..."
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
    fi
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python3.11"
fi
success "Using: $($PYTHON_CMD --version)"

if [ ! -f .env ]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    success ".env created (edit with your camera credentials)"
else
    success ".env already exists"
fi

if [ ! -d venv ]; then
    info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    success "Virtual environment created"
else
    success "Virtual environment already exists"
fi

info "Activating virtual environment..."
source venv/bin/activate
success "Virtual environment activated"

info "Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt
success "Dependencies installed"

info "Creating project directories..."
mkdir -p detections
mkdir -p classified/{correct,false,unknown}
mkdir -p dataset/{training,validation}/{correct,false}
success "Directories created"

info "Checking required files..."
files_ok=true

if [ ! -f yolov8n.pt ]; then
    warn "yolov8n.pt not found (will download on first run)"
    files_ok=false
else
    success "yolov8n.pt found"
fi

if [ ! -f bus_classifier_resnet18.pth ]; then
    warn "bus_classifier_resnet18.pth not found (download from repo)"
    files_ok=false
else
    success "bus_classifier_resnet18.pth found"
fi

if [ ! -f dataset_index.faiss ]; then
    warn "dataset_index.faiss not found (optional, will be generated)"
    files_ok=false
else
    success "dataset_index.faiss found"
fi

if [ "$files_ok" = true ]; then
    success "All required files present"
else
    warn "Some model files missing - see notes above"
fi

echo ""
echo -e "${GREEN}Setup complete!${RESET}"
echo ""
echo -e "${BLUE}Next steps:${RESET}"
echo "1. Edit .env with your camera credentials:"
echo -e "   ${YELLOW}nano .env${RESET}"
echo ""
echo "2. Run the identifier:"
echo -e "   ${YELLOW}source venv/bin/activate${RESET}"
echo -e "   ${YELLOW}python identifier.py${RESET}"
echo ""
echo "3. Or run with Docker:"
echo -e "   ${YELLOW}docker-compose up --build${RESET}"
echo ""
