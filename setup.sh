#!/bin/bash
cd "$(dirname "$0")"

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q

# Install core dependencies first
pip install streamlit opencv-python-headless numpy mediapipe Pillow scipy scikit-image -q

# Install PyTorch (CPU version for Mac)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

# Install segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git -q

# Install diffusers and related
pip install diffusers transformers accelerate safetensors -q

# Install gradio-client for API calls
pip install gradio-client -q

echo ""
echo "Setup complete! Now you can run:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"

