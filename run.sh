#!/bin/bash
cd "$(dirname "$0")"

echo "Starting Virtual Try-On Suite..."
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found!"
    echo "Please run: python3.11 -m venv venv (or python3.12)"
    exit 1
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Error: Streamlit not found. Installing dependencies..."
    pip install streamlit -q
fi

# Run the app
echo "Launching Streamlit app..."
streamlit run app.py

