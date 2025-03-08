#!/bin/bash

# Exit on error
set -e

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.cargo/bin:$PATH"  # Add uv to PATH
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
uv venv .venv
source .venv/bin/activate

# Install dependencies from lock file
echo "Installing dependencies..."
if [ -f "uv.lock" ]; then
    echo "Installing from uv.lock..."
    uv pip sync
else
    echo "No uv.lock found, installing from requirements.txt..."
    uv pip install -r requirements.txt
fi

echo "Build completed successfully!"