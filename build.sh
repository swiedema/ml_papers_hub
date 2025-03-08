#!/bin/bash

# Exit on error
set -e

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # Source the environment file that uv creates
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available in PATH"
    echo "Current PATH: $PATH"
    exit 1
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
"$HOME/.local/bin/uv" venv .venv
source .venv/bin/activate

# Install dependencies from lock file
echo "Installing dependencies..."
"$HOME/.local/bin/uv" pip install -r requirements.txt

# Install your project as a package
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing project package..."
    "$HOME/.local/bin/uv" pip install .
fi

echo "Build completed successfully!"