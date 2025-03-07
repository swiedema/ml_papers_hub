#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Format Python files with Black
echo "Formatting Python files with Black..."
black app/

# Lint Python files with Ruff
echo "Linting Python files with Ruff..."
ruff check app/ --fix

echo "Done!" 