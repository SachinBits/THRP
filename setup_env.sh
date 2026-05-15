#!/usr/bin/env bash
set -euo pipefail

# setup_env.sh - Create a Python 3.11 virtualenv and install requirements
# Usage: ./setup_env.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="$ROOT_DIR/requirements.txt"
VENV_DIR="$ROOT_DIR/.venv"

echo "THRP setup: creating Python 3.11 venv and installing dependencies"

command_exists() { command -v "$1" >/dev/null 2>&1; }

# Prefer any already-installed python3.11
if command_exists python3.11; then
  PY=python3.11
elif command_exists pyenv && pyenv versions --bare | grep -q '^3.11'; then
  echo "Using pyenv-installed Python 3.11"
  PY=$(pyenv which python)
elif command_exists pyenv; then
  echo "pyenv found; installing Python 3.11.6 via pyenv"
  pyenv install -s 3.11.6
  pyenv local 3.11.6
  PY=$(pyenv which python)
elif command_exists conda || command_exists mamba; then
  echo "Conda found; creating conda env 'thrp' with Python 3.11"
  conda create -n thrp python=3.11 -y
  echo "Activate the conda env: conda activate thrp"
  echo "Then run: python -m pip install -r $REQ_FILE"
  exit 0
else
  echo "No python3.11, pyenv, or conda detected. Please install one of them." >&2
  echo "Recommended: install pyenv (https://github.com/pyenv/pyenv) or Miniforge/Conda." >&2
  exit 2
fi

echo "Using Python: $PY"

"$PY" -m venv "$VENV_DIR"
echo "Created venv at $VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "Attempting to install PyTorch first (CPU wheels) to avoid resolution issues"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.1,<3.0" "torchvision>=0.15.2,<0.28" || true

echo "Installing remaining requirements"
python -m pip install -r "$REQ_FILE"

echo "✓ Environment ready. Activate with: source $VENV_DIR/bin/activate"

exit 0
