import os
from pathlib import Path

HOME_DIR = Path.home()
MINDFORGE_DIR = HOME_DIR / ".mindforge"
MODELS_DIR = MINDFORGE_DIR / "models"

def ensure_dirs():
    """Ensures that the necessary directories exist."""
    MINDFORGE_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)