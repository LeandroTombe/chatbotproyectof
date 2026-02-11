"""
Configuration settings for the ChatBot project.
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
API_KEY = os.getenv("API_KEY", "your-api-key-here")
API_URL = os.getenv("API_URL", "https://api.example.com")

# Model Configuration
MODEL_NAME = "default-model"
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# Database Configuration (if needed)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chatbot.db")

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", 5000))
HOST = os.getenv("HOST", "127.0.0.1")

# Chroma Configuration
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(exist_ok=True)

CHROMA_PERSIST = True
CHROMA_COLLECTION_NAME = "chatbot_collection"