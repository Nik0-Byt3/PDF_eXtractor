import os
from pathlib import Path
import torch

class BaseConfig:
    # === Logging ===
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # === Device Configuration ===
    DEVICE = os.getenv(
    "DEVICE",
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

    

    # === NLP Models ===
    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-it")
    SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


    # === Option Flags === 
    ENABLE_TRANSLATION = True
    ENABLE_SUMMARIZATION = True
    ENABLE_IMAGE_EXTRACTION = True
    
    
    DEBUG = True 

class DevConfig(BaseConfig):
    LOG_LEVEL = "DEBUG"
    OUTPUT_DIR = "output/dev"
