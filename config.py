import os
from dotenv import load_dotenv
load_dotenv()

# Vertex AI
VERTEXAI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEXAI_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL = "gemini-2.5-flash"

# Chunking
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap characters between chunks

# Retrieval
TOP_K = 5
IMAGE_THRESHOLD = 0.3     # minimum score for image results
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2" #"all-MiniLM-L6-v2"

# Image extraction
MIN_IMAGE_SIZE = 50      # pixels, skip smaller images

# Cache
CACHE_DIR = "cache"
DB_DIR = "db"
