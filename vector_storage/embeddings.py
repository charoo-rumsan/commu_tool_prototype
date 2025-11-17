
from sentence_transformers import SentenceTransformer

import time
import logging
import os


logger = logging.getLogger(__name__)

# Get Qdrant connection settings from environment variables, with fallbacks
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Cache the encoder to avoid reloading on every request
_encoder_cache = None

def _load_encoder(model_name="all-MiniLM-L6-v2", max_retries=3, retry_delay=2):
    """Load SentenceTransformer with retry logic for network issues."""
    global _encoder_cache
    
    # Return cached encoder if available
    if _encoder_cache is not None:
        return _encoder_cache
    
    # First, try to load from local cache if available
    try:
        logger.info(f"Attempting to load SentenceTransformer model '{model_name}' from cache...")
        # Try with local_files_only parameter (may not be available in all versions)
        try:
            encoder = SentenceTransformer(model_name, local_files_only=True)
        except TypeError:
            # If local_files_only is not supported, try without it
            encoder = SentenceTransformer(model_name)
        _encoder_cache = encoder
        logger.info("✅ Successfully loaded SentenceTransformer model from cache")
        return encoder
    except Exception as cache_error:
        logger.info(f"Model not found in cache, will attempt to download: {cache_error}")
    
    # If not in cache, try downloading with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading SentenceTransformer model '{model_name}' (attempt {attempt + 1}/{max_retries})...")
            encoder = SentenceTransformer(model_name)
            _encoder_cache = encoder
            logger.info("✅ Successfully loaded SentenceTransformer model")
            return encoder
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to load model (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to load model after {max_retries} attempts: {e}")
                raise RuntimeError(
                    f"Unable to load SentenceTransformer model '{model_name}' after {max_retries} attempts. "
                    f"Please check your internet connection and try again. Error: {str(e)}"
                ) from e