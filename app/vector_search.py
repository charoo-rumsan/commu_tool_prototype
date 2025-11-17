from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import time
import logging
import os
from typing import List, Dict, Optional

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

class VectorSearchEngine:
    def __init__(
        self,
        collection_name: str = "dataset_headers",
        vector_size: int = 384,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT

        # Connect to Qdrant with retry logic
        self.client = self._connect_to_qdrant()
        self.encoder = _load_encoder()
        self._init_collection(vector_size)

    def _connect_to_qdrant(self, max_retries=3, retry_delay=2):
        """Connect to Qdrant with retry logic."""
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to Qdrant at {self.host}:{self.port} "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                client = QdrantClient(host=self.host, port=self.port)
                # Test connection by getting collections
                client.get_collections()
                logger.info(f"✅ Successfully connected to Qdrant at {self.host}:{self.port}")
                return client
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to connect to Qdrant (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to Qdrant after {max_retries} attempts: {e}")
                    raise RuntimeError(
                        f"Unable to connect to Qdrant at {self.host}:{self.port} after {max_retries} attempts. "
                        f"Please ensure Qdrant is running. If running locally, use 'localhost'. "
                        f"If running in Docker, use the service name 'qdrant'. Error: {str(e)}"
                    ) from e

    def _init_collection(self, vector_size: int):
        """Initialize or recreate the Qdrant collection."""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(
                f"✅ Initialized collection '{self.collection_name}' with vector size {vector_size}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize collection '{self.collection_name}': {e}")
            raise

    def clear_collection(self):
        """Delete and recreate the collection."""
        logger.info(f"Clearing collection '{self.collection_name}'")
        self._init_collection(self.vector_size)

    def store_vectors(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        vectors = self.encoder.encode(texts)
        payloads = metadata or [{} for _ in texts]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload=self._build_payload(payloads[i], texts[i]),
            )
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return [p.id for p in points]

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """Search stored vectors for the closest matches to the query text."""
        if not query_text:
            return []

        query_vector = self.encoder.encode([query_text])[0].tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        formatted_results: List[Dict[str, float]] = []
        for hit in results:
            payload = hit.payload or {}
            formatted_results.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": payload,
                    "text": payload.get("text") or payload.get("label") or payload.get("value"),
                }
            )

        return formatted_results

    @staticmethod
    def _build_payload(payload: Dict, text: str) -> Dict:
        """Ensure payload contains the original text for downstream consumers."""
        payload_copy = dict(payload) if payload else {}
        payload_copy.setdefault("text", text)
        return payload_copy
