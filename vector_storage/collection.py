# qdrant_collection_manager.py
import logging
from qdrant_client.models import VectorParams, Distance

logger = logging.getLogger(__name__)

class QdrantCollectionManager:
    def __init__(self, client, collection_name: str, vector_size: int):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def init_collection(self):
        """Create or recreate the Qdrant collection."""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(
                f"✅ Initialized collection '{self.collection_name}' with vector size {self.vector_size}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize collection '{self.collection_name}': {e}")
            raise

    def clear_collection(self):
        """Delete and recreate the collection."""
        logger.info(f"Clearing collection '{self.collection_name}'")
        self.init_collection()
