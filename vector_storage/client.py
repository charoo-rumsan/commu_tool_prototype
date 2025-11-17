# qdrant_client_manager.py
import logging
import time
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class QdrantClientManager:
    def __init__(self, host: str, port: int, max_retries: int = 3, retry_delay: int = 2):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = self._connect_to_qdrant()

    def _connect_to_qdrant(self):
        """Connect to Qdrant with retry logic."""
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Connecting to Qdrant at {self.host}:{self.port} "
                    f"(attempt {attempt + 1}/{self.max_retries})..."
                )
                client = QdrantClient(host=self.host, port=self.port)
                client.get_collections()
                logger.info(f"✅ Successfully connected to Qdrant at {self.host}:{self.port}")
                return client
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Failed to connect to Qdrant (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to connect to Qdrant after {self.max_retries} attempts: {e}")
                    raise RuntimeError(
                        f"Unable to connect to Qdrant at {self.host}:{self.port} after {self.max_retries} attempts."
                        f" Please ensure Qdrant is running. Error: {str(e)}"
                    ) from e
