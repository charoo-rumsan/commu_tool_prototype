# vector_search_engine.py
import os
import uuid
from typing import List, Dict, Optional
import logging

from qdrant_client.models import PointStruct
from embeddings import _load_encoder
from .qdrant_client_manager import QdrantClientManager
from .qdrant_collection_manager import QdrantCollectionManager

logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

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

        # Initialize client and collection manager
        client_manager = QdrantClientManager(self.host, self.port)
        self.client = client_manager.client
        self.encoder = _load_encoder()

        self.collection_manager = QdrantCollectionManager(
            self.client, self.collection_name, self.vector_size
        )
        self.collection_manager.init_collection()

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
