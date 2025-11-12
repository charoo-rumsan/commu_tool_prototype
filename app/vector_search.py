from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

class VectorSearchEngine:
    def __init__(self, collection_name="dataset_headers", vector_size=384):
        # Connect to Qdrant running in Docker
        self.collection_name = collection_name
        self.client = QdrantClient(host="qdrant", port=6333)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_collection(vector_size)

    def _init_collection(self, vector_size: int):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def store_vectors(self, texts, metadata=None):
        vectors = self.encoder.encode(texts)
        payloads = metadata or [{} for _ in texts]
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vectors[i].tolist(), payload=payloads[i])
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return [p.id for p in points]
