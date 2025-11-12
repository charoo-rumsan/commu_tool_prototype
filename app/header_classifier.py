# app/header_classifier.py
import logging
from typing import List, Dict, Any
from .vector_search import VectorSearchEngine
from .fuzzy_utils import fuzzy_ratio

logger = logging.getLogger(__name__)

class HeaderClassifier:
    """Hybrid vector + fuzzy header classifier using Qdrant"""

    def __init__(
        self,
        standard_labels: List[str] = None,
        qdrant_host: str = "qdrant",
        qdrant_port: int = 6333,
        threshold: float = 60.0
    ):
        self.threshold = threshold
        self.standard_labels = standard_labels or [
            "name", "phone_number", "citizenship_number", "email", "address",
            "latitude", "longitude", "age", "gender", "district", "country"
        ]
        self.vector_engine = VectorSearchEngine(
            collection_name="standard_labels_collection",
            host=qdrant_host,
            port=qdrant_port
        )
        self._index_standard_labels()

    def _index_standard_labels(self):
        """Store standard labels in Qdrant"""
        self.vector_engine.clear_collection()
        metadata = [{"type": "standard_label"} for _ in self.standard_labels]
        self.vector_engine.store_vectors(self.standard_labels, metadata)
        logger.info(f"Indexed {len(self.standard_labels)} standard labels.")

    def classify_headers(self, headers: List[str]) -> List[Dict[str, Any]]:
        """Classify extracted headers with semantic + fuzzy matching"""
        results = []

        for header in headers:
            vector_matches = self.vector_engine.search(header, top_k=3)
            if not vector_matches:
                continue

            best = vector_matches[0]
            vector_score = best["score"] * 100
            fuzzy_score = fuzzy_ratio(header, best["text"])

            combined = (0.6 * vector_score) + (0.4 * fuzzy_score)
            match = combined >= self.threshold

            results.append({
                "header": header,
                "predicted_label": best["text"],
                "vector_score": round(vector_score, 2),
                "fuzzy_score": round(fuzzy_score, 2),
                "overall_score": round(combined, 2),
                "match": match
            })
        return results
