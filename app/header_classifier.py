# app/header_classifier.py
import logging
from typing import List, Dict, Any, Optional
from .fuzzy_utils import fuzzy_ratio
from .factories import ProcessorFactory

logger = logging.getLogger(__name__)

class HeaderClassifier:
    """Hybrid vector + fuzzy header classifier using Qdrant"""

    def __init__(
        self,
        standard_labels: Optional[List[str]] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        threshold: float = 60.0
    ):
        self.threshold = threshold
        self.standard_labels = standard_labels 
        self.vector_engine = ProcessorFactory.create_vector_search_engine(
            collection_name="standard_labels_collection",
            host=qdrant_host,
            port=qdrant_port,
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

            # Filter out matches with None text
            valid_matches = [m for m in vector_matches if m.get("text") is not None]
            if not valid_matches:
                continue

            # Pick top-1 as best, and others as remaining top-2
            best = valid_matches[0]
            others = valid_matches[1:]

            vector_score = best["score"] * 100
            fuzzy_score = fuzzy_ratio(header, best["text"])
            combined = (0.6 * vector_score) + (0.4 * fuzzy_score)
            match = combined >= self.threshold

            rounded_overall = round(combined, 2)

            # Prepare 'other_similar' with top-2 remaining matches and their scores
            other_similar = []
            for other in others:
                other_vector_score = other["score"] * 100
                other_fuzzy_score = fuzzy_ratio(header, other["text"])
                other_combined = round((0.6 * other_vector_score) + (0.4 * other_fuzzy_score), 2)
                other_similar.append({
                    "label": other["text"],
                    "vector_score": round(other_vector_score, 2),
                    "fuzzy_score": round(other_fuzzy_score, 2),
                    "overall_score": other_combined
                })

            results.append({
                "header": header,
                "predicted_label": best["text"],  # highest-ranked
                "other_similar": other_similar,   # next 2 in rank
                "vector_score": round(vector_score, 2),
                "fuzzy_score": round(fuzzy_score, 2),
                "overall_score": rounded_overall,
                "similarity": rounded_overall,  # backwards compatibility
                "match": match
            })

        return results
