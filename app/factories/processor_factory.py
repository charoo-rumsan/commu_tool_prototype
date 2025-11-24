"""Factory for creating processor instances."""

from typing import Optional, List, Dict, Any
from ..header_extraction import HeaderExtractor
from ..header_classifier import HeaderClassifier
from ..deduplication_utils import Deduplicator
from ..validation import PhoneValidator, CitizenshipValidator
from ..vector_search import VectorSearchEngine


class ProcessorFactory:
    """Factory class for creating processor instances."""

    @staticmethod
    def create_header_extractor() -> HeaderExtractor:
        """Create and return a HeaderExtractor instance."""
        return HeaderExtractor()

    @staticmethod
    def create_header_classifier(
        standard_labels: Optional[List[str]] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        threshold: float = 60.0
    ) -> HeaderClassifier:
        """
        Create and return a HeaderClassifier instance.
        
        Args:
            standard_labels: List of standard label names
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port number
            threshold: Similarity threshold for classification
            
        Returns:
            HeaderClassifier instance
        """
        return HeaderClassifier(
            standard_labels=standard_labels,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            threshold=threshold
        )

    @staticmethod
    def create_deduplicator(threshold: int = 85) -> Deduplicator:
        """
        Create and return a Deduplicator instance.
        
        Args:
            threshold: Similarity threshold for fuzzy matching (0-100)
            
        Returns:
            Deduplicator instance
        """
        return Deduplicator(threshold=threshold)

    @staticmethod
    def create_phone_validator(phone_col: str) -> PhoneValidator:
        """
        Create and return a PhoneValidator instance.
        
        Args:
            phone_col: Name of the phone column to validate
            
        Returns:
            PhoneValidator instance
        """
        return PhoneValidator(phone_col=phone_col)

    @staticmethod
    def create_citizenship_validator(citizenship_col: str) -> CitizenshipValidator:
        """
        Create and return a CitizenshipValidator instance.
        
        Args:
            citizenship_col: Name of the citizenship column to validate
            
        Returns:
            CitizenshipValidator instance
        """
        return CitizenshipValidator(citizenship_col=citizenship_col)

    @staticmethod
    def create_vector_search_engine(
        collection_name: str = "dataset_headers",
        vector_size: int = 384,
        host: Optional[str] = None,
        port: Optional[int] = None
    ) -> VectorSearchEngine:
        """
        Create and return a VectorSearchEngine instance.
        
        Args:
            collection_name: Name of the Qdrant collection
            vector_size: Size of the vectors
            host: Qdrant host address
            port: Qdrant port number
            
        Returns:
            VectorSearchEngine instance
        """
        return VectorSearchEngine(
            collection_name=collection_name,
            vector_size=vector_size,
            host=host,
            port=port
        )

