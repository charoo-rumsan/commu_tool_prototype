import csv
import os
import uuid
from pathlib import Path
import polars as pl
from .vector_search import VectorSearchEngine

class HeaderExtractor:
    def __init__(self):
        self.supported_formats = ['.csv', '.tsv', '.txt']

    def extract_headers_from_file(self, file_path: str, file_id: uuid.UUID):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")

        headers = self._extract_headers(file_path)
        print(f"✅ Extracted {len(headers)} headers from file.")

        # Store headers in Qdrant
        # vse = VectorSearchEngine(collection_name=f"dataset_headers_{file_id}")
        # qdrant_ids = vse.store_vectors(headers, [{"position": i} for i in range(len(headers))])
        # print(f"✅ Stored {len(qdrant_ids)} headers in Qdrant.")

        return {"headers": headers}

    def _extract_headers(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(1024)
            f.seek(0)
            delimiter = csv.Sniffer().sniff(sample).delimiter
        df = pl.read_csv(file_path, separator=delimiter, n_rows=0)
        return df.columns
