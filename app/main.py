from fastapi import FastAPI, UploadFile, File
from typing import List
import uuid
import shutil
import pandas as pd

from .header_extraction import HeaderExtractor
from .header_classifier import HeaderClassifier
from .deduplication_utils import Deduplicator

app = FastAPI(title="Header-Qdrant-Deduplication Service")

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    """
    1. Upload CSV file.
    2. Extract headers.
    3. Classify them using Qdrant + ML.
    4. Run deduplication for phone_number and citizenship_number.
    """

    # -----------------------------
    # Step 1: Save file temporarily
    # -----------------------------
    file_id = uuid.uuid4()
    file_path = f"/tmp/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -----------------------------
    # Step 2: Extract headers
    # -----------------------------
    extractor = HeaderExtractor()
    extraction_result = extractor.extract_headers_from_file(file_path, file_id)
    headers = extraction_result["headers"]

    # -----------------------------
    # Step 3: Classify headers
    # -----------------------------
    classifier = HeaderClassifier()
    classified = classifier.classify_headers(headers)

    # Identify best matches for phone_number and citizenship_number
    best_phone_col = None
    best_citizenship_col = None
    best_phone_score = 0
    best_citizenship_score = 0

    for item in classified:
        if item["predicted_label"] == "phone_number" and item["similarity"] > best_phone_score:
            best_phone_col = item["header"]
            best_phone_score = item["similarity"]
        elif item["predicted_label"] == "citizenship_number" and item["similarity"] > best_citizenship_score:
            best_citizenship_col = item["header"]
            best_citizenship_score = item["similarity"]

    # -----------------------------
    # Step 4: Deduplicate (if applicable)
    # -----------------------------
    dedup_results = {}
    df = pd.read_csv(file_path)

    deduplicator = Deduplicator()
    columns_to_check = []

    if best_phone_col and best_phone_col in df.columns:
        columns_to_check.append((best_phone_col, "phone"))
    if best_citizenship_col and best_citizenship_col in df.columns:
        columns_to_check.append((best_citizenship_col, "citizenship"))

    if columns_to_check:
        dedup_results = deduplicator.deduplicate_dataframe(df, columns_to_check)
    else:
        dedup_results = {"message": "No phone_number or citizenship_number columns detected"}

    # -----------------------------
    # Step 5: Return unified response
    # -----------------------------
    return {
        "file_id": str(file_id),
        "extracted_headers": headers,
        "classified_headers": classified,
        "deduplication_results": dedup_results,
    }
