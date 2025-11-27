from fastapi import FastAPI, UploadFile, File
from typing import List, Optional
import uuid
import shutil
import pandas as pd
import polars as pl

from .header_extraction import HeaderExtractor
from .header_classifier import HeaderClassifier
from .deduplication_utils import Deduplicator
from .validation_utils import PhoneValidator, CitizenshipValidator
from .header_labelling import standardize_headers

app = FastAPI(title="Qdrant Header Processing Service")


@app.post("/upload/")
async def upload_csv(
    file: UploadFile = File(...),
    standardize_headers: Optional[bool] = False,
):
    """
    Unified endpoint:
    1. Upload and extract headers
    2. Classify them using Qdrant
    3. Deduplicate phone_number / citizenship_number columns
    4. Validate the detected columns
    """

    # ----------------------------------
    # Step 1: Save file
    # ----------------------------------
    file_id = uuid.uuid4()
    file_path = f"/tmp/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ----------------------------------
    # Step 2: Extract Headers
    # ----------------------------------
    extractor = HeaderExtractor()
    extraction_result = extractor.extract_headers_from_file(file_path, file_id)
    headers = extraction_result["headers"]

    # ----------------------------------
    # Step 3: Classify Headers
    # ----------------------------------
    classifier = HeaderClassifier()
    classified = classifier.classify_headers(headers)

    # Find best phone and citizenship matches
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

    # ----------------------------------
    # Step 4: Deduplication
    # ----------------------------------
    df = pd.read_csv(file_path)
    dedup_results = {}
    deduplicator = Deduplicator()
    columns_to_check = []

    if best_phone_col and best_phone_col in df.columns:
        columns_to_check.append((best_phone_col, "phone"))
    if best_citizenship_col and best_citizenship_col in df.columns:
        columns_to_check.append((best_citizenship_col, "citizenship"))

    if columns_to_check:
        dedup_results = deduplicator.deduplicate_dataframe(df, columns_to_check)
    else:
        dedup_results = {"message": "No deduplicatable columns detected"}

    # ----------------------------------
    # Step 5: Validation
    # ----------------------------------
    df_polars = pl.from_pandas(df)

    if best_phone_col and best_phone_col in df.columns:
        phone_validator = PhoneValidator(best_phone_col)
        df_polars = phone_validator.transform(df_polars)

    if best_citizenship_col and best_citizenship_col in df.columns:
        citizenship_validator = CitizenshipValidator(best_citizenship_col)
        df_polars = citizenship_validator.transform(df_polars)

    validated_df = df_polars.to_pandas()

    # ----------------------------------
    # Step 6: Optional DSPy Header Standardization
    # ----------------------------------
    standardized_headers = None
    if standardize_headers:
        standardized_headers = standardize_headers(headers)

    # ----------------------------------
    # Step 7: Unified Response
    # ----------------------------------
    response = {
        "file_id": str(file_id),
        "extracted_headers": headers,
        "classified_headers": classified,
        "deduplication_results": dedup_results,
        "validation_summary": validated_df.head(10).to_dict(orient="records"),
    }

    if standardized_headers is not None:
        response["standardized_headers"] = standardized_headers

    return response
