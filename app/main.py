import os
import uuid
import time
import shutil
import pandas as pd
import psutil

from fastapi import FastAPI, UploadFile, File

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
    process = psutil.Process(os.getpid())
    overall_start = time.perf_counter()
    step_timings = {}

    file_id = uuid.uuid4()
    file_path = f"/tmp/{file.filename}"

    step_start = time.perf_counter()
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    step_timings["save_temp_file_seconds"] = round(time.perf_counter() - step_start, 4)

    # -----------------------------
    # Step 2: Extract headers
    # -----------------------------
    step_start = time.perf_counter()
    standard_labels = [
            "name", "phone_number", "citizenship_number", "email", "address",
            "latitude", "longitude", "age", "gender", "district", "country" ]
    extractor = HeaderExtractor()
    extraction_result = extractor.extract_headers_from_file(file_path, file_id)
    headers = extraction_result["headers"]
    step_timings["extract_headers_seconds"] = round(time.perf_counter() - step_start, 4)

    # -----------------------------
    # Step 3: Classify headers
    # -----------------------------

    step_start = time.perf_counter()
    classifier = HeaderClassifier(standard_labels=standard_labels)
    classified = classifier.classify_headers(headers)
    step_timings["classify_headers_seconds"] = round(time.perf_counter() - step_start, 4)

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
    step_start = time.perf_counter()
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
    step_timings["deduplicate_seconds"] = round(time.perf_counter() - step_start, 4)

    overall_duration = round(time.perf_counter() - overall_start, 4)

    # -----------------------------
    # Step 5: Capture resource usage snapshot
    # -----------------------------
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()
    disk_usage = psutil.disk_usage("/")

    resource_usage = {
        "process_rss_mb": round(memory_info.rss / (1024 * 1024), 2),
        "process_vms_mb": round(memory_info.vms / (1024 * 1024), 2),
        "process_memory_percent": round(process.memory_percent(), 2),
        "system_memory_percent": round(virtual_memory.percent, 2),
        "disk_total_gb": round(disk_usage.total / (1024 ** 3), 2),
        "disk_used_gb": round(disk_usage.used / (1024 ** 3), 2),
        "disk_free_gb": round(disk_usage.free / (1024 ** 3), 2),
        "disk_usage_percent": round(disk_usage.percent, 2),
    }

    # -----------------------------
    # Step 5: Return unified response
    # -----------------------------
    return {
        "file_id": str(file_id),
        "extracted_headers": headers,
        "classified_headers": classified,
        "deduplication_results": dedup_results,
        "metrics": {
            "timings_seconds": {**step_timings, "total_seconds": overall_duration},
            "resource_usage": resource_usage,
        },
    }
