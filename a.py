import os
import time
import logging
import tracemalloc
from uuid import uuid4

import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from data import extracted_headers  # your headers list

# -----------------------------------------
# Logging setup
# -----------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------------------
# Utils for timing & memory tracking
# -----------------------------------------
def log_step(step_name, start_time):
    """Log elapsed time and current memory usage."""
    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"⏱️ Step '{step_name}' took {elapsed:.3f}s | Mem: {current / 1e6:.2f}MB (peak {peak / 1e6:.2f}MB)")
    return time.perf_counter()

# -----------------------------------------
# Custom local encoder loader
# -----------------------------------------
_encoder_cache = None

def _load_encoder(model_name="all-MiniLM-L6-v2", max_retries=3, retry_delay=2):
    """Load SentenceTransformer with retry logic and caching."""
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    for attempt in range(max_retries):
        try:
            logger.info(f"Loading SentenceTransformer model '{model_name}' (attempt {attempt+1}/{max_retries})...")
            encoder = SentenceTransformer(model_name)
            _encoder_cache = encoder
            logger.info("✅ Model loaded successfully")
            return encoder
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to load model: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ Failed to load model after {max_retries} attempts")
                raise e

# -----------------------------------------
# LangChain Embedding Wrapper
# -----------------------------------------
class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper around SentenceTransformer."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = _load_encoder(model_name=model_name)

    def embed_query(self, text: str):
        return self.model.encode(text, show_progress_bar=False)

    def embed_documents(self, texts):
        return [self.model.encode(t, show_progress_bar=False) for t in texts]

# -----------------------------------------
# Main Execution
# -----------------------------------------
if __name__ == "__main__":
    tracemalloc.start()
    global_start = time.perf_counter()

    logger.info("🚀 Initializing FAISS vector store with SentenceTransformer...")

    step_start = global_start
    embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    step_start = log_step("Model Load", step_start)

    # Initialize FAISS
    dim = len(embeddings.embed_query("test sentence"))
    index = faiss.IndexFlatL2(dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    step_start = log_step("FAISS Init", step_start)
    query_headers = [
        "name", "phone_number", "citizenship_number", "email", "address",
        "latitude", "longitude", "age", "gender", "district", "country"
    ]

    # Convert headers to Documents
    documents = [Document(page_content=header, metadata={"source": "header"}) for header in query_headers]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    step_start = log_step("Add Documents", step_start)

    # -------------------------------
    # Batch Similarity Search
    # -------------------------------
    query_headers = extracted_headers
    k = 2  # top-k results per query
    step_start = log_step("Prepare Batch Queries", step_start)

    # Embed queries using real embedder
    query_embeddings = [embeddings.embed_query(q) for q in query_headers]
    query_embeddings = np.vstack(query_embeddings).astype("float32")
    step_start = log_step("Embed Queries", step_start)

    # Perform batch search
    distances, indices = vector_store.index.search(query_embeddings, k)
    step_start = log_step("Batch Similarity Search", step_start)

    # Map FAISS indices back to Documents
    print("\n🔍 Batch Query Results:")
    for i, query in enumerate(query_headers):
        print(f"\nQuery: '{query}'")
        for j in range(k):
            idx = indices[i, j]
            score = distances[i, j]
            if idx < len(vector_store.docstore._dict):
                doc = vector_store.docstore._dict[vector_store.index_to_docstore_id[idx]]
                print(f"* [SIM={score:.4f}] {doc.page_content} [{doc.metadata}]")

    # -------------------------------
    # Save and reload FAISS index
    # -------------------------------
    save_path = "faiss_index"
    vector_store.save_local(save_path)
    step_start = log_step("Save Index", step_start)

    new_vs = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    docs = new_vs.similarity_search("stateful LLM applications", k=1)
    step_start = log_step("Reload Index", step_start)

    print("\n✅ Reloaded index works! Example search:")
    print(f"* {docs[0].page_content} [{docs[0].metadata}]")

    # -------------------------------
    # Final logging
    # -------------------------------
    total_time = time.perf_counter() - global_start
    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"🏁 Total runtime: {total_time:.3f}s | Final Mem: {current / 1e6:.2f}MB | Peak: {peak / 1e6:.2f}MB")
    tracemalloc.stop() 