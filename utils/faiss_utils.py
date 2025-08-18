# utils/faiss_utils.py
import os
import faiss
import pickle
import logging
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Logger for FAISS utilities
# -----------------------------------------------------------------------------
logger = logging.getLogger("FaissUtils")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model identifier
USE_COSINE_SIMILARITY = False  # If True → cosine similarity (normalize vectors), else L2 distance

# -----------------------------------------------------------------------------
# Global embedding model cache
# -----------------------------------------------------------------------------
_embed_model = None

def get_embed_model() -> SentenceTransformer:
    """
    Lazily load and cache the SentenceTransformer model to avoid repeated loads.

    Returns:
        SentenceTransformer: Cached embedding model instance.
    """
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _embed_model

# -----------------------------------------------------------------------------
# Save FAISS index and metadata
# -----------------------------------------------------------------------------
def save_faiss_index(
    chunks: List[str],
    doc_id: str,
    index_dir: str = "data/faiss_indexes"
) -> bool:
    """
    Create embeddings for given text chunks, build a FAISS index,
    and save both index and metadata (chunks) to disk.

    Args:
        chunks (List[str]): Text chunks to embed and index.
        doc_id (str): Identifier for the document (used in filenames).
        index_dir (str): Directory to save FAISS index and metadata.

    Returns:
        bool: True if index and metadata saved successfully, else False.
    """
    os.makedirs(index_dir, exist_ok=True)
    try:
        logger.info("Generating embeddings for FAISS index...")
        model = get_embed_model()
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

        # Normalize vectors if cosine similarity is used
        if USE_COSINE_SIMILARITY:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        logger.info(f"Embedding dimension: {dim}")

        # Choose FAISS index type based on similarity metric
        if USE_COSINE_SIMILARITY:
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        else:
            index = faiss.IndexFlatL2(dim)  # L2/Euclidean distance

        # Add embeddings to the index
        index.add(embeddings)

        # Save FAISS index file
        index_path = os.path.join(index_dir, f"{doc_id}.index")
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to {index_path}")

        # Save original chunks as metadata
        metadata_path = os.path.join(index_dir, f"{doc_id}_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Metadata saved to {metadata_path}")

        return True
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        return False

# -----------------------------------------------------------------------------
# Load FAISS index and metadata
# -----------------------------------------------------------------------------
def load_faiss_index(
    index_path: str,
    metadata_path: str
) -> Tuple[Optional[faiss.Index], List[str]]:
    """
    Load a FAISS index and its associated metadata from disk.

    Args:
        index_path (str): Path to the saved FAISS index file.
        metadata_path (str): Path to the saved metadata pickle file.

    Returns:
        Tuple[faiss.Index or None, List[str]]:
            - FAISS index instance (or None on failure)
            - List of metadata entries (chunks)
    """
    try:
        logger.info(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        logger.info("Index loaded successfully.")

        logger.info(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"Metadata loaded successfully with {len(metadata)} items.")

        # Warn if index and metadata size mismatch
        if index.ntotal != len(metadata):
            logger.warning(
                f"FAISS index size ({index.ntotal}) and metadata length ({len(metadata)}) do not match."
            )

        return index, metadata
    except Exception as e:
        logger.error(f"Failed to load FAISS index or metadata: {e}")
        return None, []

# -----------------------------------------------------------------------------
# Query FAISS index
# -----------------------------------------------------------------------------
def query_faiss(
    index: faiss.Index,
    query_embedding: np.ndarray,
    metadata: List[str],
    k: int = 5
) -> List[str]:
    """
    Query a FAISS index with a single embedding and return top-k matching metadata entries.

    Args:
        index (faiss.Index): The FAISS index object.
        query_embedding (np.ndarray): Query embedding of shape (1, dim).
        metadata (List[str]): List of original text chunks (metadata).
        k (int): Number of top matches to return.

    Returns:
        List[str]: Retrieved metadata entries corresponding to top results.
    """
    if index is None:
        logger.error("FAISS index is None, cannot perform query.")
        return []
    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        logger.error(f"Invalid query_embedding shape {getattr(query_embedding, 'shape', 'unknown')}, expected (1, dim).")
        return []
    if k <= 0:
        logger.warning(f"Invalid k={k} given for query_faiss. Must be positive.")
        return []
    if not metadata:
        logger.warning("Metadata is empty, query results will be empty.")
        return []

    try:
        # Normalize query if cosine similarity is used
        if USE_COSINE_SIMILARITY:
            faiss.normalize_L2(query_embedding)

        # Ensure k does not exceed available vectors
        k = min(k, index.ntotal)

        # Perform FAISS search
        D, I = index.search(query_embedding, k)  # I has shape: (1, k)

        # Map FAISS indices to metadata texts
        results = [metadata[idx] for idx in I[0] if 0 <= idx < len(metadata)]
        return results
    except Exception as e:
        logger.error(f"Error querying FAISS index: {e}")
        return []
