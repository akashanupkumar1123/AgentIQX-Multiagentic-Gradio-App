# utils/chunking_utils.py
import logging
import re
from typing import List
import numpy as np
from utils.faiss_utils import get_embed_model  # Use shared cached model instance

# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("ChunkingUtils")

# -----------------------------------------------------------------------------
# Chunking and embedding utility class
# -----------------------------------------------------------------------------
class ChunkingUtils:
    def __init__(self):
        """
        Initialize the chunking utility without immediately loading the embedding model.
        This lazy-load approach helps improve speed when chunking is needed without embedding.
        """
        self.embedder = None  # Will be set when embed_chucks() is called
        logger.info("ChunkingUtils initialized (lazy load embedder)")

    # -------------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean input text by collapsing all whitespace characters into single spaces.

        Args:
            text (str): Raw input text.

        Returns:
            str: Cleaned text with normalized whitespace (no leading/trailing spaces).
        """
        cleaned = re.sub(r'\s+', ' ', text).strip()
        return cleaned

    # -------------------------------------------------------------------------
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split cleaned text into overlapping chunks of words.

        Args:
            text (str): Input text to split.
            chunk_size (int): Number of words per chunk.
            overlap (int): Number of words shared between consecutive chunks.

        Returns:
            List[str]: List of text chunks.
        """
        # Clean text before splitting
        text = self.clean_text(text)
        if not text:
            logger.warning("Empty text received for chunking.")
            return []

        # Basic validation of parameters
        if chunk_size <= 0:
            logger.error("chunk_size must be positive.")
            raise ValueError("chunk_size must be positive.")

        if overlap >= chunk_size:
            logger.warning("overlap >= chunk_size, reducing overlap to chunk_size - 1")
            overlap = max(chunk_size - 1, 0)

        # Split text into words
        words = text.split()
        chunks = []
        i = 0

        # Create overlapping word chunks
        while i < len(words):
            chunk = words[i:i + chunk_size]
            if chunk:  # Skip empty results
                chunks.append(" ".join(chunk))
            i += chunk_size - overlap

        logger.info(
            f"Chunked text into {len(chunks)} chunks "
            f"(chunk_size={chunk_size}, overlap={overlap})."
        )
        return chunks

    # -------------------------------------------------------------------------
    def embed_chunks(self, chunks: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Embed a list of text chunks into dense vectors using a shared embedding model.

        Args:
            chunks (List[str]): List of text chunks to embed.
            show_progress_bar (bool): Whether to show embedding progress in console.

        Returns:
            np.ndarray: Array of embeddings with shape (num_chunks, embedding_dim)
        """
        # Filter out empty or whitespace-only chunks
        chunks = [c for c in chunks if c.strip()]
        if not chunks:
            logger.warning("No valid chunks provided to embed.")
            return np.array([])

        # Lazy-load the embedding model from shared faiss_utils loader
        if self.embedder is None:
            logger.info("Loading shared cached embedding model from faiss_utils...")
            self.embedder = get_embed_model()
            logger.info("Embedding model loaded (shared instance in use).")

        logger.info(f"Embedding {len(chunks)} chunks.")

        # Encode chunks into dense vectors
        embeddings = self.embedder.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar
        )

        logger.info("Embedding complete.")
        return embeddings

# -----------------------------------------------------------------------------
# Example usage (commented out)
# -----------------------------------------------------------------------------
# chunker = ChunkingUtils()
# chunks = chunker.chunk_text(raw_text)  # Auto-cleans and creates word chunks
# embeddings = chunker.embed_chunks(chunks)  # Converts chunks into vector embeddings
