# agents/embed_agent.py
import os
import pickle
import logging
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Log informational messages and above
    format='[EmbedAgent] %(levelname)s: %(message)s'  # Prefix logs with [EmbedAgent]
)
logger = logging.getLogger("EmbedAgent")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Pretrained embedding model

# -----------------------------------------------------------------------------
# Global model cache
# -----------------------------------------------------------------------------
_cached_model = None  # Stores loaded model to avoid reloading repeatedly

def get_embed_model() -> SentenceTransformer:
    """
    Lazily load and cache the SentenceTransformer embedding model globally.
    Ensures only one instance is loaded per process.
    """
    global _cached_model
    if _cached_model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _cached_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _cached_model

# -----------------------------------------------------------------------------
# Embedder class
# -----------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        """
        Initialize Embedder instance with cached or newly loaded model.
        """
        self.model = get_embed_model()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts (List[str]): List of text segments.

        Returns:
            np.ndarray: 2D array of shape (num_texts x embedding_dim).
        """
        logger.info(f"Embedding {len(texts)} text chunks...")
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,  # Show progress while encoding
                convert_to_numpy=True    # Return numpy array
            )
            logger.info("Embedding complete.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            return np.array([])

    def save_embeddings(self, embeddings: np.ndarray, chunks: List[str], save_path: str):
        """
        Save embeddings and their corresponding text chunks to a pickle file.

        Args:
            embeddings (np.ndarray): The calculated embeddings.
            chunks (List[str]): Original text chunks.
            save_path (str): Output path for pickle file.
        """
        logger.info(f"Saving embeddings to {save_path} ...")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure folder exists
            with open(save_path, "wb") as f:
                pickle.dump({"embeddings": embeddings, "chunks": chunks}, f)
            logger.info("Embeddings saved successfully.")
        except Exception as e:
            logger.error(f"Could not save embeddings: {e}")

    def load_embeddings(self, path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings and text chunks from a pickle file.

        Args:
            path (str): Path to the saved embeddings file.

        Returns:
            (np.ndarray, List[str]): Loaded embeddings and corresponding chunks.
        """
        logger.info(f"Loading embeddings from {path} ...")
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info("Embeddings loaded successfully.")
            return data["embeddings"], data["chunks"]
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return np.array([]), []


# -----------------------------------------------------------------------------
# Example usage (commented out)
# -----------------------------------------------------------------------------
# embedder = Embedder()
# embeddings = embedder.embed_texts(["Hello world!", "How are you?"])
# embedder.save_embeddings(embeddings, ["Hello world!", "How are you?"], "data/embeddings/sample.pkl")
# loaded_embeds, loaded_chunks = embedder.load_embeddings("data/embeddings/sample.pkl")
