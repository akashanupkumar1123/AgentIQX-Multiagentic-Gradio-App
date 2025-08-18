import logging
from typing import List, Dict
import numpy as np
from utils.faiss_utils import load_faiss_index, query_faiss, get_embed_model
from agents.llm_agent import call_llm

# -----------------------------------------------------------------------------
# Logging setup for this module
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[RAGAgent] %(levelname)s: %(message)s'
)
logger = logging.getLogger("RAGAgent")

# -----------------------------------------------------------------------------
# Constants and cached resources
# -----------------------------------------------------------------------------
FAISS_INDEX_PATH = "data/faiss_index"          # Path to saved FAISS index
FAISS_METADATA_PATH = "data/faiss_metadata.pkl"  # Path to corresponding metadata (text chunks)
CONTEXT_CHUNK_LIMIT = 4                        # Number of top relevant chunks to use as context

# Cached embedding model instance shared across the app
embed_model = get_embed_model()

def retrieve_relevant_chunks(query: str) -> List[str]:
    """
    Embed the input query, load the FAISS index and metadata,
    then retrieve the top most relevant text chunks.

    Args:
        query (str): User's query string.

    Returns:
        List[str]: List of retrieved relevant text chunks (strings).
    """
    logger.info("Embedding query for retrieval.")
    try:
        # Create embedding vector for the query
        query_embedding = embed_model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []

    # Load FAISS index and paired metadata
    index, metadata = load_faiss_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    if index is None or not metadata:
        logger.warning("No FAISS index or metadata found.")
        return []

    logger.info("Querying FAISS index for relevant context.")
    try:
        # Query the FAISS index for top-k relevant chunks
        retrieved_chunks = query_faiss(index, query_embedding, metadata, k=CONTEXT_CHUNK_LIMIT)
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks
    except Exception as e:
        logger.error(f"Error during FAISS query: {e}")
        return []

def build_context(chunks: List[str]) -> str:
    """
    Combine retrieved chunks into a single string with double newlines separating them.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        str: Concatenated context string for LLM prompt.
    """
    logger.info("Building context from retrieved chunks.")
    return "\n\n".join(chunks)

def answer_with_rag(query: str) -> Dict[str, str]:
    """
    Full Retrieval-Augmented Generation (RAG) pipeline:
    Retrieves relevant chunks and generates an answer by passing context + query to LLM.

    Args:
        query (str): User's question.

    Returns:
        Dict[str, str]: Dictionary containing:
            'answer': Generated answer text.
            'context': The context string used for answering.
    """
    logger.info("Starting RAG pipeline.")
    
    # Retrieve relevant context chunks
    chunks = retrieve_relevant_chunks(query)
    if not chunks:
        logger.warning("No relevant data found for the query.")
        return {"answer": "❌ No relevant data found in the index.", "context": ""}

    # Build context string for the LLM
    context = build_context(chunks)

    # Construct the prompt for answering
    prompt = (
        "Answer the question using the following context. "
        "Be precise and quote if necessary.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    logger.info("Calling LLM with constructed prompt.")

    try:
        # Generate answer using the LLM call interface
        answer = call_llm(prompt)
        logger.info("LLM call successful.")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = "❌ Failed to generate an answer due to an internal error."

    return {
        "answer": answer.strip(),
        "context": context
    }

# -----------------------------------------------------------------------------
# Example usage (commented out)
# -----------------------------------------------------------------------------
# response = answer_with_rag("What does the document say about GDPR?")
# print(response["answer"])
