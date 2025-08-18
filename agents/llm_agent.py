import logging
from typing import Optional

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Log informational messages and above
    format='[LLMAgent] %(levelname)s: %(message)s'  # Prefix logs with [LLMAgent]
)
logger = logging.getLogger("LLMAgent")

# -----------------------------------------------------------------------------
# Attempt to import Ollama backend for local LLM calls
# -----------------------------------------------------------------------------
try:
    from agents.ollama_agent import query_ollama
    logger.info("Using Ollama local backend for LLM (model: tinyllama).")
except ImportError:
    logger.error("Ollama agent module not found. Please ensure agents/ollama_agent.py exists.")
    raise

# -----------------------------------------------------------------------------
# Central LLM call function using Ollama model
# -----------------------------------------------------------------------------
def call_llm(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 3000,
    model: str = "tinyllama"  # Default Ollama model
) -> str:
    """
    Call the Ollama LLM backend with the provided prompt and parameters.

    Args:
        prompt (str): Text prompt to send to the model.
        temperature (float): Sampling temperature controlling randomness.
        max_tokens (int): Maximum length of generated response.
        model (str): Which Ollama model to use.

    Returns:
        str: Generated text response from the LLM or error message.
    """
    logger.info(f"Calling Ollama LLM (model={model}) with prompt length {len(prompt)}")
    try:
        response = query_ollama(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model
        )
        if not response:
            logger.warning("Ollama LLM returned empty output.")
            return "❌ Ollama LLM returned empty response."
        return response
    except Exception as e:
        logger.exception("Error during Ollama LLM call")
        return f"❌ Failed to generate response: {e}"

# -----------------------------------------------------------------------------
# Example usage when run as a script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the benefits of using local LLMs in 3 concise points."
    result = call_llm(test_prompt)
    print("LLM Response:\n", result)
