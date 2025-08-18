# agents/explain_agent.py
import logging
from typing import Optional
from agents.llm_agent import call_llm  # Custom LLM call interface

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Log informational messages and above
    format='[ExplainAgent] %(levelname)s: %(message)s'
)
logger = logging.getLogger("ExplainAgent")

# -----------------------------------------------------------------------------
# Main explanation generator
# -----------------------------------------------------------------------------
def explain_answer(context: str, answer: str, temperature: Optional[float] = None) -> str:
    """
    Generate a step-by-step explanation for a given answer based on the provided context.

    Args:
        context (str): The context or background information used to derive the answer.
        answer (str): The answer that needs to be explained.
        temperature (Optional[float]): Sampling temperature for the LLM (if supported).

    Returns:
        str: A clear, simplified explanation of the reasoning behind the answer.
    """
    logger.info("Generating explanation for the given answer.")
    
    # Construct a prompt instructing the LLM to explain reasoning in simple step-by-step terms
    prompt = (
        "You are an AI assistant helping a student understand answers.\n\n"
        "Based on the following context and answer, explain the rationale step by step:\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Now explain the logic or reasoning clearly in simple terms."
    )
    
    # Call LLM with optional temperature parameter (if provided & supported)
    if temperature is not None:
        explanation = call_llm(prompt, temperature=temperature)
    else:
        explanation = call_llm(prompt)
    
    return explanation.strip()

# -----------------------------------------------------------------------------
# Example usage (commented out)
# -----------------------------------------------------------------------------
# explanation = explain_answer(
#     "Water boils at 100°C at sea level.",
#     "Because water molecules gain energy and change state."
# )
# print(explanation)
