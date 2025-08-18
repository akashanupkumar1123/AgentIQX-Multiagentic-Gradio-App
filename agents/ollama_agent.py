import subprocess
import logging
import requests
import json  # needed for parsing streamed lines

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[OllamaAgent] %(levelname)s: %(message)s'
)
logger = logging.getLogger("OllamaAgent")

def supports_temperature_flag() -> bool:
    """
    Check if the installed ollama CLI supports the --temperature flag.
    This affects how we invoke the CLI (with or without temperature option).
    """
    try:
        # Run `ollama run model --help` and check for --temperature in help text
        result = subprocess.run(
            ["ollama", "run", "tinyllama", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        help_text = result.stdout + result.stderr
        return '--temperature' in help_text
    except Exception as e:
        logger.warning(f"Failed to check Ollama CLI flags: {e}")
        return False

def query_ollama(
    prompt: str,
    model: str = 'tinyllama',
    temperature: float = 0.7,
    max_tokens: int = 3000
) -> str:
    """
    Query Ollama model, choosing CLI or HTTP API based on CLI capabilities.
    """
    if supports_temperature_flag():
        logger.info("Ollama CLI supports --temperature, calling CLI with flags.")
        return _query_ollama_cli(prompt, model, temperature, max_tokens)
    else:
        logger.info("Ollama CLI lacks --temperature support, calling HTTP API (stream-parsed).")
        return _query_ollama_api(prompt, model, temperature, max_tokens)

def _query_ollama_cli(prompt, model, temperature, max_tokens):
    """
    Query Ollama using CLI command with temperature and max tokens flags.
    """
    try:
        cmd = [
            "ollama",
            "run",
            model,
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
            "--quiet"
        ]
        result = subprocess.run(
            cmd,
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        output = result.stdout.decode('utf-8').strip()
        if not output:
            logger.error(f"No output from Ollama CLI. STDERR: {result.stderr.decode('utf-8').strip()}")
            return "❌ Ollama returned no content."
        logger.info("Received response from Ollama CLI.")
        return output
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else ''
        logger.error(f"Ollama CLI error: {stderr}")
        return f"❌ Ollama CLI call failed: {stderr}"
    except FileNotFoundError:
        logger.error("Ollama CLI not found in PATH.")
        return "❌ Ollama CLI not found."
    except Exception as e:
        logger.error(f"Ollama CLI unexpected error: {e}")
        return "❌ Ollama internal error."

def _query_ollama_api(prompt, model, temperature, max_tokens):
    """
    Query Ollama via HTTP API with safe streaming JSON parsing.
    Handles the streamed response line-by-line.
    """
    try:
        # Make POST request with streaming enabled
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,  # we will process line-by-line
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            },
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        output_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipped invalid JSON line: {line}")
                continue
            # Append chunk of LLM output if present
            if "response" in data and data["response"]:
                output_text += data["response"]
            # Stop if Ollama sends a 'done' flag
            if data.get("done", False):
                break

        if not output_text.strip():
            logger.warning("Ollama API returned no content.")
            return "❌ Ollama API returned no content."
        logger.info("Received streamed response from Ollama API.")
        return output_text.strip()

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return f"❌ Ollama API request failed: {e}"
