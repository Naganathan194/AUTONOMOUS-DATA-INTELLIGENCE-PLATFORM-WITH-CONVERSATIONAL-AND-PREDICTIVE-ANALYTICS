import re
import os
import logging
import time
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Gemini API (used for AI Insights on numerical data) ───────────────────────
load_dotenv()
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
_gemini_ready   = False
if _GEMINI_API_KEY:
    genai.configure(api_key=_GEMINI_API_KEY)
    _gemini_ready = True
    logging.info("Gemini API configured — model: gemini-1.5-flash")
else:
    logging.warning("GEMINI_API_KEY/GOOGLE_API_KEY not set — get_gemini_api_response() will return an error.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


def _ensure_gemini_configured() -> bool:
    """Lazy-init Gemini config so key updates don't require a process restart."""
    global _gemini_ready
    global _GEMINI_API_KEY
    if _gemini_ready:
        return True

    load_dotenv(override=False)
    key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return False

    genai.configure(api_key=key)
    _GEMINI_API_KEY = key
    _gemini_ready = True
    logging.info("Gemini API configured at runtime — model: gemini-1.5-flash")
    return True

# ── Ollama server config (used for all other AI tasks) ────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
TEXT_MODEL      = "llama3.1:8b"
VISION_MODEL    = "llava"

logging.info(f"Ollama REST backend: {OLLAMA_BASE_URL}  text={TEXT_MODEL}  vision={VISION_MODEL}")

# ── Performance profiles ───────────────────────────────────────────────────────
# num_ctx    – context window; smaller = faster prefill
# num_predict – max output tokens; caps rambling, speeds up completion
# temperature – lower = more focused/deterministic (better for structured tasks)
# top_p       – nucleus sampling threshold

_OPTS_FAST = {
    "num_ctx":      2048,
    "num_predict":   600,
    "temperature":   0.3,
    "top_p":         0.9,
}

_OPTS_THINKING = {
    "num_ctx":      2048,
    "num_predict":   800,
    "temperature":   0.2,
    "top_p":         0.9,
}

_OPTS_VISION = {
    "num_ctx":      2048,
    "num_predict":   512,
    "temperature":   0.3,
    "top_p":         0.9,
}

_MODEL_TYPE_OPTS = {
    "flash":    _OPTS_FAST,
    "lite":     _OPTS_FAST,
    "thinking": _OPTS_THINKING,
}

# Request timeouts (seconds)
_TEXT_TIMEOUT   = 120
_VISION_TIMEOUT = 180


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_response_text(text: str) -> str:
    """Remove markdown formatting, code fences, and excess whitespace."""
    if not text:
        return ""
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```', '', text)
    text = re.sub(r'[*`]', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _ollama_reachable() -> bool:
    """Quick ping to check if Ollama server is up."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Gemini API (AI Insights for numerical/tabular data) ───────────────────────

def get_gemini_api_response(prompt: str) -> str:
    """
    Call Gemini 1.5 Flash for fast, accurate AI insights on numerical data.
    Used exclusively by the /api/insights endpoint.
    Returns an error string (never raises) so the caller can handle gracefully.
    """
    if not _ensure_gemini_configured():
        return "Error: GEMINI_API_KEY or GOOGLE_API_KEY not configured. Add it to the .env file."
    if not prompt or not prompt.strip():
        return "Error: Empty prompt provided for Gemini insights."
    model = genai.GenerativeModel(GEMINI_MODEL)
    max_retries = 3
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Calling Gemini API: {GEMINI_MODEL} (attempt {attempt}/{max_retries})")
            response = model.generate_content(prompt)
            response_text = getattr(response, "text", "") or ""
            if not response_text:
                return "Error: Gemini returned an empty response."
            cleaned = clean_response_text(response_text)
            if not cleaned:
                return "Error: Gemini returned only empty/unsupported content."
            logging.info(f"Gemini response received ({len(cleaned)} chars)")
            return cleaned
        except Exception as e:
            last_error = str(e)
            err_lower = last_error.lower()
            is_quota = ("quota" in err_lower) or ("429" in err_lower) or ("rate" in err_lower)
            if not is_quota or attempt == max_retries:
                logging.error(f"Gemini API error: {e}")
                return f"Error: {e}"

            wait_s = 8.0
            m_seconds = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", last_error, flags=re.IGNORECASE)
            m_float = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)s", last_error, flags=re.IGNORECASE)
            if m_seconds:
                wait_s = float(m_seconds.group(1)) + 0.5
            elif m_float:
                wait_s = float(m_float.group(1)) + 0.5

            logging.warning(f"Gemini quota/rate-limited, retrying in {wait_s:.1f}s...")
            time.sleep(wait_s)

    return f"Error: {last_error}"


# ── Ollama text via REST ───────────────────────────────────────────────────────

def get_ollama_response(prompt: str, model: str = TEXT_MODEL,
                        options: dict = None) -> str:
    """Send a text prompt to Ollama REST API and return the cleaned reply."""
    opts = options or _OPTS_FAST
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        "options":  opts,
    }
    try:
        logging.info(f"POST {OLLAMA_BASE_URL}/api/chat  model={model}")
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=_TEXT_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"]
        cleaned = clean_response_text(text)
        logging.info(f"Ollama response received ({len(cleaned)} chars)")
        return cleaned
    except requests.exceptions.ConnectionError:
        logging.error("Ollama connection error — server not reachable.")
        return "AI service temporarily unavailable. Please check that Ollama is running (ollama serve)."
    except requests.exceptions.Timeout:
        logging.error("Ollama request timed out.")
        return "AI service timed out. Try a shorter prompt or restart Ollama."
    except Exception as e:
        logging.error(f"Ollama text error: {e}")
        return f"Error: {e}"


# ── Ollama vision via REST ─────────────────────────────────────────────────────

def get_ollama_vision_response(prompt: str, image_b64: str,
                               model: str = VISION_MODEL) -> str:
    """Send an image (base-64) + text prompt to LLaVA via Ollama REST API."""
    payload = {
        "model":   model,
        "prompt":  prompt,
        "images":  [image_b64],
        "stream":  False,
        "options": _OPTS_VISION,
    }
    try:
        logging.info(f"POST {OLLAMA_BASE_URL}/api/generate  model={model} (vision)")
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=_VISION_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        cleaned = clean_response_text(text)
        logging.info(f"Ollama vision response received ({len(cleaned)} chars)")
        return cleaned
    except requests.exceptions.ConnectionError:
        logging.error("Ollama connection error (vision).")
        return "Error: Cannot reach Ollama — make sure 'ollama serve' is running."
    except requests.exceptions.Timeout:
        logging.error("Ollama vision request timed out.")
        return "Error: Vision analysis timed out. Try restarting Ollama."
    except Exception as e:
        logging.error(f"Ollama vision error: {e}")
        return f"Error: {e}"


# ── Unified entry-point (called by main.py / generate_report.py / smart_query.py) ──

def get_gemini_response(prompt: str, model_type: str = "flash",
                        max_retries: int = 2) -> str:
    """
    Routes all text requests to Ollama REST API with tuned inference options.

    model_type selects a performance profile:
      "flash" / "lite"  → fast, compact output  (insights, commentary, report)
      "thinking"        → slightly more tokens   (smart-query code generation)
    """
    opts = _MODEL_TYPE_OPTS.get(model_type, _OPTS_FAST)
    payload = {
        "model":    TEXT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        "options":  opts,
    }

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Ollama [{model_type}] attempt {attempt}/{max_retries}")
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=_TEXT_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
            cleaned = clean_response_text(text)
            logging.info(f"Ollama [{model_type}] response ({len(cleaned)} chars)")
            return cleaned
        except requests.exceptions.ConnectionError as e:
            last_err = "Ollama server not reachable"
            logging.warning(f"Ollama [{model_type}] attempt {attempt} — connection error")
            break   # No point retrying if Ollama isn't running
        except requests.exceptions.Timeout as e:
            last_err = "request timed out"
            logging.warning(f"Ollama [{model_type}] attempt {attempt} — timeout")
        except Exception as e:
            last_err = str(e)
            logging.warning(f"Ollama [{model_type}] attempt {attempt} failed: {last_err}")

    logging.error(f"Ollama [{model_type}] failed: {last_err}")
    if "not reachable" in last_err or "connect" in last_err.lower():
        return "AI service temporarily unavailable. Please check that Ollama is running (ollama serve)."
    return f"Error: {last_err}"


def get_model_info() -> dict:
    """Return information about the currently configured models."""
    reachable = _ollama_reachable()
    status = "online" if reachable else "offline — run: ollama serve"
    return {
        "text_model":   f"{TEXT_MODEL} via Ollama REST ({status})",
        "vision_model": f"{VISION_MODEL} via Ollama REST ({status})",
        "ollama_url":   OLLAMA_BASE_URL,
    }
