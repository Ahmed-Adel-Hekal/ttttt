"""core/gemini_client.py — Unified LLM Agent (Google + OpenRouter + Groq)

Providers:
  - google     → Google Gemini (google-genai / google-generativeai SDK)
  - openrouter → OpenRouter   (OpenAI-compatible, api.openrouter.ai)
  - groq       → Groq         (OpenAI-compatible, api.groq.com)

Fixes vs original:
  1. _ask_openrouter: message.content can be None/empty when the model uses
     reasoning/thinking tokens (e.g. claude-3-5-sonnet, deepseek-r1, o3-mini).
     Now also checks message.reasoning_content and reconstructs from raw
     choices[0].message dict as a last resort.
  2. _ask_openrouter: logs the HTTP response id + model for traceability.
  3. _ask_openrouter: disables reasoning by default for JSON-output tasks so
     content is always in the expected field; callers that need reasoning
     can pass reasoning_enabled=True explicitly.
  4. _ask_google: handles both response.text and response.candidates paths
     so older generativeai SDK versions don't silently return "".
  5. ask(): logs the actual response length on success for easier debugging.
"""
from __future__ import annotations
import os
import time
import importlib
import importlib.util
import warnings

if importlib.util.find_spec("dotenv"):
    importlib.import_module("dotenv").load_dotenv()

genai        = importlib.import_module("google.genai")           if importlib.util.find_spec("google.genai")        else None
types        = importlib.import_module("google.genai.types")     if genai                                           else None
if importlib.util.find_spec("google.generativeai"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        generativeai = importlib.import_module("google.generativeai")
else:
    generativeai = None
openai_module = importlib.import_module("openai") if importlib.util.find_spec("openai") else None
OpenAI        = openai_module.OpenAI if openai_module else None

from core.logger import get_logger
logger = get_logger("Agent")


def _extract_openrouter_content(message) -> str:
    """
    Robustly extract text content from an OpenRouter chat message.

    OpenRouter models that use reasoning/thinking tokens (DeepSeek-R1,
    Claude extended thinking, o3-mini, etc.) sometimes return:
      - message.content = None  (all text in reasoning_content)
      - message.content = ""    (same)
      - message.content = "<think>…</think>"  (reasoning wrapped in tags)

    We try four sources in order:
      1. message.content — the normal field
      2. message.reasoning_content — dedicated reasoning field (DeepSeek-R1)
      3. Iterate message.content_parts / parts if present (some SDKs)
      4. Raw __dict__ / model_dump() fallback
    """
    # 1. Standard content field
    content = getattr(message, "content", None)
    if content and isinstance(content, str) and content.strip():
        # Strip <think>…</think> wrapper that some reasoning models emit
        import re
        cleaned = re.sub(r"<think>[\s\S]*?</think>\s*", "", content, flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned
        # If only thinking tags, fall through to reasoning_content

    # 2. Dedicated reasoning_content field (DeepSeek-R1 on OpenRouter)
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # 3. reasoning_details list (older OpenRouter format)
    details = getattr(message, "reasoning_details", None)
    if details and isinstance(details, list):
        parts = [d.get("content", "") if isinstance(d, dict) else str(d) for d in details]
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            return joined

    # 4. Raw dict fallback — works regardless of SDK version
    try:
        raw = message.model_dump() if hasattr(message, "model_dump") else vars(message)
        for key in ("content", "reasoning_content", "text"):
            val = raw.get(key)
            if val and isinstance(val, str) and val.strip():
                import re
                return re.sub(r"<think>[\s\S]*?</think>\s*", "", val, flags=re.IGNORECASE).strip() or val.strip()
    except Exception:
        pass

    return ""


# ── Provider metadata ─────────────────────────────────────────────────────────
_OPENAI_COMPAT_PROVIDERS = {
    # provider_name : (base_url, env_var_name)
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}


class Agent:
    def __init__(self, provider="google", model="gemini-2.5-flash", api_key=None,
                 max_retries=3, retry_delay=2, reasoning_enabled=False):
        # NOTE: reasoning_enabled defaults to False so JSON-output tasks
        # always get content in message.content rather than reasoning fields.
        # Set reasoning_enabled=True explicitly only when you want CoT output.
        normalized = (provider or "google").strip().lower()
        # Legacy aliases
        if normalized in {"openapi", "openai"}:
            normalized = "openrouter"
        if normalized in {"grok", "xai", "x-ai", "x.ai"}:
            normalized = "groq"
        self.provider          = normalized
        self.model             = model
        self.max_retries       = max_retries
        self.retry_delay       = retry_delay
        self.reasoning_enabled = reasoning_enabled

        self.last_assistant_message  = None
        self.last_reasoning_details  = None

        # ── Resolve API key ───────────────────────────────────────────────────
        if self.provider in _OPENAI_COMPAT_PROVIDERS:
            _, env_var = _OPENAI_COMPAT_PROVIDERS[self.provider]
            key = api_key or os.getenv(env_var, "")
        else:
            key = api_key or os.getenv("GEMINI_API_KEY", "")

        self.api_key_configured = bool(key)
        self.client     = None
        self.google_sdk = None

        # ── Build client ──────────────────────────────────────────────────────
        if self.provider in _OPENAI_COMPAT_PROVIDERS:
            base_url, _ = _OPENAI_COMPAT_PROVIDERS[self.provider]
            if key and OpenAI:
                self.client = OpenAI(base_url=base_url, api_key=key)
            elif not key:
                logger.warning("%s client: no API key provided", self.provider)
            elif not OpenAI:
                logger.warning("%s client: openai SDK not installed", self.provider)
        else:
            if key and genai:
                self.client     = genai.Client(api_key=key)
                self.google_sdk = "genai"
            elif key and generativeai:
                generativeai.configure(api_key=key)
                self.client     = generativeai
                self.google_sdk = "generativeai"
            elif not key:
                logger.warning("Google client: no API key provided")
            else:
                logger.warning("Google client: google-genai / google-generativeai SDK not installed")

    # ── Google ────────────────────────────────────────────────────────────────
    def _ask_google(self, prompt, max_tokens, temperature):
        if not self.client:
            reason = ("missing GEMINI_API_KEY" if not self.api_key_configured
                      else "missing google-genai/google-generativeai SDK")
            logger.warning("Google LLM client unavailable: %s", reason)
            return ""

        if self.google_sdk == "genai":
            if not types:
                return ""
            response = self.client.models.generate_content(
                model=self.model, contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens, temperature=temperature
                ),
            )
            # response.text raises if the response was blocked; handle gracefully
            try:
                text = response.text or ""
            except Exception:
                # Fall back to candidates
                text = ""
                try:
                    for cand in (response.candidates or []):
                        for part in (cand.content.parts or []):
                            text += getattr(part, "text", "") or ""
                except Exception:
                    pass
            return text

        # generativeai (legacy SDK)
        model_obj = self.client.GenerativeModel(self.model)
        response  = model_obj.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        text = getattr(response, "text", "") or ""
        if not text:
            # Try candidates path
            try:
                for cand in (response.candidates or []):
                    for part in (cand.content.parts or []):
                        text += getattr(part, "text", "") or ""
            except Exception:
                pass
        return text

    # ── OpenRouter / Groq (OpenAI-compatible) ────────────────────────────────
    def _ask_openrouter(self, prompt, max_tokens, temperature):
        if not self.client:
            if self.provider == "groq":
                reason = ("missing GROQ_API_KEY" if not self.api_key_configured
                          else "missing openai SDK")
                logger.warning("Groq LLM client unavailable: %s", reason)
            else:
                reason = ("missing OPENROUTER_API_KEY" if not self.api_key_configured
                          else "missing openai SDK")
                logger.warning("OpenRouter LLM client unavailable: %s", reason)
            return ""

        # Build request — only include reasoning param when explicitly enabled
        # because some models (e.g. base Llama) reject the extra_body field.
        extra: dict = {}
        if self.reasoning_enabled:
            extra["reasoning"] = {"enabled": True}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra if extra else None,
        )

        if not response.choices:
            logger.warning("OpenRouter: empty choices array (model=%s)", self.model)
            return ""

        message = response.choices[0].message

        # Log for traceability
        logger.debug(
            "OpenRouter response: id=%s model=%s finish=%s",
            getattr(response, "id", "?"),
            getattr(response, "model", self.model),
            getattr(response.choices[0], "finish_reason", "?"),
        )

        content = _extract_openrouter_content(message)

        if not content:
            # Last-resort: dump the full message object for debugging
            try:
                raw = message.model_dump() if hasattr(message, "model_dump") else vars(message)
                logger.warning(
                    "OpenRouter: empty content after all extraction attempts. "
                    "finish_reason=%s message_keys=%s",
                    getattr(response.choices[0], "finish_reason", "?"),
                    list(raw.keys()),
                )
            except Exception:
                logger.warning("OpenRouter: empty content, could not inspect message")

        self.last_reasoning_details  = getattr(message, "reasoning_details", None)
        self.last_assistant_message  = {"role": "assistant", "content": content}
        return content

    # ── Multi-turn ────────────────────────────────────────────────────────────
    def ask_with_messages(self, messages, max_tokens=8192, temperature=0.7):
        if self.provider not in _OPENAI_COMPAT_PROVIDERS or not self.client:
            prompt  = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            content = self.ask(prompt, max_tokens=max_tokens, temperature=temperature)
            return {"role": "assistant", "content": content}

        extra: dict = {}
        if self.reasoning_enabled:
            extra["reasoning"] = {"enabled": True}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra if extra else None,
        )
        msg       = response.choices[0].message
        content   = _extract_openrouter_content(msg)
        assistant = {"role": "assistant", "content": content}
        rd = getattr(msg, "reasoning_details", None)
        if rd is not None:
            assistant["reasoning_details"] = rd
        self.last_assistant_message = assistant
        self.last_reasoning_details = rd
        return assistant

    def generate(self, prompt, max_tokens=8192, temperature=0.7):
        return self.ask(prompt, max_tokens=max_tokens, temperature=temperature)

    # ── Core ask with retry ───────────────────────────────────────────────────
    def ask(self, prompt, max_tokens=8192, temperature=0.7):
        """Retry with exponential backoff + jitter."""
        import random
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider in _OPENAI_COMPAT_PROVIDERS:
                    result = self._ask_openrouter(prompt, max_tokens, temperature)
                else:
                    result = self._ask_google(prompt, max_tokens, temperature)

                if result:
                    logger.debug(
                        "LLM success [%s] attempt=%d len=%d",
                        self.provider, attempt, len(result),
                    )
                    return result

                # Got a 200 but empty content — log and retry once
                logger.warning(
                    "LLM attempt %d/%d [%s] returned empty content (200 OK) — retrying",
                    attempt, self.max_retries, self.provider,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

            except Exception as exc:
                is_rate_limit = any(
                    k in str(exc).lower()
                    for k in ("429", "rate", "quota", "limit")
                )
                logger.warning(
                    "LLM attempt %d/%d [%s] %s: %s",
                    attempt, self.max_retries, self.provider,
                    "rate-limit" if is_rate_limit else "error", exc,
                )
                if attempt < self.max_retries:
                    sleep_t = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    if is_rate_limit:
                        sleep_t = max(sleep_t, 15.0)
                    time.sleep(min(sleep_t, 60))
                else:
                    logger.error(
                        "LLM failed after %d attempts [%s]: %s",
                        self.max_retries, self.provider, exc,
                    )

        return ""


class GeminiClient(Agent):
    def __init__(self):
        super().__init__(model="gemini-2.5-flash")