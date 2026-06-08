"""media/static_post.py """
from __future__ import annotations

import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("StaticPost")


# ── SDK detection ─────────────────────────────────────────────────────────────
def _detect_sdk():
    """
    Returns ('genai', module) for the new google-genai SDK,
            ('generativeai', module) for the legacy google-generativeai SDK,
            (None, None) if neither is available.
    """
    try:
        import google.genai as _genai  
        return "genai", _genai
    except ImportError:
        pass
    try:
        import google.generativeai as _genai  # legacy SDK
        return "generativeai", _genai
    except ImportError:
        pass
    return None, None


_SDK_TYPE, _SDK_MOD = _detect_sdk()
logger.info("StaticPost using SDK: %s", _SDK_TYPE or "NONE")


def _humanize_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        return (
            "⚠ Image generation quota reached. Your Gemini API free tier has been used up. "
            "Upgrade your Gemini plan at https://ai.dev/rate-limit or wait for quota reset."
        )
    if "401" in msg or "403" in msg or "invalid_api_key" in low or "api key" in low:
        return (
            "🔑 Invalid or missing Gemini API key. "
            "Go to Account → API Keys and paste a valid key from https://aistudio.google.com/app/apikey"
        )
    if "404" in msg or "not found" in low or "deprecated" in low:
        return (
            "🚫 The selected image model is no longer available. "
            "Go to Account → Settings and pick a different image model."
        )
    if "timeout" in low or "connection" in low or "network" in low:
        return "🌐 Network error while connecting to Gemini. Check your connection and try again."
    if "response_modalities" in msg or "unexpected keyword" in low:
        return (
            "⚙ SDK version mismatch: your google-generativeai package is too old to support "
            "image generation via response_modalities. "
            "Run: pip install --upgrade google-genai google-generativeai"
        )
    clean = msg.split("\n")[0][:120]
    return f"Image generation failed: {clean}"


@dataclass
class PostResult:
    idea_index:  int
    status:      str   # "completed" | "partial" | "failed"
    image_path:  Optional[str] = None
    image_url:   Optional[str] = None
    error:       Optional[str] = None


class StaticPostGenerator:
    """Generate static social posts via Gemini image API."""

    MAX_CONCURRENT = 5
    DEFAULT_MODEL  = "gemini-2.5-flash-image-preview"

    def __init__(self, api_key: str, output_dir: str,
                 model: str = DEFAULT_MODEL, brand_colors=None):
        self.api_key      = api_key
        self.output_dir   = Path(output_dir)
        self.model        = model or self.DEFAULT_MODEL
        self.brand_colors = brand_colors or ["#4f8ef7"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Prompt builder ────────────────────────────────────────────────────────
    def _build_prompt(self, idea: dict, brand_color: str, language: str) -> str:
        hook     = idea.get("hook", "")
        copy_    = idea.get("post_copy", "") or idea.get("caption", "")
        img_desc = idea.get("image_description", "")
        visual   = idea.get("visual_style", "") or idea.get("visual_direction", "")
        hashtags = " ".join(
            f"#{h.strip('#')}" for h in (idea.get("hashtags") or [])[:5]
        )
        parts = [
            "Create a high-quality, professional social media static post image.",
            f"Brand color: {brand_color}.",
            f"Language: {language}.",
        ]
        if img_desc:  parts.append(f"Image description: {img_desc}.")
        if visual:    parts.append(f"Visual style: {visual}.")
        if hook:      parts.append(f"Main headline / hook text (if included in image): {hook}.")
        if copy_:     parts.append(f"Supporting copy (if space): {copy_[:200]}.")
        if hashtags:  parts.append(f"Hashtags (small text at bottom if shown): {hashtags}.")
        parts += [
            "Aspect ratio: 4:5 (portrait, Instagram optimal).",
            "Style: modern, bold, eye-catching, scroll-stopping.",
            "No stock photo clichés. High production value.",
        ]
        return " ".join(parts)

    # ── New SDK (google-genai) ────────────────────────────────────────────────
    def _generate_image_new_sdk(self, prompt: str, filename: str,
                                 genai_mod) -> tuple[Optional[str], Optional[str]]:
        """Uses google-genai (new SDK). Returns (path, error)."""
        try:
            from google.genai import types as _types
            client   = genai_mod.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.parts:
                if (hasattr(part, "inline_data") and part.inline_data
                        and part.inline_data.mime_type.startswith("image/")):
                    ext      = part.inline_data.mime_type.split("/")[-1].replace("jpeg", "jpg")
                    img_path = self.output_dir / f"{filename}.{ext}"
                    img_path.write_bytes(base64.b64decode(part.inline_data.data))
                    logger.info("Image saved (new SDK): %s", img_path)
                    return str(img_path), None
            return None, "Gemini returned no image in its response (new SDK). Try regenerating."
        except Exception as e:
            return None, _humanize_error(e)

    # ── Legacy SDK (google-generativeai) ─────────────────────────────────────
    def _generate_image_legacy_sdk(self, prompt: str, filename: str,
                                    genai_mod) -> tuple[Optional[str], Optional[str]]:
        """
        Uses google.generativeai (legacy SDK).

        The legacy SDK's GenerationConfig does NOT support response_modalities.
        Instead we rely on the model itself understanding it should produce an image,
        and we parse inline_data from the candidate parts.

        For imagen-* models the API differs — we try generate_images() if available.
        """
        try:
            model_id = self.model

            # ── Imagen models use a different API call ────────────────────────
            if "imagen" in model_id.lower():
                return self._generate_imagen_legacy(prompt, filename, genai_mod)

            # ── Gemini image-generation models ────────────────────────────────
            genai_mod.configure(api_key=self.api_key)
            model = genai_mod.GenerativeModel(model_id)

            # Do NOT pass response_modalities — it's unsupported in old SDK.
            # The image models (gemini-*-image-*) return image parts natively.
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 1.0,
                    # Some older SDK versions need candidate_count explicitly
                    "candidate_count": 1,
                },
            )

            # Parse response — try both .parts and .candidates paths
            parts = []
            try:
                parts = response.parts or []
            except Exception:
                pass
            if not parts:
                try:
                    for cand in (response.candidates or []):
                        parts.extend(cand.content.parts or [])
                except Exception:
                    pass

            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and hasattr(inline, "mime_type"):
                    mime = inline.mime_type or ""
                    if mime.startswith("image/"):
                        ext      = mime.split("/")[-1].replace("jpeg", "jpg")
                        img_path = self.output_dir / f"{filename}.{ext}"
                        data     = inline.data
                        if isinstance(data, str):
                            data = base64.b64decode(data)
                        img_path.write_bytes(data)
                        logger.info("Image saved (legacy SDK): %s", img_path)
                        return str(img_path), None

            return None, (
                "Gemini returned no image in its response (legacy SDK). "
                "This usually means the model name is wrong or your API key lacks image permissions. "
                "Try upgrading: pip install --upgrade google-genai"
            )
        except Exception as e:
            return None, _humanize_error(e)

    def _generate_imagen_legacy(self, prompt: str, filename: str,
                                 genai_mod) -> tuple[Optional[str], Optional[str]]:
        """Handle Imagen models with the legacy SDK."""
        try:
            genai_mod.configure(api_key=self.api_key)
            # Try the ImageGenerationModel API if available in this SDK version
            if hasattr(genai_mod, "ImageGenerationModel"):
                model    = genai_mod.ImageGenerationModel(self.model)
                response = model.generate_images(prompt=prompt, number_of_images=1)
                img      = response.images[0]
                data     = img._image_bytes if hasattr(img, "_image_bytes") else img.image_bytes
                img_path = self.output_dir / f"{filename}.png"
                img_path.write_bytes(data)
                logger.info("Imagen image saved: %s", img_path)
                return str(img_path), None
            # Fall back to generate_content for Imagen via GenerativeModel
            model    = genai_mod.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            parts    = getattr(response, "parts", []) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and (getattr(inline, "mime_type", "") or "").startswith("image/"):
                    ext      = inline.mime_type.split("/")[-1].replace("jpeg", "jpg")
                    img_path = self.output_dir / f"{filename}.{ext}"
                    img_path.write_bytes(base64.b64decode(inline.data))
                    return str(img_path), None
            return None, "Imagen returned no image bytes."
        except Exception as e:
            return None, _humanize_error(e)

    # ── Unified generate (tries new SDK first, falls back to legacy) ──────────
    def _generate_image(self, prompt: str,
                         filename: str) -> tuple[Optional[str], Optional[str]]:
        """
        Try the available SDK. If the new SDK is installed, use it (it supports
        response_modalities). If only the legacy SDK is available, use the
        legacy path that avoids response_modalities entirely.
        """
        sdk_type, sdk_mod = _detect_sdk()  # re-detect so hot-reloads work

        if sdk_type == "genai":
            path, err = self._generate_image_new_sdk(prompt, filename, sdk_mod)
            if path:
                return path, None
            # If new SDK gave a response_modalities-like error, don't retry
            # the same SDK — log and return the error
            logger.warning("New SDK image gen failed: %s", err)
            return None, err

        if sdk_type == "generativeai":
            return self._generate_image_legacy_sdk(prompt, filename, sdk_mod)

        return None, (
            "No Google Generative AI SDK installed. "
            "Run: pip install google-genai"
        )

    # ── Per-idea processing ───────────────────────────────────────────────────
    def _process_idea(self, idea: dict, idea_idx: int,
                       brand_color: str, language: str) -> PostResult:
        filename = f"idea_{idea_idx + 1}"
        prompt   = self._build_prompt(idea, brand_color, language)

        last_err = None
        for attempt in range(2):
            path, err = self._generate_image(prompt, filename)
            if path:
                return PostResult(idea_index=idea_idx, status="completed",
                                  image_path=path)
            last_err = err
            # Don't retry quota / auth errors — they won't resolve
            if err and any(kw in err.lower()
                           for kw in ("quota", "rate", "🔑", "invalid", "api key")):
                logger.warning("Non-retryable error for idea %d: %s",
                               idea_idx + 1, err)
                break
            if attempt == 0:
                logger.warning("Image attempt 1 failed for idea %d: %s — retrying",
                               idea_idx + 1, err)
                time.sleep(2)

        return PostResult(idea_index=idea_idx, status="partial", error=last_err)

    # ── Public entry point ────────────────────────────────────────────────────
    def generate_all(self, content_json: dict, brand_colors: list = None,
                     language: str = "English") -> list[PostResult]:
        ideas       = content_json.get("ideas", [])
        bc          = brand_colors or self.brand_colors or ["#4f8ef7"]
        brand_color = bc[0] or "#4f8ef7"
        results: list[PostResult] = []

        with ThreadPoolExecutor(
            max_workers=min(self.MAX_CONCURRENT, len(ideas) or 1)
        ) as pool:
            futures = {
                pool.submit(self._process_idea, idea, i, brand_color, language): i
                for i, idea in enumerate(ideas)
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    idx = futures[fut]
                    err = _humanize_error(e)
                    logger.error("Idea %d future raised: %s", idx + 1, err)
                    results.append(PostResult(idea_index=idx,
                                              status="failed", error=err))

        results.sort(key=lambda r: r.idea_index)
        completed = [r for r in results if r.status == "completed"]
        failed    = [r for r in results if r.status != "completed"]
        logger.info("Image generation: %d completed, %d failed/partial",
                    len(completed), len(failed))
        for r in failed:
            logger.warning("Idea %d: %s — %s",
                           r.idea_index + 1, r.status, r.error)
        return results
