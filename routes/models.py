"""routes/models.py — Dynamic model catalogue endpoint. (v8 — Groq added)

GET /api/models?provider=google|openrouter|groq|aimlapi|gemini&type=llm|image|video

Changes vs v7:
  - provider=groq supported: fetches live models from api.groq.com/openai/v1/models using
    GROQ_API_KEY (or user's saved groq_key). Falls back to static Groq model list.
  - Static catalogue includes Llama 3.3 70B and other Groq models.
  - Cache key handles groq provider.
"""
from __future__ import annotations
import hashlib
import os
import time
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from auth import get_current_user
from db import get_user_settings

router = APIRouter()
logger = logging.getLogger("Models")

# ── In-process cache ───────────────────────────────────────────────────────────
_cache: dict[str, tuple[float, list]] = {}
_CACHE_TTL = 3600  # 1 hour


# ── Static fallback catalogues (LLM + image) ──────────────────────────────────
_STATIC: dict[str, dict[str, list]] = {
    "google": {
        "llm": [
            {"id": "gemini-2.5-flash",          "name": "Gemini 2.5 Flash",     "description": "Best speed/quality balance · recommended", "recommended": True},
            {"id": "gemini-2.5-pro",             "name": "Gemini 2.5 Pro",       "description": "Most capable Gemini model"},
            {"id": "gemini-2.0-flash",           "name": "Gemini 2.0 Flash",     "description": "Fast, efficient"},
            {"id": "gemini-2.0-flash-lite",      "name": "Gemini 2.0 Flash Lite","description": "Fastest · lowest cost"},
            {"id": "gemini-1.5-pro",             "name": "Gemini 1.5 Pro",       "description": "Long context (1M tokens)"},
            {"id": "gemini-1.5-flash",           "name": "Gemini 1.5 Flash",     "description": "Fast · 1M context"},
            {"id": "gemini-1.5-flash-8b",        "name": "Gemini 1.5 Flash 8B",  "description": "Smallest Gemini model"},
        ],
        "image": [
            {"id": "gemini-2.5-flash-image-preview",            "name": "Gemini 2.5 Flash Image",  "description": "Latest image generation · recommended", "recommended": True},
            {"id": "gemini-2.0-flash-preview-image-generation", "name": "Gemini 2.0 Flash Image",  "description": "Previous generation"},
            {"id": "imagen-4.0-generate-preview-05-20",         "name": "Imagen 4.0 Preview",      "description": "Latest Imagen (preview)"},
            {"id": "imagen-3.0-generate-002",                   "name": "Imagen 3.0",              "description": "High quality photorealistic"},
            {"id": "imagen-3.0-fast-generate-001",              "name": "Imagen 3.0 Fast",         "description": "Faster Imagen variant"},
        ],
    },
    "openrouter": {
        "llm": [
            {"id": "anthropic/claude-sonnet-4-5",       "name": "Claude Sonnet 4.5",     "description": "Smart, fast · recommended", "recommended": True},
            {"id": "anthropic/claude-opus-4",           "name": "Claude Opus 4",         "description": "Most capable Claude"},
            {"id": "anthropic/claude-haiku-4-5",        "name": "Claude Haiku 4.5",      "description": "Fastest Claude"},
            {"id": "openai/gpt-4o",                     "name": "GPT-4o",                "description": "OpenAI flagship"},
            {"id": "openai/gpt-4o-mini",                "name": "GPT-4o Mini",           "description": "Fast · affordable"},
            {"id": "openai/o3-mini",                    "name": "o3 Mini",               "description": "OpenAI reasoning model"},
            {"id": "google/gemini-2.5-flash",           "name": "Gemini 2.5 Flash (OR)", "description": "Via OpenRouter"},
            {"id": "google/gemini-2.5-pro",             "name": "Gemini 2.5 Pro (OR)",   "description": "Via OpenRouter"},
            {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B",         "description": "Meta open model"},
            {"id": "mistralai/mistral-large",           "name": "Mistral Large",         "description": "Mistral flagship"},
            {"id": "deepseek/deepseek-r1",              "name": "DeepSeek R1",           "description": "Strong reasoning"},
            {"id": "deepseek/deepseek-chat-v3-0324",    "name": "DeepSeek Chat v3",      "description": "Fast DeepSeek"},
            {"id": "qwen/qwen-2.5-72b-instruct",        "name": "Qwen 2.5 72B",          "description": "Alibaba open model"},
            {"id": "x-ai/grok-3-beta",                  "name": "Grok 3 Beta",           "description": "xAI model"},
        ],
        "image": [
            {"id": "black-forest-labs/flux-1.1-pro",       "name": "Flux 1.1 Pro",         "description": "Best quality · recommended", "recommended": True},
            {"id": "black-forest-labs/flux-1.1-pro-ultra", "name": "Flux 1.1 Pro Ultra",   "description": "Highest resolution"},
            {"id": "black-forest-labs/flux-schnell",       "name": "Flux Schnell",         "description": "Fastest Flux"},
            {"id": "black-forest-labs/flux-dev",           "name": "Flux Dev",             "description": "Open weights variant"},
            {"id": "openai/dall-e-3",                      "name": "DALL·E 3",             "description": "OpenAI image generation"},
            {"id": "stability-ai/stable-diffusion-3.5",    "name": "Stable Diffusion 3.5", "description": "Stability AI"},
            {"id": "ideogram-ai/ideogram-v2",              "name": "Ideogram v2",          "description": "Strong text-in-image"},
            {"id": "recraft-ai/recraft-v3",                "name": "Recraft v3",           "description": "Design-focused"},
        ],
    },
    "groq": {
        "llm": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "description": "Meta's latest 70B model · recommended", "recommended": True},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "description": "Fast and efficient"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "description": "Mistral open model"},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "description": "Google open model"}
        ],
        "image": [],
    },
}

_OR_IMAGE_ID_KEYWORDS = (
    "flux","dall-e","dall_e","stable-diffusion","stable_diffusion",
    "imagen","midjourney","ideogram","recraft","sdxl","sd-","/sd",
    "playgroundai","kandinsky","kolors","aura-flow","sana",
    "photoreal","dream-shaper","dreamshaper","wanvideo","wan-video",
    "hiDream","hidream","cogview","janus",
)
_OR_LLM_ONLY_KEYWORDS = (
    "llama","mistral","mixtral","claude","gpt","gemini","deepseek",
    "qwen","grok","phi","falcon","wizard","vicuna","alpaca",
    "hermes","solar","command","dbrx","yi-","aya",
    "o1-","o3-","nova-","aurora",
)


# ── Live fetchers ──────────────────────────────────────────────────────────────
def _fetch_google_models(api_key: str, model_type: str) -> list | None:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())
        if not all_models: return None

        results = []
        if model_type == "image":
            IMAGE_PREFIXES  = ("imagen", "gemini-3", "gemini-2.0-flash-preview-image")
            IMAGE_KEYWORDS  = ("image", "imagen", "vision", "img")
            for m in all_models:
                mid = m.name.replace("models/","").lower()
                is_image = (
                    any(mid.startswith(p) for p in IMAGE_PREFIXES) or
                    any(kw in mid for kw in IMAGE_KEYWORDS) or
                    "generateImages" in (getattr(m,"supported_generation_methods",[]) or [])
                )
                if not is_image: continue
                model_id = m.name.replace("models/","")
                rec = model_id in ("gemini-2.5-flash-image-preview","gemini-2.5-flash-image-preview-09-2025")
                results.append({"id":model_id, "name":getattr(m,"display_name",model_id) or model_id,
                                 "description":(getattr(m,"description","") or "")[:120], "recommended":rec})
        else:
            EXCLUDE = ("image","imagen","vision")
            for m in all_models:
                methods = getattr(m,"supported_generation_methods",[]) or []
                mid     = m.name.replace("models/","").lower()
                if "generateContent" not in methods and "streamGenerateContent" not in methods: continue
                if any(kw in mid for kw in EXCLUDE): continue
                model_id = m.name.replace("models/","")
                rec = model_id in ("gemini-2.5-flash","gemini-2.5-pro")
                results.append({"id":model_id, "name":getattr(m,"display_name",model_id) or model_id,
                                 "description":(getattr(m,"description","") or "")[:120], "recommended":rec})

        if not results: return None
        results.sort(key=lambda x:(0 if x.get("recommended") else 1, x["id"]))
        return results
    except Exception as e:
        logger.warning("Google model fetch failed: %s", e)
        return None


def _fetch_openrouter_models(api_key: str, model_type: str) -> list | None:
    try:
        import requests as _req
        resp = _req.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200: return None
        data = resp.json().get("data",[])
        if not data: return None

        def _is_image(m):
            mid  = (m.get("id") or "").lower()
            arch = str(m.get("architecture",{}) or {}).lower()
            if any(kw in mid for kw in _OR_IMAGE_ID_KEYWORDS): return True
            if "image" in arch and "output" in arch: return True
            ctx = m.get("context_length") or 0
            if ctx == 0 and not any(kw in mid for kw in _OR_LLM_ONLY_KEYWORDS): return True
            return False

        def _is_llm(m):
            mid = (m.get("id") or "").lower()
            ctx = m.get("context_length") or 0
            if ctx <= 0: return False
            if _is_image(m): return False
            return True

        _rec_llm = {"anthropic/claude-sonnet-4-5","anthropic/claude-opus-4","openai/gpt-4o","google/gemini-2.5-flash"}
        _rec_img = {"black-forest-labs/flux-1.1-pro"}

        def _entry(m, rec_set):
            mid     = m.get("id","")
            pricing = m.get("pricing",{}) or {}
            desc    = []
            p_prompt = pricing.get("prompt")
            if p_prompt is not None:
                try:
                    cost = float(p_prompt)*1_000_000
                    desc.append("Free" if cost==0 else f"${cost:.2f}/1M tok")
                except Exception: pass
            ctx = m.get("context_length")
            if ctx: desc.append(f"{ctx//1000}k ctx")
            return {"id":mid,"name":m.get("name",mid),
                    "description":" · ".join(desc),"recommended":mid in rec_set,
                    "context_length":ctx}

        if model_type == "image":
            results = [_entry(m,_rec_img) for m in data if _is_image(m)]
        else:
            results = [_entry(m,_rec_llm) for m in data if _is_llm(m)]

        if not results: return None
        results.sort(key=lambda x:(0 if x.get("recommended") else 1, x.get("id","")))
        return results
    except Exception as e:
        logger.warning("OpenRouter model fetch failed: %s", e)
        return None


def _fetch_groq_models(api_key: str, model_type: str) -> list | None:
    """Fetch live model list from Groq API."""
    try:
        import requests as _req
        resp = _req.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None

        _IMAGE_KEYWORDS = ("image", "vision", "img")
        _rec_llm = {"llama-3.3-70b-versatile"}

        results = []
        for m in data:
            mid      = (m.get("id") or "").lower()
            name     = m.get("id") or mid
            is_image = any(kw in mid for kw in _IMAGE_KEYWORDS)
            if model_type == "image" and not is_image:
                continue
            if model_type == "llm" and is_image:
                continue
            rec_set = set() if model_type == "image" else _rec_llm
            results.append({
                "id":          m.get("id", mid),
                "name":        name,
                "description": "",
                "recommended": m.get("id") in rec_set,
            })

        if not results:
            return None
        results.sort(key=lambda x: (0 if x.get("recommended") else 1, x["id"]))
        return results
    except Exception as e:
        logger.warning("Groq model fetch failed: %s", e)
        return None


# ── Cache key ──────────────────────────────────────────────────────────────────
def _cache_key(provider: str, mtype: str, api_key: str) -> str:
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12]
    return f"{provider}:{mtype}:{key_hash}"


# ── Main route ─────────────────────────────────────────────────────────────────
@router.get("/api/models")
async def get_models(request: Request, provider: str = "google", type: str = "llm", api_key: str = ""):
    """
    Returns model list for provider + type.

    provider: google | openrouter | groq | aimlapi | gemini
    type:     llm | image | video
    api_key:  optional key to override saved settings
    """
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    provider = provider.lower().strip()
    mtype    = type.lower().strip()

    # ── Video models — served statically from VideoGeneratorFactory ────────────
    if mtype == "video":
        try:
            from media.video_generator import VideoGeneratorFactory
            # Map provider aliases to what the factory expects
            vp = "gemini" if provider in ("gemini", "google") else "aimlapi"
            models = VideoGeneratorFactory.models_for_provider(vp)
            return JSONResponse({"models": models, "source": "static"})
        except Exception as exc:
            logger.warning("VideoGeneratorFactory not available: %s", exc)
            # Inline fallback so the endpoint never 500s
            if provider in ("gemini", "google"):
                fallback = [
                    {"id": "veo-3.1-fast-generate-preview",   "name": "Veo 3.1 Fast",       "description": "Latest · audio · 4K cap · recommended", "recommended": True},
                    {"id": "veo-3.1-generate-preview",        "name": "Veo 3.1",            "description": "High quality · 4K · audio",            "recommended": False},
                    {"id": "veo-2.0-generate-001",            "name": "Veo 2.0",            "description": "Stable · 5-8s · image-to-video",       "recommended": False},
                ]
            else:
                fallback = [
                    {"id": "google/veo-3.1-i2v",       "name": "Veo 3.1 (i2v)", "description": "Via AIML API · recommended", "recommended": True},
                    {"id": "google/veo-3.0-t2v-480p",  "name": "Veo 3.0 480p",  "description": "Via AIML API · text-to-video", "recommended": False},
                    {"id": "google/veo-2.0-i2v",       "name": "Veo 2.0 (i2v)", "description": "Via AIML API · stable",       "recommended": False},
                ]
            return JSONResponse({"models": fallback, "source": "static"})

    # ── LLM / image models ────────────────────────────────────────────────────
    # Normalise provider for LLM/image queries
    if provider not in ("google", "openrouter", "groq"):
        # aimlapi / gemini aliases → google for LLM purposes
        provider = "google"

    if mtype not in ("llm", "image"):
        return JSONResponse({"error": "type must be llm, image, or video"}, status_code=400)

    settings = get_user_settings(user["id"])
    
    # If the frontend passes a key explicitly, use it; otherwise fallback to DB / env.
    if api_key:
        api_key = api_key.strip()
    else:
        if provider == "google":
            api_key = settings.get("gemini_key", "") or os.getenv("GEMINI_API_KEY", "")
        elif provider == "groq":
            api_key = settings.get("groq_key", "") or os.getenv("GROQ_API_KEY", "")
        else:
            api_key = settings.get("openrouter_key", "") or os.getenv("OPENROUTER_API_KEY", "")

    ck  = _cache_key(provider, mtype, api_key)
    now = time.time()
    if ck in _cache:
        ts, cached_data = _cache[ck]
        if now - ts < _CACHE_TTL:
            return JSONResponse({"models": cached_data, "source": "cache"})

    live_data = None
    if api_key:
        if provider == "google":
            live_data = _fetch_google_models(api_key, mtype)
        elif provider == "groq":
            live_data = _fetch_groq_models(api_key, mtype)
        else:
            live_data = _fetch_openrouter_models(api_key, mtype)

    if live_data:
        _cache[ck] = (now, live_data)
        return JSONResponse({"models": live_data, "source": "live"})

    static = _STATIC.get(provider, {}).get(mtype, [])
    return JSONResponse({"models": static, "source": "static"})


@router.post("/api/models/clear-cache")
async def clear_model_cache(request: Request):
    user = get_current_user(request)
    if not user or not user.get("is_admin"):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    _cache.clear()
    return JSONResponse({"ok": True})
