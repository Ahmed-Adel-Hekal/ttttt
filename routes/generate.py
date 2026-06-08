"""routes/generate.py — Generate page, result page, history. (v7 — video_provider)

Changes vs v6:
  - video_provider read from user settings and stored in generation config
  - Provider info banner on the form shows video provider too
  - cfg dict includes video_provider key
"""
from __future__ import annotations
from fastapi import APIRouter, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from auth import get_current_user, escape_html, escape_js
from db import (create_generation, get_generation, get_user_generations,
                quota_ok_atomic, release_quota_reservation,
                get_user_settings, get_default_brand,
                PLATFORM_CHOICES, LANGUAGE_CHOICES, detect_niche, get_brand_profile)
from core.i18n import normalize_lang, t as _t
import ui
import pipelines

router = APIRouter()

STATUS_BADGE = {
    "completed":         "badge-green",
    "running":           "badge-amber",
    "failed":            "badge-red",
    "pending":           "badge-gray",
    "scheduled":         "badge-blue",
    "awaiting_approval": "badge-amber",
    "cancelled":         "badge-gray",
    "generating_media":  "badge-amber",
}

PROVIDER_LABELS = {
    "google":     "Google Gemini",
    "openrouter": "OpenRouter",
    "groq":       "Groq",
}

VIDEO_PROVIDER_LABELS = {
    "aimlapi": "AIML API (Veo 3.1)",
    "gemini":  "Gemini Veo",
}


def _get_lang_and_settings(user):
    s    = get_user_settings(user["id"])
    lang = normalize_lang(s.get("ui_language", "en"))
    return lang, s


def _platform_checkboxes(prefill_plat=""):
    parts = []
    for p in PLATFORM_CHOICES:
        if prefill_plat:
            chk = "checked" if p == prefill_plat else ""
        else:
            chk = "checked" if p in ("Instagram", "TikTok") else ""
        parts.append(
            '<label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:13px;">'
            '<input type="checkbox" name="platforms" value="' + escape_html(p) + '" ' + chk +
            ' style="accent-color:var(--accent);width:15px;height:15px;"/>' +
            escape_html(p) + "</label>"
        )
    return "".join(parts)


def _lang_options(selected="English"):
    return "".join(
        f'<option value="{escape_html(lc)}" {"selected" if lc==selected else ""}>'
        f'{escape_html(lc)}</option>'
        for lc in LANGUAGE_CHOICES
    )


def _fmt_error(raw: str) -> str:
    if not raw:
        return "An unknown error occurred. Please try again."
    low = raw.lower()
    if "429" in raw or "quota" in low or "resource_exhausted" in low:
        return ("⚠ API quota reached. Your free-tier limit has been used up. "
                "Go to Account → API Keys and add a paid key, or wait for your quota to reset.")
    if "401" in raw or "403" in raw or "invalid_api_key" in low or "api key" in low:
        return "🔑 Invalid or missing API key. Go to Account → API Keys and paste a valid key."
    if "timeout" in low or "connection" in low:
        return "🌐 Network timeout. Check your connection and try again."
    if "json" in low or "parse" in low:
        return "🤖 The AI returned an unexpected response. Try regenerating."
    return raw.split("\n")[0][:300]


# ── Generate form ──────────────────────────────────────────────────────────────
@router.get("/generate", response_class=HTMLResponse)
async def generate_page(
    request: Request,
    msg: str = "",
    topic: str = "",
    platform: str = "",
    content_type: str = "",
    from_calendar: str = "",
    cal_id: str = "",
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    lang, settings = _get_lang_and_settings(user)
    brand          = get_default_brand(user["id"])
    bc  = brand["profile"].get("brand_color", "#4f8ef7") if brand else "#4f8ef7"
    bv  = brand["profile"].get("brand_voice", "")        if brand else ""

    prefill_topic = topic.strip()
    prefill_ct    = content_type if content_type in ("static", "video") else "static"
    prefill_plat  = platform.strip()

    saved_provider       = settings.get("llm_provider", "google")
    saved_model          = settings.get("llm_model", "gemini-2.5-flash")
    saved_video_provider = settings.get("video_provider", "aimlapi")
    saved_video_model    = settings.get("video_model", "google/veo-3.1-i2v")

    provider_label       = PROVIDER_LABELS.get(saved_provider, saved_provider.title())
    video_provider_label = VIDEO_PROVIDER_LABELS.get(saved_video_provider, saved_video_provider)

    provider_info = (
        f'<div class="alert alert-info" style="font-size:12px;padding:8px 12px;">'
        f'<div>📝 Text: <strong>{escape_html(provider_label)}</strong> · '
        f'<span style="font-family:var(--mono);">{escape_html(saved_model)}</span></div>'
        f'<div style="margin-top:3px;">🎬 Video: <strong>{escape_html(video_provider_label)}</strong> · '
        f'<span style="font-family:var(--mono);">{escape_html(saved_video_model)}</span></div>'
        f'<div style="margin-top:3px;">'
        f'<a href="/account" style="color:var(--blue);">Change in Account →</a>'
        f'</div></div>'
    )

    from_cal_banner = ""
    if from_calendar and prefill_topic:
        from_cal_banner = (
            '<div class="alert alert-info mb-3" style="font-size:13px;">'
            '📅 Pre-filled from your content calendar. Review and click Generate.'
            "</div>"
        )

    msg_html = ""
    if msg:
        msg_html = f'<div class="alert alert-success mb-3">{escape_html(msg)}</div>'

    content = (
        '<div class="topbar"><div>'
        f'<div class="topbar-title">{escape_html(_t(lang,"gen.title"))}</div>'
        "</div></div>"
        '<div class="content">'
        + from_cal_banner + msg_html +
        '<form method="post" action="/generate" id="gen-form">'
        '<div class="grid-2" style="gap:20px;align-items:start;">'

        # ── Left column ──────────────────────────────────────────────────────
        "<div>"
        '<div class="card mb-4">'
        '<div class="card-title">Content</div>'
        '<div class="form-group">'
        f'<label class="form-label">{escape_html(_t(lang,"gen.topic"))} *</label>'
        f'<input class="form-input" name="topic" value="{escape_html(prefill_topic)}" '
        'placeholder="e.g. AI product launch, fitness app..." required/>'
        "</div>"
        '<div class="form-group">'
        f'<label class="form-label">{escape_html(_t(lang,"gen.features"))}</label>'
        '<textarea class="form-textarea" name="product_features" style="min-height:70px;" '
        'placeholder="Key features (one per line)..."></textarea>'
        "</div>"
        '<div class="form-group">'
        f'<label class="form-label">{escape_html(_t(lang,"gen.competitor_urls"))}</label>'
        '<textarea class="form-textarea" name="competitor_urls" style="min-height:60px;" '
        'placeholder="https://instagram.com/competitor"></textarea>'
        '<div class="form-hint">Add competitor social URLs for intelligence analysis</div>'
        "</div>"
        "</div>"

        '<div class="card">'
        '<div class="card-title">Strategy</div>'
        '<div class="form-group">'
        '<label class="form-label">Static Post / Video</label>'
        '<div style="display:flex;gap:10px;">'
        f'<label style="flex:1;display:flex;align-items:center;justify-content:center;gap:8px;padding:10px;'
        f'border:1px solid {"var(--accent)" if prefill_ct!="video" else "var(--border)"};'
        f'border-radius:var(--r2);cursor:pointer;font-size:13px;" id="lbl-static">'
        f'<input type="radio" name="content_type" value="static" '
        f'{"checked" if prefill_ct!="video" else ""} style="accent-color:var(--accent);"/> Static Post'
        "</label>"
        f'<label style="flex:1;display:flex;align-items:center;justify-content:center;gap:8px;padding:10px;'
        f'border:1px solid {"var(--accent)" if prefill_ct=="video" else "var(--border)"};'
        f'border-radius:var(--r2);cursor:pointer;font-size:13px;" id="lbl-video">'
        f'<input type="radio" name="content_type" value="video" '
        f'{"checked" if prefill_ct=="video" else ""} style="accent-color:var(--accent);"/> Video'
        "</label>"
        "</div>"
        "</div>"
        '<div class="form-group">'
        f'<label class="form-label">{escape_html(_t(lang,"gen.ideas"))}</label>'
        '<select class="form-select" name="number_idea">'
        '<option value="1">1 idea</option>'
        '<option value="2">2 ideas</option>'
        '<option value="3" selected>3 ideas</option>'
        '<option value="5">5 ideas</option>'
        "</select>"
        "</div>"
        '<div class="form-group">'
        f'<label class="form-label">{escape_html(_t(lang,"gen.language"))}</label>'
        f'<select class="form-select" name="language">{_lang_options()}</select>'
        "</div>"
        + provider_info +
        "</div>"
        "</div>"  # end left col

        # ── Right column ─────────────────────────────────────────────────────
        "<div>"
        '<div class="card mb-4">'
        '<div class="card-title">Platforms</div>'
        '<div style="display:flex;flex-direction:column;gap:8px;">'
        + _platform_checkboxes(prefill_plat) +
        "</div></div>"

        '<div class="card mb-4">'
        '<div class="card-title">Brand</div>'
        '<div class="form-group">'
        '<label class="form-label">Brand Color</label>'
        '<div style="display:flex;gap:10px;align-items:center;">'
        f'<input type="color" name="brand_color" value="{escape_html(bc)}" '
        'style="width:48px;height:36px;border:none;background:none;cursor:pointer;border-radius:4px;"/>'
        '<span style="font-family:var(--mono);font-size:11px;color:var(--text3);">Primary brand color</span>'
        "</div></div>"
        '<div class="form-group">'
        f'<label class="form-label">{escape_html(_t(lang,"gen.brand_voice"))}</label>'
        '<textarea class="form-textarea" name="brand_voice" style="min-height:60px;" '
        'placeholder="Describe your brand voice...">'
        + escape_html(bv) + "</textarea>"
        "</div></div>"

        '<div class="card mb-4">'
        '<div class="card-title">Advanced</div>'
        '<div class="form-group">'
        '<label class="form-label">Aspect Ratio (Video)</label>'
        '<select class="form-select" name="aspect_ratio">'
        '<option value="9:16" selected>9:16 - Vertical (Reels/TikTok)</option>'
        '<option value="1:1">1:1 - Square</option>'
        '<option value="16:9">16:9 - Landscape</option>'
        "</select></div>"
        '<div class="form-group">'
        '<label class="form-label">Override API Key '
        '<span style="font-weight:400;color:var(--text3);">(optional)</span></label>'
        '<input class="form-input" type="password" name="llm_api_key" '
        'placeholder="Leave blank to use your saved key"/>'
        "</div>"
        '<label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:13px;">'
        '<input type="checkbox" name="human_review" value="1" checked '
        'style="accent-color:var(--accent);width:15px;height:15px;"/>'
        + escape_html(_t(lang, "gen.human_review")) +
        " - pause before media generation"
        "</label>"
        "</div>"

        '<button class="btn btn-primary" '
        'style="width:100%;justify-content:center;font-size:15px;padding:14px;" '
        f'type="submit" id="gen-btn">{escape_html(_t(lang,"gen.submit"))}</button>'
        "</div>"   # end right col
        "</div>"   # grid
        "</form>"
        "</div>"

        "<script>"
        "document.getElementById('gen-form').addEventListener('submit', function() {"
        "  var btn = document.getElementById('gen-btn');"
        f"  btn.textContent = '{escape_js(_t(lang,'gen.generating'))}';"
        "  btn.disabled = true;"
        "});"
        "document.querySelectorAll('[name=content_type]').forEach(function(r) {"
        "  r.addEventListener('change', function() {"
        "    document.getElementById('lbl-static').style.borderColor = r.value==='static' ? 'var(--accent)' : 'var(--border)';"
        "    document.getElementById('lbl-video').style.borderColor  = r.value==='video'  ? 'var(--accent)' : 'var(--border)';"
        "  });"
        "});"
        "</script>"
    )
    return HTMLResponse(ui._page(content, user, _t(lang, "gen.title"), "generate", lang))


# ── Generate submit ────────────────────────────────────────────────────────────
@router.post("/generate")
async def generate_post(
    request: Request,
    background_tasks: BackgroundTasks,
    topic: str = Form(""),
    content_type: str = Form("static"),
    language: str = Form("English"),
    brand_color: str = Form("#4f8ef7"),
    aspect_ratio: str = Form("9:16"),
    number_idea: int = Form(3),
    human_review: str = Form(""),
    llm_api_key: str = Form(""),
    brand_voice: str = Form(""),
    product_features: str = Form(""),
    competitor_urls: str = Form(""),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    topic = topic.strip()
    if not topic:
        return RedirectResponse("/generate?msg=Topic+required", status_code=303)

    if not quota_ok_atomic(user):
        return RedirectResponse(
            "/generate?msg=Quota+exceeded+—+upgrade+your+plan+at+%2Fpricing",
            status_code=303,
        )

    form      = await request.form()
    platforms = list(form.getlist("platforms")) or ["Instagram"]
    settings  = get_user_settings(user["id"])
    brand     = get_default_brand(user["id"])
    bp        = get_brand_profile(user["id"])

    merged_bp = dict(bp)
    if brand:
        merged_bp.update(brand.get("profile", {}))
    if brand_voice.strip():
        merged_bp["brand_voice"] = brand_voice.strip()

    features  = [f.strip() for f in product_features.splitlines() if f.strip()]
    comp_urls = [u.strip() for u in competitor_urls.splitlines() if u.strip().startswith("http")]

    import os
    provider       = settings.get("llm_provider", "google")
    video_provider = settings.get("video_provider", "aimlapi")

    # Resolve LLM key
    if provider == "openrouter":
        env_key   = os.getenv("OPENROUTER_API_KEY", "")
        saved_key = settings.get("openrouter_key", "")
    elif provider == "groq":
        env_key   = os.getenv("GROQ_API_KEY", "")
        saved_key = settings.get("groq_key", "")
    else:
        env_key   = os.getenv("GEMINI_API_KEY", "")
        saved_key = settings.get("gemini_key", "")
    resolved_llm_key = llm_api_key.strip() or saved_key or env_key

    # Resolve image key (always Gemini)
    resolved_img_key = settings.get("gemini_key", "") or os.getenv("GEMINI_API_KEY", "")

    # Resolve video key based on chosen video provider
    if video_provider in ("gemini", "google"):
        resolved_video_key = settings.get("gemini_key", "") or os.getenv("GEMINI_API_KEY", "")
    else:
        resolved_video_key = settings.get("aiml_key", "") or os.getenv("AIML_API_KEY", "")

    cfg = {
        "topic":            topic,
        "content_type":     content_type,
        "platforms":        platforms,
        "language":         language,
        "brand_color":      brand_color or "#4f8ef7",
        "aspect_ratio":     aspect_ratio,
        "number_idea":      max(1, min(int(number_idea), 10)),
        "niche":            detect_niche(topic),
        "human_review":     bool(human_review),
        "product_features": features,
        "competitor_urls":  comp_urls,
        "brand_profile":    merged_bp,
        "llm_provider":     provider,
        "llm_model":        settings.get("llm_model", "gemini-2.5-flash"),
        "image_model":      settings.get("image_model", "gemini-2.5-flash-image-preview"),
        "video_provider":   video_provider,                          # ← NEW
        "video_model":      settings.get("video_model", "google/veo-3.1-i2v"),
        "llm_api_key":      resolved_llm_key,
        "image_api_key":    resolved_img_key,
        "video_api_key":    resolved_video_key,
    }

    gid = create_generation(user["id"], topic, content_type, platforms, language, cfg)
    background_tasks.add_task(pipelines._run_pipeline, gid, user["id"], cfg)
    return RedirectResponse("/result/" + gid, status_code=303)


# ── Result page ────────────────────────────────────────────────────────────────
@router.get("/result/{gid}", response_class=HTMLResponse)
async def result_page(request: Request, gid: str):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    gen = get_generation(gid, user["id"])
    if not gen:
        return RedirectResponse("/history", status_code=303)

    lang, _ = _get_lang_and_settings(user)
    status  = gen["status"]

    if status in ("pending", "running", "generating_media"):
        spin_msgs = {
            "pending":          "Queued — starting pipeline…",
            "running":          "AI agents working — scraping trends, analysing competitors…",
            "generating_media": "Generating media — creating visuals…",
        }
        spin_msg = spin_msgs.get(status, "Processing…")
        content = (
            '<div class="topbar"><div>'
            '<div class="topbar-title">Generating…</div>'
            f'<div class="topbar-sub">{escape_html(gen["topic"][:60])}</div>'
            "</div>"
            f'<form method="post" action="/api/cancel-generation/{gid}" style="display:inline;">'
            '<button class="btn btn-ghost btn-sm" type="submit" '
            "onclick=\"return confirm('Cancel this generation?')\">✕ Cancel</button>"
            "</form>"
            "</div>"
            '<div class="content" style="display:flex;align-items:center;justify-content:center;min-height:60vh;">'
            '<div style="text-align:center;">'
            '<div class="spinner" style="width:48px;height:48px;border-width:3px;margin:0 auto 24px;"></div>'
            f'<div style="font-size:15px;font-weight:600;margin-bottom:8px;">{escape_html(spin_msg)}</div>'
            f'<div style="font-family:var(--mono);font-size:11px;color:var(--text3);">{gid[:16]}…</div>'
            '<div style="margin-top:24px;"><a class="btn btn-ghost btn-sm" href="/history">&larr; History</a></div>'
            "</div></div>"
            "<script>setTimeout(function(){location.replace(location.href);}, 3000);</script>"
        )
        return HTMLResponse(ui._page(content, user, "Generating…", "generate", lang))

    if status == "failed":
        raw_err  = gen.get("error") or ""
        friendly = _fmt_error(raw_err)
        raw_detail = ""
        if raw_err and raw_err != friendly:
            raw_detail = (
                '<details style="margin-top:12px;">'
                '<summary style="cursor:pointer;font-size:11px;color:var(--text3);">Technical detail</summary>'
                '<pre style="font-size:11px;font-family:var(--mono);color:var(--text3);'
                'white-space:pre-wrap;margin-top:8px;padding:10px;background:var(--surface2);'
                'border-radius:var(--r2);overflow-x:auto;">'
                + escape_html(raw_err[:600]) + "</pre></details>"
            )
        content = (
            '<div class="topbar"><div><div class="topbar-title">Generation Failed</div></div>'
            '<a class="btn btn-ghost" href="/generate">&larr; Try again</a></div>'
            '<div class="content">'
            f'<div class="alert alert-danger">{escape_html(friendly)}{raw_detail}</div>'
            '<div style="display:flex;gap:10px;margin-top:20px;">'
            '<a class="btn btn-primary" href="/generate">New Generation</a>'
            '<a class="btn btn-ghost btn-sm" href="/account">Check API Keys</a>'
            "</div></div>"
        )
        return HTMLResponse(ui._page(content, user, "Failed", "generate", lang))

    if status == "awaiting_approval":
        result     = gen.get("result") or {}
        ideas      = result.get("ideas", [])
        ideas_html = ui._build_ideas_html(gen)
        n          = len(ideas)
        gid_js     = escape_js(gid)
        content = (
            '<div class="topbar">'
            '<div><div class="topbar-title">Review Ideas</div>'
            f'<div class="topbar-sub">{escape_html(gen["topic"][:60])}</div></div>'
            '<div class="flex gap-2">'
            '<a class="btn btn-ghost" href="/generate">&larr; New</a>'
            f'<button class="btn btn-primary" onclick="approveAllIndividual(\'{gid_js}\',{n})">'
            f"Approve All ({n})"
            "</button>"
            "</div></div>"
            f'<div class="content">{ideas_html}</div>'
        )
        return HTMLResponse(ui._page(content, user, "Review Ideas", "generate", lang))

    result     = gen.get("result") or {}
    ideas_html = ui._build_ideas_html(gen)
    comp_html  = ui._build_competitor_report_html(result, gid)

    warnings = ""
    if result.get("warning"):
        warnings = (
            f'<div class="alert alert-warn mb-3" style="font-size:12px;">'
            f'{escape_html(result["warning"])}</div>'
        )
    if gen.get("fallback_used"):
        fallback_msg = _t(lang, "gen.fallback_warn")
        warnings = (
            f'<div class="alert alert-warn mb-3" style="font-size:12px;">'
            f'{escape_html(fallback_msg)}</div>' + warnings
        )

    results_list = result.get("results", [])
    if results_list:
        done    = sum(1 for r in results_list if isinstance(r,dict) and r.get("status")=="completed")
        partial = sum(1 for r in results_list if isinstance(r,dict) and r.get("status")=="partial")
        failed  = sum(1 for r in results_list if isinstance(r,dict) and r.get("status") not in ("completed","partial"))

        parts = []
        if done:    parts.append(f'<span class="badge badge-green">✓ {done} image{"s" if done!=1 else ""} generated</span>')
        if partial: parts.append(f'<span class="badge badge-amber">⚠ {partial} partial</span>')
        if failed:  parts.append(f'<span class="badge badge-red">✕ {failed} failed</span>')

        fail_details = ""
        for r in results_list:
            if isinstance(r,dict) and r.get("status") not in ("completed",) and r.get("error"):
                idx = r.get("idea_index","?")
                fail_details += (
                    f'<div style="font-size:11px;color:var(--red);margin-top:4px;">'
                    f'Idea {int(idx)+1 if str(idx).isdigit() else idx}: '
                    f'{escape_html(_fmt_error(r["error"]))}</div>'
                )

        if parts:
            media_banner = (
                '<div class="card mb-4" style="padding:12px 16px;">'
                '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
                '<span style="font-size:12px;font-weight:600;color:var(--text2);">Media:</span>'
                + " ".join(parts) + "</div>"
                + fail_details + "</div>"
            )
            warnings = media_banner + warnings

    plats   = ", ".join(gen.get("platforms") or [])
    content = (
        '<div class="topbar">'
        f'<div><div class="topbar-title">{escape_html(gen["topic"][:50])}</div>'
        f'<div class="topbar-sub">{escape_html(plats[:60])} · '
        f'{escape_html(gen["content_type"])} · {escape_html(gen["language"])}</div></div>'
        '<div class="flex gap-2">'
        '<a class="btn btn-ghost" href="/generate">&larr; New</a>'
        '<a class="btn btn-ghost btn-sm" href="/history">History</a>'
        "</div></div>"
        f'<div class="content">{warnings}{comp_html}{ideas_html}</div>'
    )
    return HTMLResponse(ui._page(content, user, gen["topic"][:40], "generate", lang))


# ── History ────────────────────────────────────────────────────────────────────
@router.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    lang, _ = _get_lang_and_settings(user)
    gens    = get_user_generations(user["id"], limit=200)

    rows = ""
    for g in gens:
        badge      = STATUS_BADGE.get(g["status"], "badge-gray")
        status_lbl = g["status"].replace("_", " ")
        ct_icon    = "Video" if g["content_type"] == "video" else "Static"
        plats      = ", ".join(g.get("platforms") or [])[:30]
        fallback_span = (
            ' <span class="badge badge-amber" style="font-size:9px;">fallback</span>'
            if g.get("fallback_used") else ""
        )
        rows += (
            "<tr>"
            '<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
            + escape_html(g["topic"]) + "</td>"
            "<td>" + escape_html(ct_icon) + "</td>"
            "<td>" + escape_html(plats) + "</td>"
            f'<td><span class="badge {badge}">{escape_html(status_lbl)}</span>{fallback_span}</td>'
            '<td style="font-family:var(--mono);font-size:10px;color:var(--text3);">'
            + g["created_at"][:16] + "</td>"
            f'<td><a class="btn btn-ghost btn-sm" href="/result/{g["id"]}">View</a></td>'
            "</tr>"
        )

    if not rows:
        rows = (
            '<tr><td colspan="6" style="text-align:center;color:var(--text3);padding:32px;">'
            + escape_html(_t(lang, "hist.no_gens")) + "</td></tr>"
        )

    content = (
        '<div class="topbar">'
        f'<div><div class="topbar-title">{escape_html(_t(lang,"hist.title"))} '
        f'<span style="font-family:var(--mono);font-size:13px;color:var(--text3);">({len(gens)})</span></div></div>'
        f'<a class="btn btn-primary" href="/generate">{escape_html(_t(lang,"action.generate"))}</a>'
        "</div>"
        '<div class="content"><div class="table-wrap">'
        "<table><thead><tr>"
        f"<th>{escape_html(_t(lang,'hist.topic'))}</th>"
        f"<th>{escape_html(_t(lang,'hist.type'))}</th>"
        f"<th>{escape_html(_t(lang,'gen.platforms'))}</th>"
        f"<th>{escape_html(_t(lang,'hist.status'))}</th>"
        f"<th>{escape_html(_t(lang,'hist.date'))}</th>"
        "<th></th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        "</div></div>"
    )
    return HTMLResponse(ui._page(content, user, _t(lang, "hist.title"), "history", lang))
