"""routes/strategy.py — AI content strategy generation. (v6 — richer prompt)

FIX: The old _run_strategy_pipeline built a bare-bones prompt that only
mentioned "topic" and "duration".  The LLM had no idea about:
  - Which platforms to target
  - Language / locale
  - Brand voice / industry
  - Content mix (static vs video)
  - Post frequency pattern
  - Desired output richness (hashtags, CTAs, visual direction)

Result: the model returned thin 1-liner plans with no hashtags, no angles,
and often just repeated "topic — Day N" as the title.

Changes:
  1. _build_strategy_prompt() — new function that injects brand context,
     platforms, language, content mix guidance, and explicit field requirements.
  2. Passes platforms + brand_profile through from the create endpoint.
  3. Fallback generator now uses the same field schema so the UI always
     gets well-formed objects.
  4. Agent uses ask() (correct method) — already fixed in v5, kept.
"""
from __future__ import annotations
import json
import datetime
import random
from fastapi import APIRouter, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from auth import get_current_user, escape_html, escape_js
from db import (create_strategy, update_strategy, get_strategy, get_user_strategies,
                add_calendar_items, get_brands, get_user_settings, quota_ok_atomic,
                detect_niche, LANGUAGE_CHOICES, PLATFORM_CHOICES)
from core.i18n import normalize_lang, t as _t
import ui
import pipelines

router = APIRouter()

CONTENT_PLATFORMS = PLATFORM_CHOICES  # ["Instagram","TikTok","LinkedIn","Twitter/X","Facebook"]


def _get_lang(user):
    s = get_user_settings(user["id"])
    return normalize_lang(s.get("ui_language", "en"))


# ── Strategy list ──────────────────────────────────────────────────────────────
@router.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request):
    user = get_current_user(request)
    if not user: return RedirectResponse("/login", status_code=303)

    lang       = _get_lang(user)
    strategies = get_user_strategies(user["id"])
    brands     = get_brands(user["id"])

    brand_opts = "".join(
        f'<option value="{escape_html(b["id"])}">{escape_html(b["name"])}</option>'
        for b in brands
    ) or '<option value="">— No brands yet —</option>'

    lang_opts = "".join(
        f'<option value="{l}">{l}</option>' for l in LANGUAGE_CHOICES
    )

    dur_opts = "".join(
        f'<option value="{d}">{d} days</option>'
        for d in [7, 14, 30, 60, 90]
    )

    plat_checkboxes = "".join(
        f'<label style="display:flex;align-items:center;gap:7px;cursor:pointer;font-size:13px;">'
        f'<input type="checkbox" name="platforms" value="{p}" '
        f'{"checked" if p in ("Instagram","TikTok") else ""} '
        f'style="accent-color:var(--accent);width:14px;height:14px;"/>{p}</label>'
        for p in CONTENT_PLATFORMS
    )

    sb = {"generating": "badge-amber", "draft": "badge-gray",
          "approved": "badge-green", "failed": "badge-red"}
    strat_rows = "".join(
        f'<tr>'
        f'<td style="font-weight:600;">{escape_html(s["title"][:60])}</td>'
        f'<td style="font-family:var(--mono);font-size:10px;">{s["duration_days"]}d</td>'
        f'<td><span class="badge {sb.get(s["status"], "badge-gray")}">{s["status"]}</span></td>'
        f'<td style="font-family:var(--mono);font-size:10px;color:var(--text3);">{s["created_at"][:10]}</td>'
        f'<td><a class="btn btn-ghost btn-sm" href="/strategy/{s["id"]}">View</a></td>'
        f'</tr>'
        for s in strategies
    ) or f'<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:24px;">No strategies yet</td></tr>'

    content = f"""
    <div class="topbar">
      <div><div class="topbar-title">{_t(lang,"nav.strategy")}</div></div>
      <button class="btn btn-primary" onclick="document.getElementById('new-strat-modal').style.display='flex'">+ New Strategy</button>
    </div>
    <div class="content">
      <div class="table-wrap">
        <table><thead><tr><th>Title</th><th>Duration</th><th>Status</th><th>Created</th><th></th></tr></thead>
        <tbody>{strat_rows}</tbody></table>
      </div>
    </div>

    <div id="new-strat-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:200;align-items:center;justify-content:center;">
      <div class="card" style="width:100%;max-width:540px;max-height:90vh;overflow-y:auto;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
          <div class="card-title" style="margin:0;">New Content Strategy</div>
          <button style="background:none;border:none;color:var(--text3);font-size:18px;cursor:pointer;" onclick="document.getElementById('new-strat-modal').style.display='none'">✕</button>
        </div>
        <form method="post" action="/strategy/create">
          <div class="form-group">
            <label class="form-label">Topic / Brand *</label>
            <input class="form-input" name="topic" placeholder="e.g. AI SaaS, fitness brand, restaurant chain" required/>
          </div>
          <div class="form-group">
            <label class="form-label">Industry / Niche</label>
            <input class="form-input" name="industry" placeholder="e.g. B2B SaaS, Health & Wellness, E-commerce"/>
          </div>
          <div class="form-group">
            <label class="form-label">Target Audience</label>
            <input class="form-input" name="target_audience" placeholder="e.g. startup founders, fitness enthusiasts aged 25-40"/>
          </div>
          <div class="form-group">
            <label class="form-label">Brand Voice</label>
            <input class="form-input" name="brand_voice" placeholder="e.g. bold and educational, warm and conversational"/>
          </div>
          <div class="form-group">
            <label class="form-label">Platforms</label>
            <div style="display:flex;flex-direction:column;gap:6px;">{plat_checkboxes}</div>
          </div>
          <div class="form-group">
            <label class="form-label">Duration</label>
            <select class="form-select" name="duration_days">{dur_opts}</select>
          </div>
          <div class="form-group">
            <label class="form-label">Language</label>
            <select class="form-select" name="language">{lang_opts}</select>
          </div>
          <div class="form-group">
            <label class="form-label">Brand <span style="color:var(--text3);font-weight:400;">(optional)</span></label>
            <select class="form-select" name="brand_id"><option value="">No brand</option>{brand_opts}</select>
          </div>
          <div style="display:flex;gap:10px;">
            <button class="btn btn-primary" type="submit">Generate Strategy →</button>
            <button class="btn btn-ghost" type="button" onclick="document.getElementById('new-strat-modal').style.display='none'">Cancel</button>
          </div>
        </form>
      </div>
    </div>"""
    return HTMLResponse(ui._page(content, user, "Strategy", "strategy", lang))


# ── Create strategy ────────────────────────────────────────────────────────────
@router.post("/strategy/create")
async def strategy_create(request: Request, background_tasks: BackgroundTasks,
                           topic: str = Form(""), duration_days: int = Form(30),
                           language: str = Form("English"), brand_id: str = Form(""),
                           industry: str = Form(""), target_audience: str = Form(""),
                           brand_voice: str = Form("")):
    user = get_current_user(request)
    if not user: return RedirectResponse("/login", status_code=303)

    topic = topic.strip()
    if not topic:
        return RedirectResponse("/strategy", status_code=303)

    form      = await request.form()
    platforms = list(form.getlist("platforms")) or ["Instagram", "TikTok"]

    import os as _os
    settings   = get_user_settings(user["id"])
    brand_data = {}
    if brand_id:
        from db import get_brand
        b = get_brand(brand_id, user["id"])
        if b:
            brand_data = b.get("profile", {})

    title    = f"{topic[:40]} — {duration_days}d Plan"
    sid      = create_strategy(user["id"], brand_id or None, title, topic, duration_days)
    provider = settings.get("llm_provider", "google")

    # Resolve the correct API key for the selected provider — same logic as generate.py.
    # IMPORTANT: do NOT fall back across providers (e.g. gemini_key when provider=openrouter)
    # because the wrong key will cause a 401 on the target provider's API.
    if provider == "openrouter":
        saved_key = settings.get("openrouter_key", "")
        env_key   = _os.getenv("OPENROUTER_API_KEY", "")
    else:
        saved_key = settings.get("gemini_key", "")
        env_key   = _os.getenv("GEMINI_API_KEY", "")

    resolved_key = saved_key or env_key

    cfg = {
        "topic":            topic,
        "duration_days":    duration_days,
        "language":         language,
        "platforms":        platforms,
        "industry":         industry.strip(),
        "target_audience":  target_audience.strip(),
        "brand_voice":      brand_voice.strip(),
        "brand_data":       brand_data,
        "brand_id":         brand_id,
        "llm_api_key":      resolved_key,
        "llm_provider":     provider,
        "llm_model":        settings.get("llm_model", "gemini-2.5-flash"),
    }
    background_tasks.add_task(_run_strategy_pipeline, sid, user["id"], cfg)
    return RedirectResponse(f"/strategy/{sid}", status_code=303)


# ── Prompt builder ─────────────────────────────────────────────────────────────
def _build_strategy_prompt(cfg: dict, duration: int) -> str:
    """
    Build a rich, context-aware strategy prompt.

    Injects: topic, industry, audience, brand voice, platforms, language,
    content-type mix guidance, and explicit per-day field requirements.
    The result is a JSON array with proper hooks, angles, hashtags, CTAs,
    and visual directions — not just a title.
    """
    topic          = cfg.get("topic", "")
    industry       = cfg.get("industry", "") or detect_niche(topic).replace("_", " ").title()
    audience       = cfg.get("target_audience", "") or "general social media audience"
    voice          = cfg.get("brand_voice", "") or "engaging and informative"
    language       = cfg.get("language", "English")
    platforms      = cfg.get("platforms", ["Instagram", "TikTok"])
    brand_data     = cfg.get("brand_data", {}) or {}
    brand_name     = brand_data.get("brand_name", "") or brand_data.get("name", "") or topic

    plat_str  = ", ".join(platforms) if platforms else "Instagram, TikTok"
    today_str = datetime.date.today().isoformat()

    # Content type rotation guidance
    # Static posts are cheaper/faster; videos go on high-engagement platforms
    video_platforms = [p for p in platforms if p in ("TikTok", "Instagram", "Facebook")]
    static_platforms = [p for p in platforms if p not in ("TikTok", "Instagram", "Facebook")]

    content_mix = (
        "Use a mix: roughly 60% static posts, 40% short video scripts. "
        f"Prefer 'video' content type for {', '.join(video_platforms) or 'TikTok/Instagram'} days "
        f"and 'static' for {', '.join(static_platforms) or 'LinkedIn/Twitter/X'} days."
        if video_platforms else
        "Use 'static' content type for all platforms since no video-first platforms are selected."
    )

    # Rotating content angles to ensure variety
    angles = [
        "problem → solution", "myth-busting", "behind the scenes",
        "before & after transformation", "customer pain point empathy",
        "social proof via unexpected story", "trending moment tie-in",
        "question-led curiosity hook", "unpopular opinion",
        "educational how-to", "day-in-the-life", "founder story",
        "comparison: old way vs new way", "future vision",
        "3-2-1 framework", "user-generated content style",
    ]

    brand_context = ""
    if brand_data:
        parts = []
        if brand_data.get("tagline"):       parts.append(f"Tagline: {brand_data['tagline']}")
        if brand_data.get("usp"):           parts.append(f"USP: {brand_data['usp']}")
        if brand_data.get("banned_words"):  parts.append(f"Never use: {brand_data['banned_words']}")
        if brand_data.get("tone"):          parts.append(f"Tone: {brand_data['tone']}")
        if parts:
            brand_context = "\nBrand context:\n" + "\n".join(f"  - {p}" for p in parts)

    arabic_note = ""
    if "arabic" in language.lower():
        arabic_note = (
            "\nCRITICAL: ALL text fields (title, hook, angle, hashtags, cta, visual_direction) "
            "MUST be written in natural Arabic — not translated, natively written. "
            "Hashtags should use Arabic text with # prefix."
        )

    prompt = f"""You are an expert social media strategist specializing in {industry}.

Create a {duration}-day content marketing strategy for: {brand_name}
Topic: {topic}
Industry: {industry}
Target audience: {audience}
Brand voice: {voice}
Platforms: {plat_str}
Language: {language}
Starting date: {today_str}
{brand_context}{arabic_note}

Content guidance: {content_mix}

Generate a JSON array of exactly {duration} objects (one per day).
Each object MUST include ALL of these fields:

{{
  "day": <integer 1 to {duration}>,
  "date": "<ISO date string, starting {today_str}>",
  "platform": "<one of: {plat_str}>",
  "content_type": "<'static' or 'video'>",
  "title": "<compelling post title, max 70 chars, specific and scroll-stopping>",
  "hook": "<opening line that stops the scroll — question, bold claim, or pattern interrupt, max 120 chars>",
  "angle": "<content angle from this list: {', '.join(angles[:8])}, or a creative variant>",
  "post_copy": "<2-4 sentence post body, conversational, on-brand, includes emojis where appropriate>",
  "visual_direction": "<specific visual description: scene, colors, mood, props — 1-2 sentences>",
  "cta": "<one clear call-to-action, e.g. 'Save this for later', 'Drop a 🔥 if you agree', 'Link in bio'>",
  "trend_tie_in": "<relevant trend, hashtag event, or cultural moment to piggyback on — or empty string>",
  "competitor_gap": "<one thing competitors are NOT doing that this post will do differently>",
  "hashtags": ["<5 to 8 relevant hashtags WITHOUT the # symbol>"]
}}

RULES:
1. Every day must have a DIFFERENT angle — do not repeat the same angle twice.
2. Rotate platforms across the {duration} days according to the platform list: {plat_str}.
3. Hooks must be specific to the topic — no generic "Did you know?" openers.
4. post_copy must sound human, not AI-generated.
5. Hashtags should mix broad reach tags with niche-specific ones.
6. Return ONLY the JSON array — no markdown, no preamble, no explanation.
"""
    return prompt


def _run_strategy_pipeline(sid: str, uid: str, cfg: dict):
    """Generate a full content strategy and seed calendar items."""
    import logging as _logging
    import os
    _log = _logging.getLogger("Strategy")
    try:
        update_strategy(sid, "generating")

        from core.gemini_client import Agent
        provider = cfg.get("llm_provider", "google")
        model    = cfg.get("llm_model", "gemini-2.5-flash")
        duration = cfg.get("duration_days", 30)

        # Resolve the env fallback from the CORRECT provider's variable.
        # cfg["llm_api_key"] is pre-resolved per-provider by strategy_create;
        # the env fallback only fires when the pipeline is called another way.
        if provider == "openrouter":
            env_fallback = os.getenv("OPENROUTER_API_KEY", "")
        else:
            env_fallback = os.getenv("GEMINI_API_KEY", "")

        api_key = cfg.get("llm_api_key", "") or env_fallback

        if not api_key:
            raise ValueError(
                f"No API key found for provider '{provider}'. "
                f"Go to Account \u2192 API Keys and add your "
                f"{'OpenRouter' if provider == 'openrouter' else 'Gemini'} key."
            )

        agent = Agent(provider=provider, model=model, api_key=api_key)
        prompt = _build_strategy_prompt(cfg, duration)

        # Use higher token limit — a 30-day plan with rich fields needs space
        raw = agent.ask(prompt, max_tokens=min(8000, duration * 280))

        if not raw:
            raise ValueError(
                f"LLM returned empty response. "
                f"Check your {provider} API key in Account → API Keys."
            )

        plan = _parse_strategy_json(raw, duration, cfg)
        update_strategy(sid, "approved", plan={"days": plan})

        # Seed calendar items
        base_date = datetime.date.today()
        cal_items = []
        for item in plan:
            day_offset = item.get("day", 1) - 1
            pub_date   = (base_date + datetime.timedelta(days=day_offset)).isoformat()
            cal_items.append({
                "strategy_id":  sid,
                "title":        item.get("title", item.get("hook", f"Day {day_offset+1}")),
                "platform":     item.get("platform", "Instagram"),
                "content_type": item.get("content_type", "static"),
                "publish_date": pub_date,
                "publish_time": "09:00",
                "status":       "scheduled",
                "idea":         item,
            })
        if cal_items:
            add_calendar_items(uid, cal_items)

        _log.info("Strategy %s done — %d days, %d calendar items", sid, len(plan), len(cal_items))

    except Exception as e:
        _log.error("Strategy %s failed: %s", sid, e)
        update_strategy(sid, "failed", plan={"error": str(e)})


def _parse_strategy_json(raw: str, duration: int, cfg: dict) -> list:
    """
    Parse the LLM output into a list of day-plan dicts.
    Falls back to a richly-populated synthetic plan if parsing fails.
    """
    import re

    # Try to extract a JSON array from the response
    m = re.search(r'\[[\s\S]*\]', raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list) and parsed:
                # Ensure every item has the required fields
                return [_normalise_day(item, i + 1, cfg) for i, item in enumerate(parsed)]
        except Exception:
            pass

    # Fallback: generate a synthetic but rich plan
    return _synthetic_plan(duration, cfg)


def _normalise_day(item: dict, day_num: int, cfg: dict) -> dict:
    """Fill in any missing fields so the UI always gets well-formed objects."""
    platforms = cfg.get("platforms", ["Instagram", "TikTok"])
    plat      = item.get("platform") or platforms[(day_num - 1) % len(platforms)]
    base_date = datetime.date.today()
    date_str  = (base_date + datetime.timedelta(days=day_num - 1)).isoformat()

    return {
        "day":              item.get("day", day_num),
        "date":             item.get("date", date_str),
        "platform":         plat,
        "content_type":     item.get("content_type", "static"),
        "title":            item.get("title", f"Day {day_num}: {cfg.get('topic','')} content"),
        "hook":             item.get("hook", ""),
        "angle":            item.get("angle", ""),
        "post_copy":        item.get("post_copy", ""),
        "visual_direction": item.get("visual_direction", ""),
        "cta":              item.get("cta", ""),
        "trend_tie_in":     item.get("trend_tie_in", ""),
        "competitor_gap":   item.get("competitor_gap", item.get("competitor_angle", "")),
        "hashtags":         item.get("hashtags", []),
    }


def _synthetic_plan(duration: int, cfg: dict) -> list:
    """
    Richer synthetic fallback — each day gets a different angle, platform,
    and basic hook so the calendar isn't a wall of identical entries.
    """
    topic     = cfg.get("topic", "content")
    platforms = cfg.get("platforms", ["Instagram", "TikTok"])
    language  = cfg.get("language", "English")
    base_date = datetime.date.today()

    angles = [
        "problem → solution", "myth-busting", "behind the scenes",
        "before & after", "social proof", "trending moment tie-in",
        "curiosity hook", "unpopular opinion", "educational how-to",
        "founder story", "comparison", "future vision",
        "3-2-1 framework", "user story", "day-in-the-life", "data-driven",
    ]
    hooks = [
        f"Nobody talks about this side of {topic}…",
        f"The {topic} mistake 90% of people make",
        f"Here's what {topic} actually looks like behind the scenes",
        f"Stop doing this with your {topic} strategy",
        f"This {topic} framework changed everything for us",
        f"Unpopular opinion: {topic} doesn't have to be hard",
        f"3 things I wish I knew about {topic} sooner",
        f"Why your {topic} isn't converting (and the fix)",
    ]
    ctaS = [
        "Save this for later 📌",
        "Drop a 🔥 if this helped",
        "Share with someone who needs this",
        "Comment your thoughts below 👇",
        "Follow for more tips like this",
        "Link in bio for the full guide",
        "Tag a friend who needs to see this",
        "Try this today and tell us how it goes",
    ]
    content_types = ["static", "video"]
    video_plats   = {"TikTok", "Instagram", "Facebook"}

    plan = []
    for i in range(duration):
        plat    = platforms[i % len(platforms)]
        ct      = "video" if plat in video_plats and i % 3 == 0 else "static"
        angle   = angles[i % len(angles)]
        hook    = hooks[i % len(hooks)]
        cta     = ctaS[i % len(ctaS)]
        date_s  = (base_date + datetime.timedelta(days=i)).isoformat()
        tags    = [topic.replace(" ", ""), "ContentStrategy", "SocialMedia",
                   "Marketing", "GrowthHacking"]

        plan.append({
            "day":              i + 1,
            "date":             date_s,
            "platform":         plat,
            "content_type":     ct,
            "title":            f"Day {i+1}: {topic} — {angle.split('→')[0].strip().title()}",
            "hook":             hook,
            "angle":            angle,
            "post_copy":        f"Here's your Day {i+1} content idea for {topic}. Use this as a starting point and customise it to your brand voice.",
            "visual_direction": f"Clean, branded visual with strong contrast. Feature {topic} prominently.",
            "cta":              cta,
            "trend_tie_in":     "",
            "competitor_gap":   "More specific, action-oriented content than competitors",
            "hashtags":         tags[:5],
        })
    return plan


# ── Strategy detail ────────────────────────────────────────────────────────────
@router.get("/strategy/{sid}", response_class=HTMLResponse)
async def strategy_detail(request: Request, sid: str):
    user = get_current_user(request)
    if not user: return RedirectResponse("/login", status_code=303)

    lang  = _get_lang(user)
    strat = get_strategy(sid, user["id"])
    if not strat: return RedirectResponse("/strategy", status_code=303)

    status = strat["status"]

    if status == "generating":
        content = f"""
        <div class="topbar"><div><div class="topbar-title">Generating Strategy…</div></div></div>
        <div class="content" style="display:flex;align-items:center;justify-content:center;min-height:60vh;">
          <div style="text-align:center;">
            <div class="spinner" style="width:48px;height:48px;border-width:3px;margin:0 auto 24px;"></div>
            <div style="font-size:15px;font-weight:600;">AI is building your {strat['duration_days']}-day strategy…</div>
            <div style="font-size:12px;color:var(--text3);margin-top:8px;">This usually takes 20–45 seconds for a detailed plan.</div>
          </div>
        </div>
        <script>setTimeout(()=>location.replace(location.href),3000);</script>"""
        return HTMLResponse(ui._page(content, user, "Strategy…", "strategy", lang))

    if status == "failed":
        plan       = strat.get("plan") or {}
        err_detail = plan.get("error", "")
        err_html   = (
            f'<div class="alert alert-danger" style="font-size:13px;margin-bottom:16px;">'
            f'<strong>Error:</strong> {escape_html(err_detail)}</div>'
            if err_detail else ""
        )
        content = f"""
        <div class="topbar"><div><div class="topbar-title">Strategy Failed</div></div>
          <a class="btn btn-ghost" href="/strategy">← Back</a></div>
        <div class="content">
          {err_html}
          <div class="alert alert-danger">Strategy generation failed. Check your API key in
          <a href="/account" style="color:var(--red);font-weight:600;">Account → API Keys</a>
          and try again.</div>
          <div style="margin-top:16px;">
            <a class="btn btn-primary" href="/strategy">← New Strategy</a>
          </div>
        </div>"""
        return HTMLResponse(ui._page(content, user, "Failed", "strategy", lang))

    plan = strat.get("plan", {}) or {}
    days = plan.get("days", [])

    plat_icons = {"Instagram": "📸", "TikTok": "🎬", "LinkedIn": "💼",
                  "Twitter/X": "🐦", "Facebook": "👥"}
    type_badge = {"static": "badge-blue", "video": "badge-purple"}

    day_rows = "".join(
        f'<tr>'
        f'<td style="font-family:var(--mono);font-size:11px;color:var(--text3);">Day {d.get("day", i+1)}</td>'
        f'<td style="font-family:var(--mono);font-size:10px;color:var(--text3);">{d.get("date", "")}</td>'
        f'<td>{plat_icons.get(d.get("platform", ""), "📱")} {escape_html(d.get("platform", ""))}</td>'
        f'<td><span class="badge {type_badge.get(d.get("content_type","static"),"badge-gray")}">{d.get("content_type","static")}</span></td>'
        f'<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{escape_html(d.get("title", "")[:80])}</td>'
        f'<td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text2);font-size:12px;">{escape_html(d.get("hook", "")[:80])}</td>'
        f'<td style="font-family:var(--mono);font-size:10px;color:var(--text3);">{escape_html(", ".join(("#" + h) for h in (d.get("hashtags") or [])[:3]))}</td>'
        f'</tr>'
        for i, d in enumerate(days[:100])
    ) or '<tr><td colspan="7" style="text-align:center;padding:24px;color:var(--text3);">No days generated</td></tr>'

    # Summary stats
    platforms_used = {}
    content_types  = {"static": 0, "video": 0}
    for d in days:
        p = d.get("platform", "Unknown")
        platforms_used[p] = platforms_used.get(p, 0) + 1
        ct = d.get("content_type", "static")
        content_types[ct] = content_types.get(ct, 0) + 1

    plat_pills = "".join(
        f'<span class="badge badge-blue">{plat_icons.get(p,"📱")} {p}: {cnt}</span>'
        for p, cnt in sorted(platforms_used.items(), key=lambda x: -x[1])
    )
    mix_pills = (
        f'<span class="badge badge-blue">📸 Static: {content_types["static"]}</span>'
        f'<span class="badge badge-purple">🎬 Video: {content_types["video"]}</span>'
    )

    content = f"""
    <div class="topbar">
      <div>
        <div class="topbar-title">{escape_html(strat["title"])}</div>
        <div class="topbar-sub">{len(days)} days · {escape_html(strat.get("topic", "")[:40])}</div>
      </div>
      <div class="flex gap-2">
        <a class="btn btn-ghost" href="/strategy">← Back</a>
        <a class="btn btn-ghost btn-sm" href="/calendar">📅 Calendar</a>
      </div>
    </div>
    <div class="content">
      <div class="card mb-4" style="padding:14px 18px;">
        <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
          <span style="font-size:12px;color:var(--text3);font-weight:600;">Platforms:</span>
          {plat_pills}
          <span style="font-size:12px;color:var(--text3);font-weight:600;margin-left:8px;">Mix:</span>
          {mix_pills}
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Day</th><th>Date</th><th>Platform</th><th>Type</th><th>Title</th><th>Hook</th><th>Hashtags</th></tr></thead>
          <tbody>{day_rows}</tbody>
        </table>
      </div>
    </div>"""
    return HTMLResponse(ui._page(content, user, strat["title"][:40], "strategy", lang))
