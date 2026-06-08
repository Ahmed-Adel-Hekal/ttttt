"""routes/account.py — Account settings. (v7 — video provider toggle)

Changes vs v6:
  - Video section now shows a provider toggle: AIML API | Gemini
  - Model dropdown updates dynamically per provider via /api/models
  - video_provider saved to user_settings
"""
from __future__ import annotations
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from auth import get_current_user, escape_html, hash_password, verify_password
from db import (get_user_settings, save_user_settings, get_conn, quota_status,
                PLAN_QUOTAS, PLAN_PRICES)
from core.i18n import normalize_lang, t as _t, SUPPORTED_LANGUAGES
import ui

router = APIRouter()


def _get_lang(user):
    s = get_user_settings(user["id"])
    return normalize_lang(s.get("ui_language", "en"))


@router.get("/account", response_class=HTMLResponse)
async def account_page(request: Request, msg: str = "", error: str = ""):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    lang     = _get_lang(user)
    settings = get_user_settings(user["id"])
    q        = quota_status(user)
    pct      = round(q["used"] / max(q["limit"], 1) * 100)

    msg_html = f'<div class="alert alert-success mb-3">✓ {escape_html(msg)}</div>' if msg else ""
    err_html = f'<div class="alert alert-danger mb-3">✕ {escape_html(error)}</div>' if error else ""

    llm_provider   = settings.get("llm_provider", "google")
    llm_model      = escape_html(settings.get("llm_model",   "gemini-2.5-flash"))
    image_model    = escape_html(settings.get("image_model", "gemini-2.5-flash-image-preview"))
    video_provider = settings.get("video_provider", "aimlapi")
    video_model    = escape_html(settings.get("video_model", "google/veo-3.1-i2v"))

    def _prov_border(p): return "var(--accent)" if llm_provider == p else "var(--border)"

    lang_opts = "".join(
        f'<option value="{code}" {"selected" if code == settings.get("ui_language","en") else ""}>'
        f'{info["flag"]} {info["label"]}</option>'
        for code, info in SUPPORTED_LANGUAGES.items()
    )

    content = f"""
    <div class="topbar">
      <div><div class="topbar-title">{_t(lang,"acct.title")}</div></div>
    </div>
    <div class="content">
      {msg_html}{err_html}
      <div class="grid-2" style="gap:20px;align-items:start;">
        <div>
          <!-- Profile -->
          <div class="card mb-4">
            <div class="card-title">👤 {_t(lang,"acct.profile")}</div>
            <form method="post" action="/account/profile">
              <div class="form-group">
                <label class="form-label">{_t(lang,"acct.full_name")}</label>
                <input class="form-input" name="name" value="{escape_html(user['name'])}"/>
              </div>
              <div class="form-group">
                <label class="form-label">{_t(lang,"acct.email")}</label>
                <input class="form-input" type="email" name="email" value="{escape_html(user['email'])}"/>
              </div>
              <button class="btn btn-primary btn-sm" type="submit">{_t(lang,"acct.save_changes")}</button>
            </form>
          </div>

          <!-- Password -->
          <div class="card mb-4">
            <div class="card-title">🔑 {_t(lang,"acct.password")}</div>
            <form method="post" action="/account/password">
              <div class="form-group">
                <label class="form-label">Current Password</label>
                <input class="form-input" type="password" name="current_password" required/>
              </div>
              <div class="form-group">
                <label class="form-label">New Password</label>
                <input class="form-input" type="password" name="new_password" required minlength="8"/>
              </div>
              <button class="btn btn-ghost btn-sm" type="submit">Update Password</button>
            </form>
          </div>

          <!-- Language -->
          <div class="card">
            <div class="card-title">🌐 {_t(lang,"acct.ui_language")}</div>
            <form method="post" action="/account/settings">
              <div class="form-group">
                <label class="form-label">{_t(lang,"lang.label")}</label>
                <select class="form-select" name="ui_language">{lang_opts}</select>
              </div>
              <input type="hidden" name="_lang_only" value="1"/>
              <button class="btn btn-primary btn-sm" type="submit">{_t(lang,"action.save")}</button>
            </form>
          </div>
        </div>

        <div>
          <!-- Usage -->
          <div class="card mb-4">
            <div class="card-title">📊 {_t(lang,"acct.usage")}</div>
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;margin-bottom:6px;">
                <span>{q["used"]} {_t(lang,"quota.used")}</span>
                <span>{q["limit"]} limit · {q["plan"].upper()}</span>
              </div>
              <div class="progress">
                <div class="progress-bar" style="width:{pct}%;{"background:var(--red)" if pct>=90 else ""}"></div>
              </div>
            </div>
            <a class="btn btn-ghost btn-sm" href="/pricing">{_t(lang,"alert.upgrade")}</a>
          </div>

          <!-- API Keys + Models -->
          <div class="card">
            <div class="card-title">🔑 {_t(lang,"acct.api_keys")} &amp; Models</div>
            <form method="post" action="/account/settings" id="api-form">

              <!-- ── LLM provider toggle ─────────────────────────────── -->
              <div class="form-group">
                <label class="form-label">LLM Provider</label>
                <div style="display:flex;gap:8px;">
                  <label id="lbl-google"
                    style="flex:1;display:flex;align-items:center;justify-content:center;gap:8px;
                           padding:9px;border:1px solid {_prov_border('google')};
                           border-radius:var(--r2);cursor:pointer;font-size:13px;transition:border-color 0.15s;">
                    <input type="radio" name="llm_provider" value="google"
                      {"checked" if llm_provider=="google" else ""}
                      style="accent-color:var(--accent);" onchange="switchLlmProvider('google')"/>
                    🔵 Google Gemini
                  </label>
                  <label id="lbl-openrouter"
                    style="flex:1;display:flex;align-items:center;justify-content:center;gap:8px;
                           padding:9px;border:1px solid {_prov_border('openrouter')};
                           border-radius:var(--r2);cursor:pointer;font-size:13px;transition:border-color 0.15s;">
                    <input type="radio" name="llm_provider" value="openrouter"
                      {"checked" if llm_provider=="openrouter" else ""}
                      style="accent-color:var(--accent);" onchange="switchLlmProvider('openrouter')"/>
                    🟠 OpenRouter
                  </label>
                  <label id="lbl-groq"
                    style="flex:1;display:flex;align-items:center;justify-content:center;gap:8px;
                           padding:9px;border:1px solid {_prov_border('groq')};
                           border-radius:var(--r2);cursor:pointer;font-size:13px;transition:border-color 0.15s;">
                    <input type="radio" name="llm_provider" value="groq"
                      {"checked" if llm_provider=="groq" else ""}
                      style="accent-color:var(--accent);" onchange="switchLlmProvider('groq')"/>
                    ⚡ Groq
                  </label>
                </div>
              </div>

              <!-- Google section -->
              <div id="section-google" style="display:{"block" if llm_provider=="google" else "none"};">
                <div class="form-group">
                  <label class="form-label">Gemini API Key</label>
                  <input class="form-input" type="password" name="gemini_key"
                    value="{escape_html(settings.get('gemini_key',''))}"
                    placeholder="AIza… — aistudio.google.com"
                    oninput="scheduleLlmRefresh('google')"/>
                  <div class="form-hint">Also used for Gemini video generation (Veo).</div>
                </div>
                <div class="form-group">
                  <label class="form-label">Text Model
                    <span id="llm-status-google" style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-left:6px;"></span>
                  </label>
                  <select class="form-select" id="llm-sel-google" disabled>
                    <option value="{llm_model}">{llm_model} (loading…)</option>
                  </select>
                </div>
                <div class="form-group">
                  <label class="form-label">Image Model
                    <span id="img-status-google" style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-left:6px;"></span>
                  </label>
                  <select class="form-select" id="img-sel-google" disabled>
                    <option value="{image_model}">{image_model} (loading…)</option>
                  </select>
                </div>
              </div>

              <!-- OpenRouter section -->
              <div id="section-openrouter" style="display:{"block" if llm_provider=="openrouter" else "none"};">
                <div class="form-group">
                  <label class="form-label">OpenRouter API Key</label>
                  <input class="form-input" type="password" name="openrouter_key"
                    value="{escape_html(settings.get('openrouter_key',''))}"
                    placeholder="sk-or-… — openrouter.ai"
                    oninput="scheduleLlmRefresh('openrouter')"/>
                </div>
                <div class="form-group">
                  <label class="form-label">Text Model
                    <span id="llm-status-openrouter" style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-left:6px;"></span>
                  </label>
                  <select class="form-select" id="llm-sel-openrouter" disabled>
                    <option value="{llm_model}">{llm_model} (loading…)</option>
                  </select>
                </div>
                <div class="form-group">
                  <label class="form-label">Image Model
                    <span id="img-status-openrouter" style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-left:6px;"></span>
                  </label>
                  <select class="form-select" id="img-sel-openrouter" disabled>
                    <option value="{image_model}">{image_model} (loading…)</option>
                  </select>
                </div>
              </div>

              <!-- Groq section -->
              <div id="section-groq" style="display:{"block" if llm_provider=="groq" else "none"};">
                <div class="form-group">
                  <label class="form-label">Groq API Key</label>
                  <input class="form-input" type="password" name="groq_key"
                    value="{escape_html(settings.get('groq_key',''))}"
                    placeholder="gsk_… — console.groq.com"
                    oninput="scheduleLlmRefresh('groq')"/>
                  <div class="form-hint">Get your key at <a href="https://console.groq.com" target="_blank" style="color:var(--accent);">console.groq.com</a></div>
                </div>
                <div class="form-group">
                  <label class="form-label">Text Model
                    <span id="llm-status-groq" style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-left:6px;"></span>
                  </label>
                  <select class="form-select" id="llm-sel-groq" disabled>
                    <option value="{llm_model}">{llm_model} (loading…)</option>
                  </select>
                </div>
                <div class="form-group">
                  <label class="form-label">Image Model
                    <span id="img-status-groq" style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-left:6px;"></span>
                  </label>
                  <select class="form-select" id="img-sel-groq" disabled>
                    <option value="{image_model}">{image_model} (loading…)</option>
                  </select>
                </div>
              </div>

              <!-- ── Video provider section ──────────────────────────── -->
              <div style="border-top:1px solid var(--border);padding-top:16px;margin-top:4px;">
                <div class="form-group">
                  <label class="form-label">🎬 Video Provider</label>
                  <div style="display:flex;gap:8px;">
                    <label id="lbl-vp-aiml"
                      style="flex:1;display:flex;align-items:center;justify-content:center;gap:8px;
                             padding:9px;border:1px solid {"var(--accent)" if video_provider=="aimlapi" else "var(--border)"};
                             border-radius:var(--r2);cursor:pointer;font-size:13px;transition:border-color 0.15s;">
                      <input type="radio" name="video_provider" value="aimlapi"
                        {"checked" if video_provider=="aimlapi" else ""}
                        style="accent-color:var(--accent);" onchange="switchVideoProvider('aimlapi')"/>
                      ⚡ AIML API
                    </label>
                    <label id="lbl-vp-gemini"
                      style="flex:1;display:flex;align-items:center;justify-content:center;gap:8px;
                             padding:9px;border:1px solid {"var(--accent)" if video_provider=="gemini" else "var(--border)"};
                             border-radius:var(--r2);cursor:pointer;font-size:13px;transition:border-color 0.15s;">
                      <input type="radio" name="video_provider" value="gemini"
                        {"checked" if video_provider=="gemini" else ""}
                        style="accent-color:var(--accent);" onchange="switchVideoProvider('gemini')"/>
                      🔵 Gemini (Veo)
                    </label>
                  </div>
                  <div class="form-hint" id="vp-hint" style="margin-top:6px;">
                    {"Uses your AIML API key below. Supports Veo 3.1 image-to-video." if video_provider=="aimlapi"
                     else "Uses your Gemini API key above. Supports Veo 2.0 & Veo 3.0 Preview."}
                  </div>
                </div>

                <!-- AIML key (shown only for aimlapi provider) -->
                <div id="section-vp-aiml" style="display:{"block" if video_provider=="aimlapi" else "none"};">
                  <div class="form-group">
                    <label class="form-label">AIML API Key <span style="color:var(--text3);font-weight:400;">(Video — Veo 3.1)</span></label>
                    <input class="form-input" type="password" name="aiml_key"
                      value="{escape_html(settings.get('aiml_key',''))}"
                      placeholder="aimlapi.com"/>
                    <div class="form-hint">Get your key at <a href="https://aimlapi.com" target="_blank" style="color:var(--accent);">aimlapi.com</a></div>
                  </div>
                </div>

                <!-- Gemini note (shown only for gemini provider) -->
                <div id="section-vp-gemini" style="display:{"block" if video_provider=="gemini" else "none"};">
                  <div class="alert alert-info" style="font-size:12px;">
                    🔵 Gemini Veo uses the <strong>Gemini API key</strong> you entered above.
                    Make sure your project has the <strong>Video Generation API</strong> enabled.
                    <a href="https://ai.google.dev/api/generate-content#v1beta.models.generateVideo"
                       target="_blank" style="color:var(--blue);margin-left:4px;">Docs ↗</a>
                  </div>
                </div>

                <!-- Video model selector -->
                <div class="form-group">
                  <label class="form-label">Video Model
                    <span id="vid-model-status" style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-left:6px;"></span>
                  </label>
                  <select class="form-select" id="vid-model-sel">
                    <option value="{video_model}">{video_model}</option>
                  </select>
                </div>
              </div>

              <!-- Hidden fields carry chosen model IDs to POST handler -->
              <input type="hidden" name="llm_model"   id="h-llm-model"   value="{llm_model}"/>
              <input type="hidden" name="image_model" id="h-image-model" value="{image_model}"/>
              <input type="hidden" name="video_model" id="h-video-model" value="{video_model}"/>

              <button class="btn btn-primary btn-sm" type="submit"
                onclick="syncModels(event)">{_t(lang,"acct.save_changes")}</button>
            </form>
          </div>
        </div>
      </div>
    </div>

    <script>
    var _llmProvider   = '{llm_provider}';
    var _vidProvider   = '{video_provider}';
    var _savedLlm      = '{llm_model}';
    var _savedImg      = '{image_model}';
    var _savedVid      = '{video_model}';
    var _llmTimer      = null;
    var _ALL_PROVIDERS = ['google', 'openrouter', 'groq'];

    // ── LLM provider toggle ──────────────────────────────────────────────────
    function switchLlmProvider(val) {{
      _llmProvider = val;
      _ALL_PROVIDERS.forEach(function(p) {{
        var sec = document.getElementById('section-' + p);
        var lbl = document.getElementById('lbl-' + p);
        if (sec) sec.style.display = (p === val) ? 'block' : 'none';
        if (lbl) lbl.style.borderColor = (p === val) ? 'var(--accent)' : 'var(--border)';
      }});
      loadLlmModels(val);
    }}

    function scheduleLlmRefresh(prov) {{
      if (_llmTimer) clearTimeout(_llmTimer);
      _llmTimer = setTimeout(function(){{ loadLlmModels(prov); }}, 800);
    }}

    // ── Video provider toggle ────────────────────────────────────────────────
    var _VID_HINTS = {{
      aimlapi: 'Uses your AIML API key. Supports Veo 3.1 image-to-video.',
      gemini:  'Uses your Gemini API key above. Supports Veo 2.0 & Veo 3.0 Preview.',
    }};

    function switchVideoProvider(val) {{
      _vidProvider = val;
      document.getElementById('lbl-vp-aiml').style.borderColor   = val==='aimlapi' ? 'var(--accent)':'var(--border)';
      document.getElementById('lbl-vp-gemini').style.borderColor = val==='gemini'  ? 'var(--accent)':'var(--border)';
      document.getElementById('section-vp-aiml').style.display   = val==='aimlapi' ? 'block':'none';
      document.getElementById('section-vp-gemini').style.display = val==='gemini'  ? 'block':'none';
      document.getElementById('vp-hint').textContent = _VID_HINTS[val] || '';
      loadVideoModels(val);
    }}

    // ── Model loaders ────────────────────────────────────────────────────────
    function _fillSelect(sel, models, savedId) {{
      sel.innerHTML = '';
      models.forEach(function(m) {{
        var opt = document.createElement('option');
        opt.value = m.id;
        var label = m.recommended ? '⭐ ' + m.name : m.name;
        if (m.description) label += '  —  ' + m.description;
        opt.textContent = label;
        if (m.id === savedId) opt.selected = true;
        sel.appendChild(opt);
      }});
      if (!sel.value && models.length) sel.options[0].selected = true;
      sel.disabled = false;
    }}

    function _setStatus(id, text, color) {{
      var el = document.getElementById(id);
      if (!el) return;
      el.textContent = text;
      el.style.color = color || 'var(--text3)';
    }}

    async function loadLlmModels(prov) {{
      var llmSel = document.getElementById('llm-sel-' + prov);
      var imgSel = document.getElementById('img-sel-' + prov);
      if (!llmSel || !imgSel) return;

      var keyInputMap = {{
          'google': document.querySelector('input[name="gemini_key"]'),
          'openrouter': document.querySelector('input[name="openrouter_key"]'),
          'groq': document.querySelector('input[name="groq_key"]')
      }};
      var inputEl = keyInputMap[prov];
      var keyParam = (inputEl && inputEl.value) ? '&api_key=' + encodeURIComponent(inputEl.value) : '';

      _setStatus('llm-status-' + prov, 'loading…', 'var(--text3)');
      _setStatus('img-status-' + prov, 'loading…', 'var(--text3)');
      try {{
        var [lr, ir] = await Promise.all([
          fetch('/api/models?provider=' + prov + '&type=llm' + keyParam,   {{cache:'no-store'}}),
          fetch('/api/models?provider=' + prov + '&type=image' + keyParam, {{cache:'no-store'}}),
        ]);
        var ld = await lr.json(); var id_ = await ir.json();
        _fillSelect(llmSel, ld.models || [], _savedLlm);
        _fillSelect(imgSel, id_.models || [], _savedImg);
        var src = ld.source==='live' ? '✓ live' : '● cached';
        var col = ld.source==='live' ? 'var(--green)' : 'var(--text3)';
        _setStatus('llm-status-' + prov, src, col);
        _setStatus('img-status-' + prov, src, col);
      }} catch(e) {{
        _setStatus('llm-status-' + prov, '⚠ load failed', 'var(--red)');
        _setStatus('img-status-' + prov, '⚠ load failed', 'var(--red)');
      }}
    }}

    async function loadVideoModels(prov) {{
      _setStatus('vid-model-status', 'loading…', 'var(--text3)');
      try {{
        var r = await fetch('/api/models?provider=' + prov + '&type=video', {{cache:'no-store'}});
        var d = await r.json();
        _fillSelect(document.getElementById('vid-model-sel'), d.models || [], _savedVid);
        _setStatus('vid-model-status', d.source==='live' ? '✓ live' : '● static',
                   d.source==='live' ? 'var(--green)' : 'var(--text3)');
      }} catch(e) {{
        _setStatus('vid-model-status', '⚠ load failed', 'var(--red)');
      }}
    }}

    function syncModels(e) {{
      var llmSel = document.getElementById('llm-sel-' + _llmProvider);
      var imgSel = document.getElementById('img-sel-' + _llmProvider);
      var vidSel = document.getElementById('vid-model-sel');
      if (llmSel && llmSel.value) document.getElementById('h-llm-model').value   = llmSel.value;
      if (imgSel && imgSel.value) document.getElementById('h-image-model').value = imgSel.value;
      if (vidSel && vidSel.value) document.getElementById('h-video-model').value = vidSel.value;
    }}

    document.addEventListener('DOMContentLoaded', function() {{
      loadLlmModels(_llmProvider);
      loadVideoModels(_vidProvider);
    }});
    </script>
    """
    return HTMLResponse(ui._page(content, user, _t(lang, "acct.title"), "account", lang))


@router.post("/account/profile")
async def account_profile_save(request: Request,
                                name: str = Form(""), email: str = Form("")):
    user = get_current_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    name = name.strip(); email = email.strip().lower()
    if not name or not email:
        return RedirectResponse("/account?error=Name+and+email+required", status_code=303)

    # Don't allow collisions — if another user already owns this email, reject.
    if email != user["email"]:
        with get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email=? AND id<>?",
                (email, user["id"]),
            ).fetchone()
        if existing:
            return RedirectResponse(
                "/account?error=Email+already+in+use+by+another+account",
                status_code=303,
            )

    with get_conn() as conn:
        conn.execute("UPDATE users SET name=?, email=? WHERE id=?",
                     (name, email, user["id"]))
    return RedirectResponse("/account?msg=Profile+updated", status_code=303)


@router.post("/account/password")
async def account_password_save(request: Request,
                                 current_password: str = Form(""),
                                 new_password: str = Form("")):
    user = get_current_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    from auth import verify_password as vp
    if not vp(current_password, user["password_hash"]):
        return RedirectResponse("/account?error=Current+password+incorrect", status_code=303)
    if len(new_password) < 8:
        return RedirectResponse(
            "/account?error=New+password+must+be+at+least+8+chars", status_code=303
        )
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                     (hash_password(new_password), user["id"]))
    return RedirectResponse("/account?msg=Password+updated", status_code=303)


@router.post("/account/settings")
async def account_settings_save(request: Request):
    user = get_current_user(request)
    if not user: return RedirectResponse("/login", status_code=303)

    form     = await request.form()
    settings = get_user_settings(user["id"])

    if form.get("_lang_only"):
        settings["ui_language"] = normalize_lang(str(form.get("ui_language", "en")))
        save_user_settings(user["id"], settings)
        return RedirectResponse("/account?msg=Language+updated", status_code=303)

    settings.update({
        "gemini_key":      str(form.get("gemini_key",      "")).strip(),
        "openrouter_key":  str(form.get("openrouter_key",  "")).strip(),
        "groq_key":        str(form.get("groq_key",        "")).strip(),
        "aiml_key":        str(form.get("aiml_key",        "")).strip(),
        "llm_provider":    str(form.get("llm_provider",    "google")),
        "llm_model":       str(form.get("llm_model",       "gemini-2.5-flash")).strip(),
        "image_model":     str(form.get("image_model",     "gemini-2.5-flash-image-preview")).strip(),
        "video_provider":  str(form.get("video_provider",  "aimlapi")).strip(),
        "video_model":     str(form.get("video_model",     "google/veo-3.1-i2v")).strip(),
        "ui_language":     normalize_lang(
            str(form.get("ui_language", settings.get("ui_language", "en")))
        ),
    })
    save_user_settings(user["id"], settings)
    return RedirectResponse("/account?msg=Settings+saved", status_code=303)


@router.get("/pricing", response_class=HTMLResponse)
async def pricing_page(request: Request):
    user = get_current_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    lang = _get_lang(user)
    plans = [
        ("free",    "$0",   "10",    ["10 generations/month","Static posts only","1 platform"]),
        ("starter", "$19",  "50",    ["50 generations/month","Static + video","3 platforms","Competitor analysis"]),
        ("pro",     "$49",  "200",   ["200 generations/month","All content types","All platforms","Trend intelligence","Strategy calendar","Priority support"]),
        ("agency",  "$149", "1,000", ["1,000 generations/month","Everything in Pro","White-label reports","API access","Dedicated support"]),
    ]
    cards = ""
    for plan, price, quota, features in plans:
        is_current   = user.get("plan") == plan
        feature_list = "".join(
            f'<li style="padding:4px 0;font-size:13px;color:var(--text2);">✓ {f}</li>'
            for f in features
        )
        cards += f"""
        <div class="card" style="{"border-color:var(--accent);" if is_current else ""}text-align:center;">
          <div style="font-weight:800;font-size:16px;text-transform:capitalize;margin-bottom:4px;">{plan.title()}</div>
          <div style="font-size:32px;font-weight:900;color:var(--accent);margin-bottom:2px;">{price}<span style="font-size:13px;color:var(--text3);">/mo</span></div>
          <div style="font-family:var(--mono);font-size:11px;color:var(--text3);margin-bottom:16px;">{quota} generations</div>
          <ul style="list-style:none;text-align:left;margin-bottom:20px;">{feature_list}</ul>
          {"<span class='badge badge-green' style='font-size:11px;'>Current plan</span>" if is_current else
           f"<button class='btn btn-primary' style='width:100%;justify-content:center;' onclick='selectPlan(\"{plan}\")'>{_t(lang,'alert.upgrade')}</button>"}
        </div>"""
    content = f"""
    <div class="topbar">
      <div><div class="topbar-title">{_t(lang,"pricing.title")}</div>
        <div class="topbar-sub">{_t(lang,"pricing.current")}: {user.get("plan","free").title()}</div></div>
    </div>
    <div class="content">
      <div class="grid-4" style="gap:16px;">{cards}</div>
    </div>
    <script>
    function selectPlan(plan){{
      toast('Stripe billing coming soon — contact us at support@signalmind.ai', 'info');
    }}
    </script>"""
    return HTMLResponse(ui._page(content, user, _t(lang, "pricing.title"), "pricing", lang))
