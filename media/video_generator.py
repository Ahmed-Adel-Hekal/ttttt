"""media/video_generator.py — Video generation via AIML API or Google Gemini Veo.

Two concrete generators share a common interface:

    AimlVideoGenerator   — original Veo 3.1 via https://api.aimlapi.com/v2
                           (requires AIML_API_KEY, model = "google/veo-3.1-i2v")

    GeminiVideoGenerator — Veo 2 / Veo 3 directly via google-genai SDK
                           (requires GEMINI_API_KEY, model = "veo-2.0-generate-001"
                            or "veo-3.0-generate-preview")

Use VideoGeneratorFactory.create() — it returns the right generator based on
the video_provider argument ("aimlapi" or "gemini").

Both generators expose the same public method:
    generate_all(content_json: dict) -> list[VideoResult]

Backward-compat alias: VideoGenerator = AimlVideoGenerator
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger("VideoGenerator")

DEFAULT_MAX_POLLS    = 60
AIML_DEFAULT_MODEL   = "google/veo-3.1-i2v"
GEMINI_DEFAULT_MODEL = "veo-2.0-generate-001"


# ── JSON parser ────────────────────────────────────────────────────────────────
def parse_llm_json(raw: str):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1)
    else:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
    raw = raw.strip()

    def fix_nl(text):
        result = []; i = 0; in_str = False
        while i < len(text):
            ch = text[i]
            if in_str and ch == "\\" and i + 1 < len(text):
                result.append(ch); result.append(text[i+1]); i += 2; continue
            if ch == '"': in_str = not in_str; result.append(ch)
            elif in_str and ch in ("\n","\r"): result.append(" ")
            else: result.append(ch)
            i += 1
        return "".join(result)

    raw = fix_nl(raw)
    try: return json.loads(raw)
    except json.JSONDecodeError: pass
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try: return json.loads(cleaned)
    except json.JSONDecodeError: pass
    try:
        from json_repair import repair_json
        return json.loads(repair_json(cleaned))
    except Exception: pass
    raise ValueError(f"Could not parse JSON.\nFirst 300 chars:\n{raw[:300]}")


@dataclass
class VideoResult:
    idea_index:    int
    scene_index:   int
    generation_id: str
    status:        str
    video_url:     Optional[str] = None
    error:         Optional[str] = None


# ── Prompt builder ─────────────────────────────────────────────────────────────
class VeoPromptBuilder:
    @staticmethod
    def _build_character_text(character: dict, anchor: str = "") -> str:
        if anchor: return anchor
        if not character: return ""
        parts = []
        for key in ("gender","age","skin","hair","eye_color","facial_details","physical_details","outfit"):
            val = character.get(key) or character.get(key.replace("_"," "))
            if val: parts.append(str(val))
        if not parts: return ""
        expr = character.get("facial_expression","")
        return (
            f"CHARACTER ANCHOR — this exact person appears in EVERY scene: {', '.join(parts)}. "
            f"Face, hair, skin tone, build, and outfit are IDENTICAL across all scenes. "
            f"{'Current expression: '+expr+'. ' if expr else ''}"
            "This is the same continuous person throughout the entire video."
        )

    @staticmethod
    def _build_image_ref_hint(is_first: bool) -> str:
        if is_first:
            return ("Use the provided reference image as the exact visual anchor. "
                    "Match its subject, appearance, color palette, and style precisely in every scene.")
        return ("Maintain perfect visual consistency with the reference image. "
                "Subject, environment style, and color palette must remain identical to scene 1.")

    @staticmethod
    def _build_lighting(lighting: dict) -> str:
        if not lighting: return ""
        parts = []
        for k, label in [("camera_angle","camera angle"),("camera_type","camera"),
                          ("lighting_mode","lighting"),("lighting_position","light position"),
                          ("camera_movement","movement")]:
            val = lighting.get(k) or lighting.get(k.replace("_"," "))
            if val: parts.append(f"{label}: {val}")
        return "Cinematography — " + ", ".join(parts) + "." if parts else ""

    @staticmethod
    def _build_voiceover_style(vo_props: dict, language: str, voiceover_text: str) -> str:
        if not voiceover_text: return ""
        gender = (vo_props or {}).get("gender","Female")
        tone   = (vo_props or {}).get("tone","confident")
        return f'Voiceover: {gender} voice, {tone} tone, speaking in {language}: "{voiceover_text}".'

    @classmethod
    def build(cls, scene, hook, cta, visual_direction, brand_colors, language,
              image_url="", character=None, character_anchor="", style_anchor="",
              lighting=None, vo_props=None, is_first_scene=False, is_last_scene=False):
        vd          = visual_direction or {}
        brand_color = brand_colors[0] if brand_colors else "#FF0000"
        has_image   = bool(image_url and image_url.strip())

        char_block  = (cls._build_image_ref_hint(is_first_scene) if has_image
                       else cls._build_character_text(character or {}, anchor=character_anchor))
        light_block = (f"[LOCKED STYLE from scene 1] {style_anchor}"
                       if style_anchor and not is_first_scene
                       else cls._build_lighting(lighting or {}))
        vo_block    = cls._build_voiceover_style(vo_props or {}, language, scene.get("voiceover",""))
        hook_block  = (f'OPENING HOOK ({hook.get("duration_seconds",3)}s): bold on-screen text reads '
                       f'"{hook.get("text","")}" — eye-catching, high contrast, centered.'
                       if is_first_scene and hook else "")
        cta_block   = (f'END CTA: overlay text "{cta.get("text","")}" at {cta.get("placement","end")}.'
                       if is_last_scene and cta else "")
        overlay     = f'On-screen text: "{scene.get("text_overlay","")}".' if scene.get("text_overlay") else ""
        style       = (f"Brand color {brand_color}. {vd.get('color_usage','')} "
                       f"Pacing: {vd.get('pacing','medium')}. Transitions: {vd.get('transitions','cut')}. "
                       f"Vertical 9:16, professional social-media quality.")

        flat = " ".join(filter(None,[char_block,light_block,hook_block,
                                     f"Scene visuals: {scene.get('visuals','')}.",
                                     vo_block, overlay, cta_block, style])).strip()
        return flat, {"flat_prompt": flat, "scene": scene.get("scene",1)}


# ── FFmpeg joiner ──────────────────────────────────────────────────────────────
class VideoJoiner:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.concat_dir = os.path.join(output_dir, "concat")
        os.makedirs(self.concat_dir, exist_ok=True)

    @staticmethod
    def _ffmpeg_ok() -> bool:
        try:
            subprocess.run(["ffmpeg","-version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def join(self, scene_paths: list, idea_idx: int) -> Optional[str]:
        valid = [p for p in scene_paths if p and os.path.isfile(p)]
        if not valid: return None
        if len(valid) == 1: return valid[0]
        if not self._ffmpeg_ok():
            logger.warning("FFmpeg not available — cannot join %d scenes", len(valid))
            return None
        out   = os.path.join(self.output_dir, f"idea_{idea_idx+1}_full.mp4")
        lst   = os.path.join(self.concat_dir,  f"idea_{idea_idx+1}_concat.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p in valid:
                f.write(f"file '{os.path.abspath(p).replace(chr(92),'/')}'\n")
        # Try stream copy first, re-encode on failure
        for cmd in (
            ["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy", out],
            ["ffmpeg","-y","-f","concat","-safe","0","-i",lst,
             "-c:v","libx264","-preset","fast","-crf","18",
             "-c:a","aac","-b:a","192k","-movflags","+faststart", out],
        ):
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if r.returncode == 0: return out
            logger.warning("FFmpeg cmd failed: %s", cmd[:4])
        return None


# ── Shared scene helpers (used by both generators via inheritance) ──────────────
class _SceneHelperMixin:
    @staticmethod
    def _safe_get(idea, *keys, default=None):
        for key in keys:
            val = idea.get(key)
            if val is not None and val != {} and val != []: return val
        return default if default is not None else {}

    @staticmethod
    def _merge_delta(scene: dict, prev: dict) -> dict:
        if not prev: return scene
        merged = dict(prev); merged.update(scene)
        if scene.get("use_character") is False:
            merged.pop("character_details", None)
            return merged
        for key in ("character_details","lighting_conditions","visual_direction"):
            pv = prev.get(key) or {}; cv = scene.get(key) or {}
            if pv or cv: merged[key] = {**pv, **cv}
        return merged

    def _write_idea_json(self, idea: dict, idea_idx: int, scenes: list) -> str:
        caption = idea.get("caption","")
        if isinstance(caption, list): caption = " ".join(str(c) for c in caption)
        data = {
            "idea_index":       idea_idx+1,
            "caption":          str(caption),
            "hashtags":         idea.get("hashtags",[]),
            "hook":             idea.get("hook",{}),
            "cta":              idea.get("cta",{}),
            "script":           idea.get("script",[]),
            "generated_scenes": [s for s in scenes if s.get("status")=="completed"],
        }
        path = os.path.join(self.output_dir, f"idea_{idea_idx+1}.json")
        with open(path,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        return path

    def _finalise_idea(self, json_path: str, scene_paths: list, idea_idx: int):
        full = self.joiner.join(scene_paths, idea_idx)
        if full:
            try:
                with open(json_path) as f: d = json.load(f)
                d["full_video_path"] = full
                with open(json_path,"w") as f: json.dump(d,f,ensure_ascii=False,indent=2)
            except Exception: pass

    def _build_scene_anchors(self, scene_char, scene_light, visual_direction, image_url):
        character_anchor = VeoPromptBuilder._build_character_text(scene_char)
        style_parts = []
        if scene_light: style_parts.append(VeoPromptBuilder._build_lighting(scene_light))
        if (visual_direction or {}).get("pacing"):
            style_parts.append(f"pacing: {visual_direction['pacing']}")
        style_anchor = " | ".join(filter(None, style_parts))
        return character_anchor, style_anchor


# ══════════════════════════════════════════════════════════════════════════════
# Provider 1 — AIML API
# ══════════════════════════════════════════════════════════════════════════════
class AimlVideoGenerator(_SceneHelperMixin):
    """Veo 3.1 via AIML API. Requires AIML_API_KEY."""

    BASE_URL = "https://api.aimlapi.com/v2"

    def __init__(self, api_key, image_url="", language="English", brand_colors=None,
                 aspect_ratio="9:16", poll_interval=20, output_dir="output_videos",
                 model=AIML_DEFAULT_MODEL, max_polls=DEFAULT_MAX_POLLS, **_):
        self.api_key       = api_key
        self.image_url     = image_url or ""
        self.language      = language
        self.brand_colors  = brand_colors or ["#FF0000"]
        self.aspect_ratio  = aspect_ratio
        self.poll_interval = poll_interval
        self.output_dir    = output_dir
        self.model         = model or AIML_DEFAULT_MODEL
        self.max_polls     = max_polls
        os.makedirs(output_dir, exist_ok=True)
        self.joiner = VideoJoiner(output_dir)

    def _headers(self): return {"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}

    def _submit(self, prompt: str) -> Optional[str]:
        payload = {"model":self.model,"prompt":prompt,"aspect_ratio":self.aspect_ratio}
        if self.image_url: payload["image_url"] = self.image_url
        try:
            r = requests.post(f"{self.BASE_URL}/video/generations",
                              json=payload, headers=self._headers(), timeout=60)
            if r.status_code >= 400:
                logger.error("AIML submit %d: %s", r.status_code, r.text[:200]); return None
            return r.json().get("id")
        except requests.RequestException as e:
            logger.error("AIML submit failed: %s", e); return None

    def _poll(self, gen_id: str) -> tuple[Optional[str], Optional[str]]:
        logger.info("AIML polling %s", gen_id)
        for _ in range(self.max_polls):
            time.sleep(self.poll_interval)
            try:
                r = requests.get(f"{self.BASE_URL}/video/generations",
                                 params={"generation_id":gen_id},
                                 headers=self._headers(), timeout=30)
                if r.status_code >= 400: return None, f"Poll HTTP {r.status_code}"
                data = r.json(); status = data.get("status","")
                if status == "completed": return data.get("video",{}).get("url"), None
                if status in ("failed","error"): return None, str(data.get("error","Generation failed"))
            except requests.RequestException as e:
                logger.warning("AIML poll failed: %s", e)
        return None, f"Timed out after {self.max_polls} polls"

    def _download(self, url: str, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        try:
            r = requests.get(url, timeout=120)
            with open(path,"wb") as f: f.write(r.content)
            return path
        except requests.RequestException as e:
            logger.error("AIML download failed: %s", e); return url

    def generate_all(self, content_json: dict) -> list[VideoResult]:
        ideas = content_json.get("ideas",[]); results = []; builder = VeoPromptBuilder()
        for idea_idx, idea in enumerate(ideas):
            hook    = idea.get("hook",{}); script  = idea.get("script",[])
            cta     = idea.get("cta",{}); n       = len(script)
            vd      = self._safe_get(idea,"visual_direction","visual direction")
            char    = self._safe_get(idea,"character_details","character details")
            light   = self._safe_get(idea,"lighting_conditions","Lighting condition")
            vo      = self._safe_get(idea,"voiceover_properties","voiceover_props")
            prev    = {}; paths = []; scenes_out = []
            char_anchor = ""; style_anchor = ""

            for si, scene in enumerate(script):
                is_first = si == 0; is_last = si == n-1
                snum      = scene.get("scene", si+1)
                full      = self._merge_delta(scene, prev); prev = full
                sc        = full.get("character_details") or char or {}
                sl        = full.get("lighting_conditions") or light or {}
                if is_first and not self.image_url:
                    char_anchor, style_anchor = self._build_scene_anchors(sc, sl, vd, self.image_url)

                prompt, _ = builder.build(
                    scene=full, hook=hook, cta=cta, visual_direction=vd,
                    brand_colors=self.brand_colors, language=self.language,
                    image_url=self.image_url, character=sc,
                    character_anchor=char_anchor, style_anchor=style_anchor,
                    lighting=sl, vo_props=vo,
                    is_first_scene=is_first, is_last_scene=is_last,
                )

                gen_id = self._submit(prompt)
                if not gen_id:
                    err = "AIML submission failed"
                    scenes_out.append({"scene":snum,"status":"failed","error":err})
                    results.append(VideoResult(idea_idx,si,"",status="failed",error=err))
                    continue

                video_url, poll_err = self._poll(gen_id)
                if not video_url:
                    err = poll_err or "Generation failed"
                    scenes_out.append({"scene":snum,"status":"failed","generation_id":gen_id,"error":err})
                    results.append(VideoResult(idea_idx,si,gen_id,status="failed",error=err))
                    continue

                local = self._download(video_url, f"idea{idea_idx+1}_scene{snum}.mp4")
                scenes_out.append({"scene":snum,"status":"completed","generation_id":gen_id,"video_path":local})
                results.append(VideoResult(idea_idx,si,gen_id,status="completed",video_url=local))
                paths.append(local)

            jp = self._write_idea_json(idea, idea_idx, scenes_out)
            self._finalise_idea(jp, paths, idea_idx)

        done=[r for r in results if r.status=="completed"]
        logger.info("AIML: %d completed, %d failed", len(done), len(results)-len(done))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Provider 2 — Google Gemini Veo (google-genai SDK)
# ══════════════════════════════════════════════════════════════════════════════
class GeminiVideoGenerator(_SceneHelperMixin):
    """
    Veo 2 / Veo 3 directly via google-genai SDK.

    Models:
        "veo-2.0-generate-001"       GA, 5-8 s, image-to-video
        "veo-3.0-generate-preview"   Preview, audio support
    """

    _ASPECT_MAP   = {"9:16":"9:16","16:9":"16:9","1:1":"1:1"}
    _POLL_SECONDS = 15
    _MAX_POLLS    = 80   # 80 × 15 s = 20 min

    def __init__(self, api_key, image_url="", language="English", brand_colors=None,
                 aspect_ratio="9:16", output_dir="output_videos",
                 model=GEMINI_DEFAULT_MODEL, **_):
        self.api_key      = api_key
        self.image_url    = image_url or ""
        self.language     = language
        self.brand_colors = brand_colors or ["#FF0000"]
        self.aspect_ratio = self._ASPECT_MAP.get(aspect_ratio, "9:16")
        self.output_dir   = output_dir
        self.model        = model or GEMINI_DEFAULT_MODEL
        os.makedirs(output_dir, exist_ok=True)
        self.joiner   = VideoJoiner(output_dir)
        self._client  = None

    def _get_client(self):
        if self._client: return self._client
        try:
            import google.genai as genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except ImportError:
            raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai>=1.0.0")

    def _image_ref(self):
        if not self.image_url: return None
        try:
            import google.genai.types as T
            url = self.image_url.strip()
            if url.startswith("http://") or url.startswith("https://"):
                r = requests.get(url, timeout=20); r.raise_for_status()
                mime = r.headers.get("Content-Type","image/jpeg").split(";")[0]
                return T.Image(image_bytes=r.content, mime_type=mime)
            with open(url,"rb") as f: data = f.read()
            return T.Image(image_bytes=data, mime_type="image/jpeg")
        except Exception as e:
            logger.warning("GeminiVideo: could not load image ref: %s", e); return None

    def _generate_clip(self, prompt: str, idea_idx: int, scene_num: int) -> Optional[str]:
        try:
            import google.genai.types as T
        except ImportError:
            logger.error("google-genai not installed"); return None

        client = self._get_client()
        kwargs = {
            "model":  self.model,
            "prompt": prompt,
            "config": T.GenerateVideoConfig(
                aspect_ratio=self.aspect_ratio,
                number_of_videos=1,
            ),
        }
        img_ref = self._image_ref()
        if img_ref: kwargs["image"] = img_ref

        try:
            operation = client.models.generate_videos(**kwargs)
        except Exception as e:
            logger.error("GeminiVideo: generate_videos failed (idea %d scene %d): %s",
                         idea_idx+1, scene_num, e)
            return None

        logger.info("GeminiVideo: polling operation (idea %d scene %d)", idea_idx+1, scene_num)
        for _ in range(self._MAX_POLLS):
            time.sleep(self._POLL_SECONDS)
            try: operation = client.operations.get(operation)
            except Exception as e:
                logger.warning("GeminiVideo poll error: %s", e); continue

            if not getattr(operation,"done",False): continue

            if getattr(operation,"error",None):
                logger.error("GeminiVideo operation error: %s", operation.error); return None

            result = getattr(operation,"result",None)
            if not result: logger.error("GeminiVideo: no result"); return None

            vids = getattr(result,"generated_videos",[]) or []
            if not vids: logger.error("GeminiVideo: empty generated_videos"); return None

            video = getattr(vids[0],"video",None)
            if not video: logger.error("GeminiVideo: video object None"); return None

            return self._save_clip(video, idea_idx, scene_num)

        logger.error("GeminiVideo: timed out (idea %d scene %d)", idea_idx+1, scene_num)
        return None

    def _save_clip(self, video_obj, idea_idx: int, scene_num: int) -> Optional[str]:
        filename = f"idea{idea_idx+1}_scene{scene_num}.mp4"
        out_path = os.path.join(self.output_dir, filename)
        uri    = getattr(video_obj,"uri",None)
        vbytes = getattr(video_obj,"video_bytes",None)
        try:
            if uri:
                r = requests.get(uri,
                                 headers={"Authorization":f"Bearer {self.api_key}"},
                                 timeout=120)
                r.raise_for_status()
                with open(out_path,"wb") as f: f.write(r.content)
            elif vbytes:
                with open(out_path,"wb") as f: f.write(vbytes)
            else:
                logger.error("GeminiVideo: no uri or video_bytes"); return None
            logger.info("GeminiVideo: saved %s", out_path)
            return out_path
        except Exception as e:
            logger.error("GeminiVideo: save failed: %s", e); return None

    def generate_all(self, content_json: dict) -> list[VideoResult]:
        ideas = content_json.get("ideas",[]); results = []; builder = VeoPromptBuilder()
        for idea_idx, idea in enumerate(ideas):
            hook = idea.get("hook",{}); script = idea.get("script",[])
            cta  = idea.get("cta",{});  n      = len(script)
            vd   = self._safe_get(idea,"visual_direction","visual direction")
            char = self._safe_get(idea,"character_details","character details")
            light= self._safe_get(idea,"lighting_conditions","Lighting condition")
            vo   = self._safe_get(idea,"voiceover_properties","voiceover_props")
            prev = {}; paths = []; scenes_out = []
            char_anchor = ""; style_anchor = ""

            for si, scene in enumerate(script):
                is_first = si == 0; is_last = si == n-1
                snum = scene.get("scene", si+1)
                full = self._merge_delta(scene, prev); prev = full
                sc   = full.get("character_details") or char or {}
                sl   = full.get("lighting_conditions") or light or {}
                if is_first and not self.image_url:
                    char_anchor, style_anchor = self._build_scene_anchors(sc, sl, vd, self.image_url)

                prompt, _ = builder.build(
                    scene=full, hook=hook, cta=cta, visual_direction=vd,
                    brand_colors=self.brand_colors, language=self.language,
                    image_url=self.image_url, character=sc,
                    character_anchor=char_anchor, style_anchor=style_anchor,
                    lighting=sl, vo_props=vo,
                    is_first_scene=is_first, is_last_scene=is_last,
                )

                local = self._generate_clip(prompt, idea_idx, snum)
                if local:
                    scenes_out.append({"scene":snum,"status":"completed","video_path":local})
                    results.append(VideoResult(idea_idx,si,"gemini","completed",video_url=local))
                    paths.append(local)
                else:
                    err = f"Gemini clip failed (idea {idea_idx+1} scene {snum})"
                    scenes_out.append({"scene":snum,"status":"failed","error":err})
                    results.append(VideoResult(idea_idx,si,"gemini","failed",error=err))

            jp = self._write_idea_json(idea, idea_idx, scenes_out)
            self._finalise_idea(jp, paths, idea_idx)

        done = [r for r in results if r.status=="completed"]
        logger.info("Gemini: %d completed, %d failed", len(done), len(results)-len(done))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════
class VideoGeneratorFactory:
    """
    Single call-site used by content_agent.py:

        gen = VideoGeneratorFactory.create(
            video_provider="gemini",   # "aimlapi" | "gemini"
            api_key=...,
            image_url=...,
            language=...,
            brand_colors=...,
            aspect_ratio=...,
            output_dir=...,
            model=...,
        )
        results = gen.generate_all(content_json)
    """
    _MAP = {
        "aimlapi": AimlVideoGenerator,
        "aiml":    AimlVideoGenerator,
        "gemini":  GeminiVideoGenerator,
        "google":  GeminiVideoGenerator,
    }

    @classmethod
    def create(cls, video_provider: str = "aimlapi", **kwargs):
        key   = (video_provider or "aimlapi").lower().strip()
        klass = cls._MAP.get(key, AimlVideoGenerator)
        logger.info("VideoGeneratorFactory: %s → %s", video_provider, klass.__name__)
        return klass(**kwargs)

    @classmethod
    def models_for_provider(cls, provider: str) -> list[dict]:
        p = (provider or "aimlapi").lower()
        if p in ("gemini","google"):
            return [
                {"id":"veo-3.1-fast-generate-preview",   "name":"Veo 3.1 Fast",
                 "description":"Latest · audio · 4K cap · recommended","recommended":True},
                {"id":"veo-3.1-generate-preview",        "name":"Veo 3.1",
                 "description":"High quality · 4K · audio","recommended":False},
                {"id":"veo-2.0-generate-001",            "name":"Veo 2.0",
                 "description":"Stable · 5-8s · image-to-video","recommended":False},
            ]
        return [
            {"id":"google/veo-3.1-i2v",        "name":"Veo 3.1 (image-to-video)",
             "description":"Via AIML API · recommended","recommended":True},
            {"id":"google/veo-3.0-t2v-480p",   "name":"Veo 3.0 480p (text-to-video)",
             "description":"Via AIML API · no reference image needed","recommended":False},
            {"id":"google/veo-2.0-i2v",        "name":"Veo 2.0 (image-to-video)",
             "description":"Via AIML API · stable","recommended":False},
        ]


# Backward-compat alias
VideoGenerator = AimlVideoGenerator
