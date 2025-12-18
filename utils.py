from __future__ import annotations

import json
import math
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from schemas import Prompt, PromptsFile, Storyboard

NEGATIVE_LIBRARY = [
    "No sudden zooms",
    "No random cuts",
    "No camera shake",
    "Do not change character outfits",
    "Do not add new characters",
    "Keep subject centered",
    "Do not change environment",
    "No fisheye lenses",
]


def normalize_azure_openai_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return endpoint

    parts = urlsplit(endpoint)
    if parts.scheme and parts.netloc:
        base = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    else:
        base = endpoint.rstrip("/")

    lowered = base.lower()
    for suffix in ("/openai/v1", "/openai"):
        if lowered.endswith(suffix):
            base = base[: -len(suffix)]
            lowered = base.lower()

    return base.rstrip("/")


def runs_root() -> Path:
    return Path("runs")


def make_job_id(project: Optional[str] = None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    suffix = secrets.token_hex(3)
    if project:
        project_slug = "".join(c for c in project.lower().replace(" ", "-") if c.isalnum() or c == "-")
        return f"{timestamp}-{project_slug}-{suffix}"
    return f"{timestamp}-{suffix}"


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_run_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def append_log(run_dir: Path, message: str) -> None:
    log_path = run_dir / "logs.txt"
    timestamp = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def ensure_clip_counts(total_seconds: int, clip_seconds: int) -> Tuple[int, int]:
    clips = max(1, math.ceil(total_seconds / clip_seconds))
    normalized_total = clips * clip_seconds
    return clips, normalized_total


def parse_size(size: str) -> Tuple[int, int]:
    raw = (size or "").strip().lower().replace(" ", "")
    if "x" not in raw:
        raise ValueError(f"Invalid size '{size}'; expected like '1280x720'")
    w_s, h_s = raw.split("x", 1)
    if not (w_s.isdigit() and h_s.isdigit()):
        raise ValueError(f"Invalid size '{size}'; expected like '1280x720'")
    w = int(w_s)
    h = int(h_s)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size '{size}'; width/height must be > 0")
    return w, h


def prepare_reference_image(
    src_path: Path,
    dst_path: Path,
    target_w: int,
    target_h: int,
    mode: str = "cover",
    jpeg_quality: int = 95,
) -> Path:
    """Prepare an image for Sora `input_reference`.

    Resizes to *exactly* target_w x target_h.

    mode:
      - "cover": scale to fill then center-crop (recommended; no letterbox bars)
      - "contain": scale to fit then pad (letterbox)
    """

    if not src_path.exists():
        raise FileNotFoundError(f"Reference image not found: {src_path}")

    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency 'Pillow'. Install it with: pip install Pillow") from exc

    if target_w <= 0 or target_h <= 0:
        raise ValueError("target_w/target_h must be > 0")

    mode = (mode or "cover").strip().lower()
    if mode not in {"cover", "contain"}:
        raise ValueError("mode must be 'cover' or 'contain'")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in {"RGB", "RGBA"}:
            im = im.convert("RGB")

        src_w, src_h = im.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError("Invalid source image dimensions")

        if mode == "cover":
            scale = max(target_w / src_w, target_h / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            im2 = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = max(0, (new_w - target_w) // 2)
            top = max(0, (new_h - target_h) // 2)
            right = left + target_w
            bottom = top + target_h
            im2 = im2.crop((left, top, right, bottom))
        else:
            # contain + pad
            scale = min(target_w / src_w, target_h / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            im2 = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if im2.mode != "RGB":
                im2 = im2.convert("RGB")
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            left = (target_w - new_w) // 2
            top = (target_h - new_h) // 2
            canvas.paste(im2, (left, top))
            im2 = canvas

        if im2.mode != "RGB":
            im2 = im2.convert("RGB")
        im2.save(dst_path, format="JPEG", quality=jpeg_quality, optimize=True)

    return dst_path


def build_storyboard(
    job_id: str,
    goal: str,
    total_seconds: int,
    clip_seconds: int,
    aspect_ratio: str,
    seed: Optional[int],
    style: str,
    camera: str,
    lighting: str,
    negatives: List[str],
    characters: str,
    environment: str,
    template: Optional[str] = None,
) -> Storyboard:
    clips, normalized_total = ensure_clip_counts(total_seconds, clip_seconds)
    return Storyboard(
        version="1.0",
        job_id=job_id,
        total_seconds=normalized_total,
        clip_seconds=clip_seconds,
        aspect_ratio=aspect_ratio,
        seed=seed,
        global_style=style,
        camera=camera,
        lighting=lighting,
        negatives=negatives,
        goal=goal,
        characters=characters,
        environment=environment,
        template=template,
    )


def _clip_prompt_intro(index: int, total_clips: int, goal: str) -> str:
    if index == 1:
        return f"Scene opens (beat 1 of {total_clips}) focused on: {goal}."
    return f"Continue seamlessly from prior frame (beat {index} of {total_clips}) to advance: {goal}."


def _negatives_text(negatives: List[str]) -> str:
    if not negatives:
        return ""
    items = "\n".join([f"  - {n}" for n in negatives])
    return f"Negatives:\n{items}"


def is_training_goal(text: str) -> bool:
    t = (text or "").lower()
    return any(
        key in t
        for key in (
            "instructor-led training",
            "ilt course title",
            "modality:",
            "prerequisites:",
            "this course will prepare you to",
        )
    )


def _sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _clamp_list(items: List[str], max_items: int) -> List[str]:
    out: List[str] = []
    for x in items:
        s = (x or "").strip()
        if not s:
            continue
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def generate_training_script(goal_text: str, total_clips: int, clip_seconds: int) -> Dict[str, object]:
    """Compatibility wrapper for a deterministic fallback script.

    Preferred: use `PromptLLMClient.generate_training_script` (LLM-assisted) and then
    pass the result through `training_script_to_segments`.
    """

    return build_training_script_fallback(goal_text, total_clips, clip_seconds)


def extract_training_outline(text: str) -> Dict[str, object]:
    """Best-effort parser for ILT-style goal text.

    Returns a dict with keys:
      - course_title: str
      - learning_level: str
      - modality: str
      - goals: List[str]
      - prerequisites: List[str]
      - domain_tags: List[str]  (keywords to keep the visuals on-topic)
    """

    raw = (text or "").strip()
    course_title = ""
    learning_level = ""
    modality = ""

    # Title (handles: ILT Course Title: X or ILT Course Title: 'X')
    m = re.search(r"ILT\s*Course\s*Title\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        course_title = m.group(1).strip().strip('"').strip("'")
        # Many users continue the sentence on the same line. Trim common continuations.
        for token in (" (Learning Level", " (Level", " Modality:", " Training description:"):
            if token.lower() in course_title.lower():
                course_title = re.split(re.escape(token), course_title, flags=re.IGNORECASE)[0].strip()
        course_title = course_title.rstrip(". ")
        course_title = course_title.strip(" '\"")

    # Learning level (handles: (Learning Level: Intermediate))
    m = re.search(r"Learning\s*Level\s*:\s*([A-Za-z\- ]+)", raw, flags=re.IGNORECASE)
    if m:
        learning_level = m.group(1).strip().rstrip(")").strip()

    # Modality
    m = re.search(r"Modality\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        modality = m.group(1).strip()

    def _block(after_header: str, until_headers: List[str]) -> str:
        start = re.search(after_header, raw, flags=re.IGNORECASE)
        if not start:
            return ""
        start_idx = start.end()
        tail = raw[start_idx:]
        end_idx = len(tail)
        for h in until_headers:
            mm = re.search(h, tail, flags=re.IGNORECASE)
            if mm:
                end_idx = min(end_idx, mm.start())
        return tail[:end_idx].strip()

    goals_block = _block(r"\bGoals\s*:\s*", [r"\bPrerequisites\s*:\s*", r"\bModality\s*:\s*", r"\bCourse\s*Duration\s*:\s*"])
    prereq_block = _block(r"\bPrerequisites\s*:\s*", [r"\bModality\s*:\s*", r"\bCourse\s*Duration\s*:\s*", r"\bTone\s*:\s*"])

    def _lines(block: str) -> List[str]:
        out: List[str] = []
        for line in (block or "").splitlines():
            s = line.strip().lstrip("-•*").strip()
            if not s:
                continue
            # Skip intro helper line
            if s.lower() in {"this course will prepare you to:", "this course will prepare you to"}:
                continue
            out.append(s)
        return out

    goals = _lines(goals_block)
    prerequisites = _lines(prereq_block)

    # Domain tags help keep the visuals anchored to the syllabus language.
    domain_tags: List[str] = []
    domain_lower = raw.lower()
    for tag in (
        "m3 cloud",
        "food and beverage",
        "system navigation",
        "planning",
        "procurement",
        "production",
        "inventory",
        "distribution",
        "order fulfillment",
        "finance",
        "warehouse",
        "quality",
        "safety",
        "integration",
    ):
        if tag in domain_lower:
            domain_tags.append(tag)

    return {
        "course_title": course_title,
        "learning_level": learning_level,
        "modality": modality,
        "goals": goals,
        "prerequisites": prerequisites,
        "domain_tags": domain_tags,
    }


def build_training_script_fallback(
    course_block: str,
    total_scenes: int,
    scene_seconds: int,
) -> Dict[str, object]:
    """Deterministic fallback script when LLM isn't available."""

    outline = extract_training_outline(course_block)
    title = str(outline.get("course_title") or "Course Introduction").strip() or "Course Introduction"
    level = str(outline.get("learning_level") or "").strip()
    subtitle = "End-to-end training overview" if not level else f"{level} training overview"

    def _as_str_list(value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    goals = _as_str_list(outline.get("goals"))
    prereqs = _as_str_list(outline.get("prerequisites"))

    scenes: List[Dict[str, object]] = []
    # Scene 1: title
    scenes.append(
        {
            "index": 1,
            "duration_seconds": scene_seconds,
            "narration": (
                f"Welcome to {title}. In this short introduction, you'll learn what the course covers and how it helps you build practical, job-ready skills."
            ),
            "on_screen_text": {"lines": [title, subtitle, "Instructor-led training"]},
            "visuals": "Friendly light-brown teddy bear host speaks to camera in a clean studio. Light motion graphics show labeled tiles near the host.",
        }
    )

    # Middle scenes: goals chunked
    remaining = total_scenes - 2
    if remaining < 0:
        remaining = 0
    chunks: List[List[str]] = []
    if goals and remaining > 0:
        per = max(1, math.ceil(len(goals) / remaining))
        for i in range(0, len(goals), per):
            chunks.append(goals[i : i + per])
        while len(chunks) < remaining:
            chunks.append([])
        chunks = chunks[:remaining]
    else:
        chunks = [[] for _ in range(remaining)]

    for idx in range(remaining):
        chunk = chunks[idx] if idx < len(chunks) else []
        lines = ["What you'll learn"]
        lines.extend(chunk[:6] if chunk else ["Key workflows", "Core terminology", "Practical examples"])
        narration = " ".join(
            [
                "In this course, you'll focus on practical workflows.",
                "You'll work through hands-on steps and knowledge checks that reinforce the concepts.",
            ]
        )
        if chunk:
            narration = "You will learn to: " + "; ".join(chunk[:4]) + "."
        scenes.append(
            {
                "index": len(scenes) + 1,
                "duration_seconds": scene_seconds,
                "narration": narration,
                "on_screen_text": {"lines": lines},
                "visuals": "Host gestures while bullet tiles fade in sequentially; subtle domain UI silhouettes in the background; clean typography.",
            }
        )

    # Final scene: prerequisites + CTA
    final_lines = ["Prerequisites"]
    final_lines.extend(prereqs[:5] if prereqs else ["Baseline navigation knowledge", "Planning fundamentals"])
    final_lines.append("Let's begin")
    scenes.append(
        {
            "index": len(scenes) + 1,
            "duration_seconds": scene_seconds,
            "narration": (
                "Before you start, review the prerequisites listed here. When you're ready, enroll and begin building hands-on expertise in this course."
            ),
            "on_screen_text": {"lines": final_lines},
            "visuals": "Host smiles and gives a small wave; checklist icons appear; clean fade-out on 'Let's begin'.",
        }
    )

    return {
        "title": title,
        "subtitle": subtitle,
        "host": {"description": "Light-brown teddy bear host (friendly, non-real)", "voice": "Same narrator voice across all scenes"},
        "aesthetic": {
            "style": "Modern enterprise training aesthetic",
            "typography": "Clean sans-serif, high contrast",
            "motion_graphics": "Light motion graphics with labeled tiles and sequential bullets",
        },
        "scenes": scenes[:total_scenes],
    }


def training_script_to_segments(script: Dict[str, object], total_clips: int) -> List[str]:
    """Convert script scenes into segment strings usable by `build_prompts`."""

    voice_profile = (
        "Single narrator voice (MUST stay identical across all clips): gentle adult male voice (warm baritone), "
        "friendly and calm cadence; clear American English; steady pitch; close-mic studio voiceover; "
        "consistent loudness; no voice switching; no accent drift."
    )

    scenes_raw = script.get("scenes") if isinstance(script, dict) else None
    if not isinstance(scenes_raw, list):
        return ["" for _ in range(total_clips)]

    segments: List[str] = []
    for idx in range(min(total_clips, len(scenes_raw))):
        scene = scenes_raw[idx]
        if not isinstance(scene, dict):
            segments.append("Present the training content with clear on-screen headings.")
            continue

        narration = str(scene.get("narration") or "").strip()
        ost = scene.get("on_screen_text")
        lines: List[str] = []
        if isinstance(ost, dict):
            raw_lines = ost.get("lines")
            if isinstance(raw_lines, list):
                for l in raw_lines:
                    if isinstance(l, str) and l.strip():
                        lines.append(l.strip())
        visuals = str(scene.get("visuals") or "").strip()

        # Sora prompt segment: enforce exact text and encourage visible speaking + captions.
        exact_text_block = "\n".join([f"• {l}" for l in lines]) if lines else ""
        seg = (
            "TRAINING SCRIPT SEGMENT (follow strictly): "
            "Show the host speaking to camera with calm gestures and visible mouth movement. "
            "Do NOT invent new headings. Render the on-screen text EXACTLY as provided. "
        )
        if exact_text_block:
            seg += "\nON-SCREEN TEXT (EXACT):\n" + exact_text_block + "\n"
        # Sora 2 guide: put spoken lines in a labeled dialogue block below the prose.
        if narration:
            safe_line = narration.replace('"', "'").strip()
            if safe_line:
                seg += (
                    "\nDIALOGUE AND AUDIO (follow strictly):\n"
                    f"Audio: {voice_profile}\n"
                    "Dialogue:\n"
                    f"- Narrator (same voice as clip 1, voiceover/off-screen): \"{safe_line}\"\n"
                    "Background Sound: subtle room tone only; no music; no extra speakers.\n"
                )
        if visuals:
            seg += "\nVISUALS:\n" + visuals + "\n"

        seg += "EXIT MOTION STATE: gentle camera glide continues, host centered, hands return to neutral." 
        segments.append(seg.strip())

    while len(segments) < total_clips:
        segments.append("Continue the training introduction with a clear heading and readable bullets. EXIT MOTION STATE: gentle camera glide, host centered.")
    return segments[:total_clips]


def training_script_voiceover_text(script: Dict[str, object]) -> str:
    scenes_raw = script.get("scenes") if isinstance(script, dict) else None
    if not isinstance(scenes_raw, list):
        return ""
    parts: List[str] = []
    for s in scenes_raw:
        if not isinstance(s, dict):
            continue
        idx = s.get("index")
        narration = str(s.get("narration") or "").strip()
        if narration:
            parts.append(f"Scene {idx}: {narration}")
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


def training_script_to_srt(script: Dict[str, object]) -> str:
    """Simple SRT using cumulative scene durations (for post voiceover/captions)."""

    scenes_raw = script.get("scenes") if isinstance(script, dict) else None
    if not isinstance(scenes_raw, list):
        return ""

    def _fmt(t: float) -> str:
        ms = int(round(t * 1000))
        h = ms // 3_600_000
        ms -= h * 3_600_000
        m = ms // 60_000
        ms -= m * 60_000
        s = ms // 1000
        ms -= s * 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out: List[str] = []
    t0 = 0.0
    n = 1
    for scene in scenes_raw:
        if not isinstance(scene, dict):
            continue
        dur = scene.get("duration_seconds")
        try:
            dur_s = float(dur) if dur is not None else 0.0
        except Exception:
            dur_s = 0.0
        if dur_s <= 0:
            dur_s = 1.0
        narration = str(scene.get("narration") or "").strip()
        if not narration:
            t0 += dur_s
            continue
        out.append(str(n))
        out.append(f"{_fmt(t0)} --> {_fmt(t0 + dur_s)}")
        out.append(narration)
        out.append("")
        n += 1
        t0 += dur_s
    return "\n".join(out).strip() + ("\n" if out else "")


def build_training_segments(goal_text: str, total_clips: int) -> List[str]:
    """Deterministic ILT intro outline that stays on-topic without an LLM."""

    outline = extract_training_outline(goal_text)
    title = str(outline.get("course_title") or "")
    level = str(outline.get("learning_level") or "")
    modality = str(outline.get("modality") or "")

    def _as_str_list(value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    goals = _as_str_list(outline.get("goals"))
    prereqs = _as_str_list(outline.get("prerequisites"))

    if not title:
        # Fallback: use first sentence-ish chunk
        title = (goal_text or "").strip().splitlines()[0][:120]

    course_anchor = title
    if level:
        course_anchor = f"{course_anchor} ({level})"

    # Allocate clips: 1 = title, last = prerequisites/close, middle = goals chunks.
    if total_clips <= 1:
        return [f"ON-SCREEN HEADING: '{course_anchor}'. Avatar introduces the course and who it's for. EXIT MOTION STATE: stable presenter posture, gentle camera glide." ]

    mid_slots = max(0, total_clips - 2)
    goal_chunks: List[List[str]] = []
    if goals and mid_slots > 0:
        per = max(1, math.ceil(len(goals) / mid_slots))
        for i in range(0, len(goals), per):
            goal_chunks.append(goals[i : i + per])
        # pad
        while len(goal_chunks) < mid_slots:
            goal_chunks.append([])
        goal_chunks = goal_chunks[:mid_slots]
    else:
        goal_chunks = [[] for _ in range(mid_slots)]

    segments: List[str] = []

    intro_line = f"ON-SCREEN HEADING (verbatim): '{course_anchor}'."
    if modality:
        intro_line += f" Callout text: 'Modality: {modality}'."
    intro_line += " Avatar greets learners and states what the course covers in one sentence, staying strictly on the course domain."
    intro_line += " EXIT MOTION STATE: slow forward dolly, presenter centered, calm gestures."
    segments.append(intro_line)

    for idx in range(mid_slots):
        chunk = goal_chunks[idx] if idx < len(goal_chunks) else []
        if chunk:
            bullets = "; ".join(chunk[:4])
            heading = f"ON-SCREEN HEADING (verbatim): 'What you'll learn — Part {idx + 1}'."
            say = f" Avatar references these outcomes by name: {bullets}."
        else:
            heading = f"ON-SCREEN HEADING (verbatim): 'Course outcomes — Part {idx + 1}'."
            say = " Avatar previews the upcoming hands-on workflows with labeled UI callouts (not abstract)."
        seg = heading + say + " EXIT MOTION STATE: gentle camera arc, presenter remains centered, gestures settle." 
        segments.append(seg)

    outro_heading = "ON-SCREEN HEADING (verbatim): 'Prerequisites & next steps'."
    if prereqs:
        prereq_list = "; ".join(prereqs[:4])
        outro_say = f" Avatar states prerequisites by name: {prereq_list}."
    else:
        outro_say = " Avatar states any required baseline knowledge in one line."
    outro_close = " Closing call-to-action text: 'Let\'s begin'. EXIT MOTION STATE: camera eases to stop, presenter neutral stance." 
    segments.append(outro_heading + outro_say + outro_close)

    return segments[:total_clips]


def build_prompts(
    storyboard: Storyboard,
    total_clips: int,
    segments: Optional[List[str]] = None,
    template: Optional[str] = None,
) -> PromptsFile:
    effective_template = (template or getattr(storyboard, "template", None) or "default").strip().lower()
    if effective_template not in {"default", "training-intro", "training-script", "auto"}:
        effective_template = "default"

    if effective_template == "auto":
        effective_template = "training-intro" if is_training_goal(storyboard.goal) else "default"

    if effective_template == "training-script" and not segments:
        script = build_training_script_fallback(storyboard.goal, total_clips, storyboard.clip_seconds)
        segments = training_script_to_segments(script, total_clips)

    if effective_template == "training-intro" and not segments:
        segments = build_training_segments(storyboard.goal, total_clips)

    training_preamble = ""
    if effective_template in {"training-intro", "training-script"}:
        outline = extract_training_outline(storyboard.goal)
        title = str(outline.get("course_title") or "").strip()
        level = str(outline.get("learning_level") or "").strip()
        domain_tags_raw = outline.get("domain_tags")
        domain_tags = domain_tags_raw if isinstance(domain_tags_raw, list) else []

        if not title:
            title = (storyboard.goal or "").strip().splitlines()[0][:120]
        title_line = title if not level else f"{title} ({level})"

        keywords = ", ".join(domain_tags[:8]) if domain_tags else ""
        if effective_template == "training-script":
            training_preamble = (
                "TRAINING SCRIPT ANCHORS (follow strictly):\n"
                f"- Course title: {title_line}\n"
                "- Render on-screen text EXACTLY as provided in each scene block (spelling/case/punctuation).\n"
                "- Single consistent narrator voice throughout (gentle adult male voice, warm baritone; same speaker, same timbre/accent/pitch/cadence).\n"
                "- Friendly light-brown teddy bear host on camera in every clip; do not change appearance/outfit.\n"
                "- For spoken lines, put them in a labeled 'Dialogue:' block below the prose; keep them concise to match the clip length.\n"
                "- Audio must NOT introduce new speakers, voice changes, accent shifts, or background music.\n"
                + (f"- Domain keywords to stay on-topic: {keywords}\n" if keywords else "")
            ).strip()
        else:
            training_preamble = (
                "TRAINING ANCHORS (follow strictly):\n"
                f"- Course title (verbatim on-screen at least once): {title_line}\n"
                "- Keep visuals and narration tightly scoped to this course (no generic corporate training).\n"
                "- Every clip must include a clear, readable on-screen heading that matches the segment heading text.\n"
                "- If there is voiceover/dialogue, use a single consistent narrator voice across all clips; no voice switching.\n"
                + (f"- Domain keywords to stay on-topic: {keywords}\n" if keywords else "")
            ).strip()

    clips: List[Prompt] = []
    for i in range(1, total_clips + 1):
        segment_action = None
        if segments and len(segments) >= i:
            segment_action = segments[i - 1]

        if i == 1:
            # Clip 1: Establish everything (Sora 2 guide Clip 1 template)
            prompt_text = (
                f"Scene: {storyboard.environment}. "
                f"Lighting: {storyboard.lighting}. "
                f"Characters: {storyboard.characters}. "
                f"Camera: {storyboard.camera}. "
                f"Style: {storyboard.global_style}. \n"
                + (f"\n{training_preamble}\n\n" if training_preamble else "")
                + f"Action (segment 1 of {total_clips}): {segment_action or 'establish the scene and begin the story'}. \n"
                + f"Duration: {storyboard.clip_seconds} seconds. \n"
                + f"{_negatives_text(storyboard.negatives)}"
            )
        else:
            # Clip 2+: Continuity template (Sora 2 guide continuity template)
            # Extract motion continuity hint if present in previous segment
            motion_hint = ""
            if segments and len(segments) >= i - 1:
                prev_segment = segments[i - 2]
                # Look for EXIT motion state marker
                if "EXIT:" in prev_segment.upper() or "EXITING AT" in prev_segment.lower():
                    motion_hint = "Continue with the EXACT velocity, direction, and momentum from the previous clip's ending. "
                elif any(word in prev_segment.lower() for word in ["accelerating", "speed", "mph", "velocity", "momentum"]):
                    motion_hint = "Maintain seamless motion continuity from the previous clip's exit state. "
            
            prompt_text = (
                f"Continue from the previous clip's last frame. {motion_hint}\n"
                + (f"\n{training_preamble}\n\n" if training_preamble else "")
                + f"Maintain: \n"
                + f"  - Camera: {storyboard.camera} \n"
                + f"  - Lighting: {storyboard.lighting} \n"
                + f"  - Characters: {storyboard.characters} \n"
                + f"  - Environment: {storyboard.environment} \n"
                + f"  - Style: {storyboard.global_style} \n"
                + f"Action (segment {i} of {total_clips} ONLY): {segment_action or 'continue to the next part of the story'}. \n"
                + f"Duration: {storyboard.clip_seconds} seconds. \n"
                + f"{_negatives_text(storyboard.negatives)}"
            )
        clips.append(Prompt(index=i, prompt=prompt_text.strip()))
    return PromptsFile(clips=clips, seed=storyboard.seed, global_style=storyboard.global_style)


def pretty_json(data: Dict) -> str:
    return json.dumps(data, indent=2)


def render_prompts_markdown(storyboard: Storyboard, prompts: PromptsFile) -> str:
    """Render prompts into a human-readable Markdown document."""

    def _extract_action(prompt_text: str) -> str:
        t = prompt_text or ""
        if "Action (segment" not in t:
            return ""
        after = t.split("Action (segment", 1)[-1]
        if ") ONLY):" in after:
            after = after.split(") ONLY):", 1)[-1]
        elif "):" in after:
            after = after.split("):", 1)[-1]
        for stop in ("\nDuration:", "\nNegatives:"):
            if stop in after:
                after = after.split(stop, 1)[0]
        return after.strip()

    def _extract_negatives(prompt_text: str) -> List[str]:
        t = prompt_text or ""
        if "\nNegatives:\n" not in t:
            return []
        tail = t.split("\nNegatives:\n", 1)[-1]
        out: List[str] = []
        for line in tail.splitlines():
            s = line.strip()
            if not s:
                break
            if s.startswith("-"):
                out.append(s.lstrip("- ").strip())
        return out

    title = (storyboard.job_id or "Run").strip() if getattr(storyboard, "job_id", None) else "Run"
    goal = (storyboard.goal or "").strip()
    template = (getattr(storyboard, "template", None) or "default").strip()

    lines: List[str] = []
    lines.append(f"# Prompts — {title}")
    lines.append("")
    lines.append(f"- Template: `{template}`")
    lines.append(f"- Aspect: `{storyboard.aspect_ratio}`")
    lines.append(f"- Clip seconds: `{storyboard.clip_seconds}`")
    if prompts.seed is not None:
        lines.append(f"- Seed: `{prompts.seed}`")
    if prompts.global_style:
        lines.append(f"- Global style: {prompts.global_style}")

    if goal:
        lines.append("")
        lines.append("## Goal")
        lines.append("")
        lines.append(goal)

    lines.append("")
    lines.append("## Clips")

    for clip in prompts.clips:
        prompt_text = clip.prompt or ""
        action = _extract_action(prompt_text)
        negatives = _extract_negatives(prompt_text)

        lines.append("")
        lines.append(f"### Clip {clip.index}")
        if action:
            lines.append("")
            lines.append("**Action summary**")
            lines.append("")
            lines.append(action)
        if negatives:
            lines.append("")
            lines.append("**Negatives**")
            lines.append("")
            for n in negatives:
                lines.append(f"- {n}")
        lines.append("")
        lines.append("**Full prompt**")
        lines.append("")
        lines.append("```text")
        lines.append(prompt_text.rstrip())
        lines.append("```")

    lines.append("")
    return "\n".join(lines)
