from __future__ import annotations

import json
import math
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


def build_prompts(
    storyboard: Storyboard,
    total_clips: int,
    segments: Optional[List[str]] = None,
) -> PromptsFile:
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
                f"Action (segment 1 of {total_clips}): {segment_action or 'establish the scene and begin the story'}. \n"
                f"Duration: {storyboard.clip_seconds} seconds. \n"
                f"{_negatives_text(storyboard.negatives)}"
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
                f"Maintain: \n"
                f"  - Camera: {storyboard.camera} \n"
                f"  - Lighting: {storyboard.lighting} \n"
                f"  - Characters: {storyboard.characters} \n"
                f"  - Environment: {storyboard.environment} \n"
                f"  - Style: {storyboard.global_style} \n"
                f"Action (segment {i} of {total_clips} ONLY): {segment_action or 'continue to the next part of the story'}. \n"
                f"Duration: {storyboard.clip_seconds} seconds. \n"
                f"{_negatives_text(storyboard.negatives)}"
            )
        clips.append(Prompt(index=i, prompt=prompt_text.strip()))
    return PromptsFile(clips=clips, seed=storyboard.seed, global_style=storyboard.global_style)


def pretty_json(data: Dict) -> str:
    return json.dumps(data, indent=2)
