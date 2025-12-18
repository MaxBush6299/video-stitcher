from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class Vision:
    """User's creative vision captured through Q&A"""
    tone: str  # e.g., "intense and dramatic" or "peaceful and meditative"
    style: str  # e.g., "gritty handheld realism" or "polished cinematic"
    pacing: str  # e.g., "fast-paced action" or "slow contemplative"
    key_moment: str  # The most important moment to capture
    atmosphere: str  # Overall mood/feeling

    def to_dict(self) -> Dict:
        return {
            "tone": self.tone,
            "style": self.style,
            "pacing": self.pacing,
            "key_moment": self.key_moment,
            "atmosphere": self.atmosphere,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Vision":
        return cls(
            tone=data["tone"],
            style=data["style"],
            pacing=data["pacing"],
            key_moment=data["key_moment"],
            atmosphere=data["atmosphere"],
        )


@dataclass
class Storyboard:
    version: str
    job_id: str
    total_seconds: int
    clip_seconds: int
    aspect_ratio: str
    seed: Optional[int]
    global_style: str
    camera: str
    lighting: str
    negatives: List[str]
    goal: str
    characters: str
    environment: str
    template: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "job_id": self.job_id,
            "total_seconds": self.total_seconds,
            "clip_seconds": self.clip_seconds,
            "aspect_ratio": self.aspect_ratio,
            "seed": self.seed,
            "global_style": self.global_style,
            "camera": self.camera,
            "lighting": self.lighting,
            "negatives": self.negatives,
            "goal": self.goal,
            "characters": self.characters,
            "environment": self.environment,
            "template": self.template,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Storyboard":
        return cls(
            version=data.get("version", "1.0"),
            job_id=data["job_id"],
            total_seconds=data["total_seconds"],
            clip_seconds=data["clip_seconds"],
            aspect_ratio=data["aspect_ratio"],
            seed=data.get("seed"),
            global_style=data["global_style"],
            camera=data["camera"],
            lighting=data["lighting"],
            negatives=data.get("negatives", []),
            goal=data.get("goal", ""),
            characters=data.get("characters", ""),
            environment=data.get("environment", ""),
            template=data.get("template"),
        )


@dataclass
class Prompt:
    index: int
    prompt: str

    def to_dict(self) -> Dict:
        return {"index": self.index, "prompt": self.prompt}

    @classmethod
    def from_dict(cls, data: Dict) -> "Prompt":
        return cls(index=data["index"], prompt=data["prompt"])


@dataclass
class PromptsFile:
    clips: List[Prompt]
    seed: Optional[int] = None
    global_style: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "global_style": self.global_style,
            "clips": [clip.to_dict() for clip in self.clips],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PromptsFile":
        clips = [Prompt.from_dict(c) for c in data.get("clips", [])]
        return cls(clips=clips, seed=data.get("seed"), global_style=data.get("global_style"))


@dataclass
class ClipArtifact:
    clip: str
    last_frame: str
    video_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {"clip": self.clip, "last_frame": self.last_frame, "video_id": self.video_id}

    @classmethod
    def from_dict(cls, data: Dict) -> "ClipArtifact":
        return cls(clip=data["clip"], last_frame=data["last_frame"], video_id=data.get("video_id"))


@dataclass
class RunState:
    status: str
    current_clip: int
    total_clips: int
    artifacts: Dict[str, ClipArtifact] = field(default_factory=dict)
    reference_image: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "current_clip": self.current_clip,
            "total_clips": self.total_clips,
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "reference_image": self.reference_image,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RunState":
        artifacts = {k: ClipArtifact.from_dict(v) for k, v in data.get("artifacts", {}).items()}
        return cls(
            status=data.get("status", "new"),
            current_clip=data.get("current_clip", 1),
            total_clips=data.get("total_clips", len(artifacts)),
            artifacts=artifacts,
            reference_image=data.get("reference_image"),
        )

    @classmethod
    def load(cls, path: Path) -> "RunState":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
