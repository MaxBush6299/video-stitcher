from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from utils import normalize_azure_openai_endpoint


@dataclass
class PromptPlan:
    global_style: str
    camera: str
    lighting: str
    characters: str
    environment: str
    negatives: List[str]
    segments: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_style": self.global_style,
            "camera": self.camera,
            "lighting": self.lighting,
            "characters": self.characters,
            "environment": self.environment,
            "negatives": self.negatives,
            "segments": self.segments,
        }


class PromptLLMClient:
    """Generates a structured storyboard/prompt plan from a topic + time.

    Uses Azure OpenAI Chat Completions via a deployment name.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        mock: bool = False,
    ) -> None:
        import os

        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_TEXT_MODEL")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_TEXT_API_VERSION", "2024-10-01-preview")
        self.mock = mock

        if not self.mock and not all([self.endpoint, self.api_key, self.deployment]):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_TEXT_MODEL must be set for --auto (or use --mock-llm)."
            )

        self.session = requests.Session()

    def _chat_url(self) -> str:
        if not self.endpoint or not self.deployment:
            raise ValueError("Missing endpoint/deployment")
        base = normalize_azure_openai_endpoint(self.endpoint)
        return f"{base}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key or "",
        }

    def generate_plan(
        self,
        topic: str,
        total_clips: int,
        clip_seconds: int,
        aspect_ratio: str,
        negative_library: List[str],
        seed: Optional[int] = None,
    ) -> PromptPlan:
        if self.mock:
            segments = [
                f"Segment {i}/{total_clips}: Part {i} of the story '{topic}'" for i in range(1, total_clips + 1)
            ]
            return PromptPlan(
                global_style="cinematic, high detail, shallow depth of field",
                camera="slow dolly-in at constant speed",
                lighting="golden hour warm glow",
                characters="One main subject with consistent outfit, no changes",
                environment="A coherent setting that matches the topic",
                negatives=negative_library[:3],
                segments=segments,
            )

        system = (
            "You are a Sora 2 prompt planner following Reddit-validated best practices. "
            "Given a topic and constraints, output ONLY valid JSON matching the schema. "
            "\n\nKey principles:\n"
            "- Specificity beats creativity: define camera, motion, lighting, scene details explicitly.\n"
            "- Sora adds unwanted interpretations unless constrained: use negative constraints.\n"
            "- Continuity requires repeating key descriptors: camera, lighting, character, environment must be identical across segments.\n"
            "- Camera instructions are reliable: specify exact motion (e.g., 'slow dolly-in at constant speed').\n"
            "\nTask: Split the topic into sequential story segments (beginning→middle→end) that flow as one continuous shot. "
            "Each segment describes what happens in that portion of the timeline only."
        )

        schema = {
            "global_style": "string",
            "camera": "string",
            "lighting": "string",
            "characters": "string",
            "environment": "string",
            "negatives": ["string"],
            "segments": ["string"],
        }

        user = {
            "task": "Create a structured video plan for Sora-style clip prompting.",
            "topic": topic,
            "constraints": {
                "total_clips": total_clips,
                "clip_seconds": clip_seconds,
                "aspect_ratio": aspect_ratio,
                "seed": seed,
                "negative_library": negative_library,
                "negatives_pick_count": "Pick exactly 4 to 6 items from negative_library verbatim. These prevent unwanted camera changes, character outfit changes, environment shifts, and random cuts.",
            },
            "output_schema": schema,
            "rules": [
                "Return JSON only (no markdown).",
                "segments must have exactly total_clips entries.",
                "SPLIT the topic into total_clips sequential parts (beginning→middle→end).",
                "Each segment describes ONLY the action for that portion of the timeline.",
                "Segments must be SEQUENTIAL and NON-OVERLAPPING: segment 2 starts where segment 1 ends.",
                "Do NOT repeat actions across segments (e.g., if segment 1 = 'skis through trees', segment 2 = 'approaches jump', NOT 'skis through trees again').",
                "ALL segments must share IDENTICAL: character (full appearance/outfit), environment (setting details), camera (exact motion description), lighting (mood/quality), and style.",
                "Camera: Use precise motion descriptions (e.g., 'slow dolly-in at constant speed', 'steady tracking left'). NO vague terms.",
                "Lighting: Be specific (e.g., 'golden hour warm glow', 'overcast diffused gray light').",
                "Negatives: Pick 4-6 from negative_library that prevent Sora from adding unwanted zooms, cuts, character changes, or environment shifts.",
                "Visual continuity: Each segment assumes the next clip starts on the EXACT last frame of the prior clip and continues motion immediately (no pause, no reset).",
                "Write segments as concrete, filmable actions with visible motion/change.",
                "Example: 'bear skiing...' → segment 1: 'bear carves downhill through dense pine trees'; segment 2: 'bear accelerates toward large snow jump and launches upward'; segment 3: 'bear rotates mid-air and crashes face-first into snow with skis above head'.",
            ],
        }

        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
        }

        resp = self.session.post(self._chat_url(), headers=self._headers(), json=payload, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = (resp.text or "").strip()
            if len(body) > 2000:
                body = body[:2000] + "..."
            raise RuntimeError(f"LLM request failed: HTTP {resp.status_code} for {resp.url}; body={body}") from exc
        data = resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        try:
            obj = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM did not return valid JSON: {exc}; content={content[:500]}")

        return self._validate_plan(obj, total_clips, negative_library)

    def _validate_plan(self, obj: Dict[str, Any], total_clips: int, negative_library: List[str]) -> PromptPlan:
        required = ["global_style", "camera", "lighting", "characters", "environment", "negatives", "segments"]
        for k in required:
            if k not in obj:
                raise RuntimeError(f"LLM output missing key: {k}")

        segments = obj["segments"]
        if not isinstance(segments, list) or len(segments) != total_clips:
            raise RuntimeError(f"segments must be a list of length {total_clips}")

        # Light guardrail: avoid exact duplicate segments (a common failure mode).
        # We don't hard-fail on near-duplicates to keep the demo resilient.
        normalized = [str(s).strip().lower() for s in segments]
        if len(set(normalized)) != len(normalized):
            deduped: List[str] = []
            seen: set[str] = set()
            for i, s in enumerate(segments, start=1):
                key = str(s).strip().lower()
                if key in seen:
                    deduped.append(f"Segment {i}: continuation from prior segment; do not repeat previous action.")
                else:
                    seen.add(key)
                    deduped.append(str(s))
            segments = deduped

        negatives = obj["negatives"]
        if not isinstance(negatives, list) or not (3 <= len(negatives) <= 6):
            raise RuntimeError("negatives must be a list of 3 to 6 items")

        normalized_negatives: List[str] = []
        allowed = set(negative_library)
        for n in negatives:
            if isinstance(n, str) and n in allowed:
                normalized_negatives.append(n)
        if len(normalized_negatives) < 1:
            normalized_negatives = negative_library[:3]

        return PromptPlan(
            global_style=str(obj["global_style"]).strip(),
            camera=str(obj["camera"]).strip(),
            lighting=str(obj["lighting"]).strip(),
            characters=str(obj["characters"]).strip(),
            environment=str(obj["environment"]).strip(),
            negatives=normalized_negatives,
            segments=[str(s).strip() for s in segments],
        )
