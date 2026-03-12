from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

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

        # Allow longer timeouts for large prompts / slower regions.
        try:
            self.timeout_s = float(os.getenv("AZURE_OPENAI_TEXT_TIMEOUT", "180"))
        except Exception:
            self.timeout_s = 180.0

        try:
            self.max_retries = int(os.getenv("AZURE_OPENAI_TEXT_RETRIES", "2"))
        except Exception:
            self.max_retries = 2

        try:
            self.retry_backoff_s = float(os.getenv("AZURE_OPENAI_TEXT_RETRY_BACKOFF", "2"))
        except Exception:
            self.retry_backoff_s = 2.0

        if not self.mock and not all([self.endpoint, self.api_key, self.deployment]):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_TEXT_MODEL must be set for --auto (or use --mock-llm)."
            )

        self.session = requests.Session()

    def _post_with_retries(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        """POST JSON with basic retry on timeouts / transient connection errors."""
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
                return resp
            except (RequestsTimeout, RequestsConnectionError) as exc:
                last_exc = exc
                attempt += 1
                if attempt > self.max_retries:
                    break
                # Simple linear backoff
                import time

                time.sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last_exc}")

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
        vision: Optional[Dict[str, str]] = None,
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

        # Build vision context if provided
        vision_context = ""
        if vision:
            vision_context = (
                f"\n## Creative Vision (Honor these user preferences):\n"
                f"• TONE: {vision.get('tone', 'cinematic and engaging')}\n"
                f"• STYLE: {vision.get('style', 'polished cinematic')}\n"
                f"• PACING: {vision.get('pacing', 'steady and rhythmic')}\n"
                f"• KEY MOMENT: {vision.get('key_moment', 'the main action')}\n"
                f"• ATMOSPHERE: {vision.get('atmosphere', 'visually compelling')}\n\n"
            )

        system = (
            "You are an expert cinematographer and director creating shot lists for Sora 2. "
            "Think of this as briefing a film crew—be specific, visual, and structured like a professional storyboard.\n\n"
            + vision_context +
            "## PROMPT ANATOMY (Mirror a real film shoot):\n"
            "Write prompts that read like shot lists with clear beats:\n"
            "1. SHOT TYPE: Wide shot, close-up, medium shot, over-the-shoulder, bird's eye, Dutch angle, high/low angle\n"
            "2. CAMERA MOVEMENT: Pan (left/right), tilt (up/down), dolly (in/out), truck (left/right), crane (up/down), "
            "zoom, steadicam follow, tracking shot, arc around, whip pan\n"
            "3. LIGHTING & ATMOSPHERE: Soft golden-hour, high-contrast noir, diffused natural, dramatic backlit, "
            "neon-lit, moody shadows, bright daylight, moonlit, warm candlelight\n"
            "4. COLOR PALETTE: Monochromatic blue, warm amber tones, vibrant saturated colors, muted pastels, "
            "high-contrast black and white, neon cyberpunk, earth tones\n"
            "5. SUBJECT ACTION: Break into beats—e.g. 'takes three steps, pauses, looks upward, then turns'\n"
            "6. SPATIAL DETAILS: Describe what appears, where, and how (scene, props, spatial relationships)\n\n"
            "## VISUAL CUES FOR CONTROL:\n"
            "• NAME A STYLE/ERA EARLY: '70s film noir', '16mm black & white', 'neo-Tokyo cyberpunk'\n"
            "• USE CONCRETE NOUNS: Instead of 'beautiful street', say 'wet asphalt with neon reflections, scattered puddles, glowing signs flickering'\n"
            "• CALL OUT SPECIFICS: 'soft warm backlight', 'diffused shadows', 'cool blue tones with warm highlights', 'shallow depth of field'\n"
            "• GIVE SUBJECT ANCHORS: Unique visual/behavioral details (clothing, posture, accessories, gestures) that help track them\n\n"
            "## MOTION & TIMING (CRITICAL):\n"
            "• ONE clear subject action + ONE camera motion per segment\n"
            "• Use BEATS approach: 'she takes four steps, stops, raises her hand, camera dollies in final frame'\n"
            "• BE EXPLICIT WITH TIMING: 'camera slowly pushes upward over 2 seconds, resolving on her face'\n"
            "• EVERY segment MUST end with EXIT MOTION STATE: velocity, direction, position\n"
            "• NEXT segment continues that EXACT motion seamlessly\n\n"
            "## CONSISTENCY RULES:\n"
            "• LIGHTING MODEL: If scene 1 is dusk, don't switch to midday in scene 2\n"
            "• DESCRIBE LIGHT QUALITY: Hard/soft, directional, sources (window, backlight, practical lamp), temperature (warm/cool)\n"
            "• CHARACTER ANCHORS: Same outfit, same visual details throughout all segments\n"
            "• ENVIRONMENT CONTINUITY: Same location, same weather, same atmospheric conditions\n\n"
            "## CONTENT SAFETY:\n"
            "✅ USE: flowing, gliding, graceful, smooth, elegant, gentle, controlled, drifting, rising, floating, arcing, sweeping\n"
            "❌ NEVER: grinding, slamming, crashing, smashing, violent, aggressive, attacking, dangerous, extreme, impact, collision, injury\n"
            "Emphasize TECHNIQUE, GRACE, and BEAUTY over force or danger.\n\n"
            "## GOOD PROMPT EXAMPLE:\n"
            "'Medium shot of a chrome-plated humanoid robot snowboarder with glowing cyan LED eyes and cobalt-blue accent panels. "
            "Opens with camera tracking steadily from 10 feet behind at matching speed. Golden-hour sunlight creates warm highlights on metallic surfaces "
            "with soft elongated shadows across pristine white powder. The robot flows smoothly down slope in rhythmic carved turns, "
            "polished metal chassis reflecting amber light, loose snow dispersing in graceful arcs around articulated ankle joints. "
            "Camera dollies forward slightly as robot approaches sculpted jump ramp with gentle upward curve. "
            "Warm amber and cool blue color palette. Serene mountain atmosphere with scattered ice crystals catching light. "
            "Robot compresses mechanical legs smoothly, board edge slicing clean line through powder. "
            "EXIT MOTION STATE: 22mph, low crouch, straight downhill trajectory toward jump lip.'\n\n"
            "## BAD PROMPT EXAMPLE:\n"
            "'A snowboarder goes down a mountain and does tricks. Cool lighting. Fast action.'\n"
            "(Too vague—no shot type, no camera movement, no lighting specifics, no color palette, no spatial details, no timing)\n\n"
            "## YOUR TASK:\n"
            "Create {total_clips} sequential segments that flow as ONE continuous shot. Each segment:\n"
            "• Starts by continuing previous EXIT MOTION STATE\n"
            "• Describes shot type, camera movement, lighting, color palette\n"
            "• Breaks action into clear beats with timing\n"
            "• Uses vivid concrete nouns and spatial details\n"
            "• Ends with specific EXIT MOTION STATE\n\n"
            "Fill in creative details the user didn't specify—this is YOUR cinematographer vision.\n"
            "Output ONLY valid JSON. No markdown, no preamble."
        )

        schema = {
            "global_style": "string (visual style/era + color palette + depth of field + film quality, e.g. '70s film noir with monochromatic blue palette, shallow depth of field, 16mm grain')",
            "camera": "string (camera movement type + direction + speed + framing, e.g. 'Steady tracking dolly following 10 feet behind, maintaining centered medium shot, slight upward 10-degree tilt')",
            "lighting": "string (lighting style + quality + temperature + shadow type + sources, e.g. 'Soft golden-hour at 4300K creating warm highlights with diffused shadows from low sun angle, rim-lighting on subject')",
            "characters": "string (shot-list style description: appearance + unique anchors + materials/textures + visible motion details, e.g. 'Sleek humanoid robot, polished chrome chassis with turquoise LED accents on joints, matte black torso plating with brushed texture, articulated fingers with visible servo movement')",
            "environment": "string (spatial layout + terrain type + atmospheric effects + background elements + weather + texture details, e.g. 'Open alpine slope with rolling powder drifts, sculpted jump with smooth edges, evergreen treeline on horizon, airborne snow crystals drifting in golden light, thin ground mist')",
            "negatives": ["string (4-6 items from negative_library verbatim)"],
            "segments": ["string (SHOT LIST FORMAT: Shot type → Camera movement → Subject action in beats with timing → Lighting/color notes → Spatial details → EXIT MOTION STATE with velocity/direction/momentum)"],
        }

        user = {
            "task": "Create a structured cinematic video plan for Sora 2 multi-clip prompting.",
            "topic": topic,
            "constraints": {
                "total_clips": total_clips,
                "clip_seconds": clip_seconds,
                "aspect_ratio": aspect_ratio,
                "seed": seed,
                "negative_library": negative_library,
            },
            "output_schema": schema,
            "CRITICAL_RULES": [
                "═══ FORMATTING ═══",
                "• Return ONLY valid JSON, no markdown fences, no explanatory text",
                "• EXACTLY {total_clips} segments",
                "",
                "═══ CINEMATOGRAPHY APPROACH ═══",
                "Think like a professional cinematographer briefing a crew:",
                "• Each segment is a SHOT in a shot list",
                "• Describe HOW you'd film it (shot type, camera move, lens, lighting setup)",
                "• Give crew enough detail to execute the shot exactly as you envision",
                "• Include timing cues ('over 3 seconds', 'pauses for beat', 'then')",
                "",
                "═══ STORY STRUCTURE ═══",
                "• Split topic into {total_clips} sequential story beats (beginning → middle → end)",
                "• Each segment advances the narrative—NO repeated actions",
                "• Create a story arc even if topic is simple",
                "• Fill in creative details the user didn't specify (this is YOUR vision as director)",
                "",
                "═══ VISUAL CONSISTENCY (CRITICAL) ═══",
                "These MUST remain IDENTICAL across all segments:",
                "• Lighting model (if dusk, stay dusk; don't jump to midday)",
                "• Character appearance (same outfit, same anchors, same visual details)",
                "• Environment (same location, same weather, same atmospheric conditions)",
                "• Color palette (maintain the same dominant tones)",
                "• Visual style/era (70s noir stays 70s noir)",
                "",
                "═══ MOTION CONTINUITY (For seamless transitions) ═══",
                "• EVERY segment MUST end with: EXIT MOTION STATE: [velocity] [direction] [body position]",
                "• NEXT segment MUST start by continuing that EXACT state",
                "• Example: 'EXIT: 25mph, forward lean, descending' → Next: 'Continuing at 25mph in forward lean, descending...'",
                "",
                "═══ SHOT LIST FORMAT FOR SEGMENTS ═══",
                "Each segment should follow cinematographer shot list structure:",
                "1. Shot type (wide/medium/close-up/angle)",
                "2. Camera movement (dolly/pan/tilt/crane + direction + speed)",
                "3. Subject action broken into beats with timing",
                "4. Lighting notes (quality, temperature, shadows, sources)",
                "5. Color palette emphasis",
                "6. Spatial/atmospheric details (props, weather, particles)",
                "7. EXIT MOTION STATE",
                "",
                "═══ DETAIL REQUIREMENTS ═══",
                "global_style: Name style/era early + color palette + technical specs",
                "  Example: '1970s documentary style with warm amber and earth-tone palette, 16mm film grain, shallow depth of field'",
                "",
                "camera: Specific movement type + speed/distance + framing + angle",
                "  Example: 'Smooth tracking dolly following from 12 feet behind at constant speed, maintaining centered medium shot, 10-degree upward tilt for heroic profile'",
                "",
                "lighting: Style + quality + temperature + shadow characteristics + sources",
                "  Example: 'Soft golden-hour light at 4300K creating warm highlights on metallic surfaces, diffused atmosphere, gentle rim-lighting, elongated shadows across snow'",
                "",
                "characters: Physical description + unique visual anchors + materials/colors + motion details",
                "  Example: 'Medium-height humanoid robot with polished silver alloy plating, turquoise LED accents glowing along joints, finely articulated fingers, brushed texture panels catching light, visible servo movement in knee joints'",
                "  Note: Include unique anchors (LED color, texture, accessories) so model can track subject across frames",
                "",
                "environment: Spatial layout + terrain + atmospheric conditions + background + weather + textures",
                "  Example: 'Open alpine slope with gently rolling powder, soft untouched snow drifts, neatly sculpted jump with rounded edges, evergreen trees lining distant horizon, clear sky with faint pastel clouds, airborne snow crystals drifting lazily, thin mist rising from cold surface'",
                "",
                "segments: SHOT LIST style—shot type, camera move, action beats with timing, lighting, spatial details, EXIT STATE",
                "  Use SAFE motion verbs: glides, flows, drifts, rises, rotates, arcs, sweeps, descends, floats",
                "  AVOID: grinds, slams, crashes, strikes, impacts, attacks",
                "",
                "negatives: Pick 4-6 from negative_library that prevent unwanted visual changes",
                "",
                f"═══ YOUR CREATIVE BRIEF ═══",
                f"TOPIC: '{topic}'",
                "Your job: Transform this into a compelling {total_clips}-shot sequence with:",
                "• Professional cinematography (shot types, camera moves, lighting)",
                "• Rich visual details (colors, textures, atmospheric effects)",
                "• Clear story progression (beginning → middle → end)",
                "• Graceful, elegant motion (emphasize technique and beauty)",
                "• Consistent visual world (lighting, style, character, environment)",
            ],
        }

        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, indent=2)},
            ],
        }

        resp = self._post_with_retries(self._chat_url(), payload)
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

    def weave_feedback_into_prompt(self, original_prompt: str, feedback: str, clip_index: int) -> str:
        """Use LLM to integrate feedback naturally into the existing prompt, making it longer and more detailed."""
        if self.mock:
            return f"{original_prompt}\n\nFEEDBACK INTEGRATED: {feedback}"

        system = (
            "You are an expert at enhancing Sora 2 video generation prompts. Your job is to take an existing prompt "
            "and user feedback, then CREATE A NEW, LONGER, MORE DETAILED PROMPT that:\n\n"
            "1. **PRESERVES** all the original scene details, camera work, lighting, characters, and environment\n"
            "2. **WEAVES IN** the feedback constraints naturally throughout the prompt\n"
            "3. **EXPANDS** the prompt with more specific details, action beats, and visual descriptions\n"
            "4. **MAINTAINS** the cinematographer shot-list format\n"
            "5. **ADDS** explicit mentions of the feedback constraints in relevant action segments\n\n"
            "The new prompt should be SIGNIFICANTLY LONGER (aim for 2500-3500 characters) with:\n"
            "- More detailed action beats with explicit timing\n"
            "- Specific mentions of physical interactions (if relevant to feedback)\n"
            "- Additional visual details about materials, textures, lighting quality\n"
            "- Explicit statements about what should NOT happen (based on feedback)\n"
            "- More precise camera movement specifications\n\n"
            "Return ONLY the new enhanced prompt text, no explanations or meta-commentary."
        )

        user_message = (
            f"**ORIGINAL PROMPT (Clip {clip_index}):**\n{original_prompt}\n\n"
            f"**USER FEEDBACK TO INTEGRATE:**\n{feedback}\n\n"
            f"Create a new, significantly longer and more detailed prompt that naturally integrates this feedback "
            f"while preserving all original elements. Make it 2500-3500 characters."
        )

        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        }

        try:
            resp = self.session.post(self._chat_url(), headers=self._headers(), json=body, timeout=120)
            if not resp.ok:
                error_detail = resp.json() if resp.text else "No error details"
                print(f"Warning: LLM weaving failed (status {resp.status_code}): {error_detail}")
                print(f"Falling back to appending feedback instead...")
                return f"{original_prompt}\n\n**CRITICAL FEEDBACK CONSTRAINTS:**\n{feedback}"
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            # If request fails, fall back to appending feedback
            print(f"Warning: LLM weaving failed ({e}), appending feedback instead...")
            return f"{original_prompt}\n\n**CRITICAL FEEDBACK CONSTRAINTS:**\n{feedback}"

        content = data["choices"][0]["message"]["content"].strip()
        
        # Remove any markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        return content


    def generate_training_script(
        self,
        course_block: str,
        total_clips: int,
        clip_seconds: int,
        aspect_ratio: str,
        seed: Optional[int] = None,
        style_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a scene-by-scene training script.

        Output schema (JSON dict):
          {
            "title": str,
            "subtitle": str,
            "host": {"description": str, "voice": str},
            "aesthetic": {"style": str, "typography": str, "motion_graphics": str},
            "scenes": [
              {
                "index": int,
                "duration_seconds": int,
                "narration": str,
                "on_screen_text": {"lines": [str, ...]},
                "visuals": str
              }
            ]
          }
        """

        course_block = (course_block or "").strip()
        if not course_block:
            raise ValueError("course_block is required")

        if self.mock:
            title = "Course Introduction"
            # Best-effort pull of an inline title
            for line in course_block.splitlines():
                if "course title" in line.lower():
                    title = line.split(":", 1)[-1].strip().strip("\"'")
                    break
            scenes: List[Dict[str, Any]] = []
            for i in range(1, total_clips + 1):
                scenes.append(
                    {
                        "index": i,
                        "duration_seconds": clip_seconds,
                        "narration": f"Welcome to {title}. In this short introduction, we preview what you will learn and who the course is for.",
                        "on_screen_text": {"lines": [title, "Instructor-Led Training", f"Scene {i} of {total_clips}"]},
                        "visuals": "Friendly light-brown teddy bear host in a clean studio with subtle manufacturing UI silhouettes and light motion graphics.",
                    }
                )
            return {
                "title": title,
                "subtitle": "Instructor-Led Training",
                "host": {"description": "Light-brown teddy bear host (friendly, non-real)", "voice": "Same narrator voice across all scenes"},
                "aesthetic": {
                    "style": style_hint or "Modern enterprise training aesthetic",
                    "typography": "Clean sans-serif, high contrast",
                    "motion_graphics": "Light motion graphics with labeled tiles and bullets",
                },
                "scenes": scenes,
            }

        system = (
            "You are an instructional designer and motion-graphics director writing a 60-second enterprise training intro script for Sora 2 video generation. "
            "Your output MUST be usable to generate a content-dense intro video with clear on-screen headings and lots of spoken narration (voiceover).\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Output ONLY valid JSON. No markdown. No preamble.\n"
            "- EXACTLY the requested number of scenes.\n"
            "- Every scene must include: narration, on_screen_text.lines (strings), and visuals.\n"
            "- The on_screen_text.lines must be short and readable, and MUST be treated as EXACT text to render on-screen.\n"
            "- Keep a consistent host character across all scenes: a friendly light-brown teddy bear (non-real), presenting to camera in a clean studio.\n"
            "- Keep the narrator voice consistent across all scenes: same tone, same persona, no switching speakers.\n"
            "- Make the narration informational and specific to the course content; avoid generic corporate filler.\n"
            "- Preserve the course title verbatim if present in the input.\n"
            "- Use modern enterprise training styling: clean typography, light motion graphics, subtle UI silhouettes relevant to the domain (but do not invent brand logos).\n"
        )

        schema = {
            "title": "string",
            "subtitle": "string",
            "host": {"description": "string", "voice": "string"},
            "aesthetic": {"style": "string", "typography": "string", "motion_graphics": "string"},
            "scenes": [
                {
                    "index": "int (1-based)",
                    "duration_seconds": "int (match clip_seconds)",
                    "narration": "string (full sentence voiceover; content-dense)",
                    "on_screen_text": {"lines": ["string (exact)"]},
                    "visuals": "string (what we see; motion graphics; host actions)",
                }
            ],
        }

        user = {
            "task": "Convert the provided course block into a 60-second scene script.",
            "inputs": {
                "course_block": course_block,
                "total_scenes": total_clips,
                "scene_seconds": clip_seconds,
                "aspect_ratio": aspect_ratio,
                "seed": seed,
                "style_hint": style_hint or "Modern enterprise training aesthetic with light motion graphics and clean typography",
            },
            "output_schema": schema,
            "scene_guidance": [
                "Scene 1: Title + what the course is about (1-2 sentences) + on-screen title/subtitle",
                "Scene 2-4: What you'll learn (group goals into readable bullets; keep narration explicit)",
                "Scene 5: Who it's for + prerequisites (if provided)",
                "Scene 6: Close with call-to-action (enroll / begin)",
                "If there are fewer/more scenes than 6, distribute the same content proportionally.",
            ],
            "host_constraints": [
                "Host is a light-brown teddy bear, friendly and professional, consistent outfit and facial features.",
                "Host remains on-camera for most scenes; motion graphics appear beside/behind host.",
            ],
            "voice_constraints": [
                "Single narrator voice across all scenes.",
                "Tone: warm, encouraging, clear, enterprise-professional.",
            ],
        }

        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, indent=2)},
            ]
        }

        resp = self._post_with_retries(self._chat_url(), payload)
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

        # Minimal validation
        scenes = obj.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != total_clips:
            raise RuntimeError(f"training script scenes must be a list of length {total_clips}")
        for i, s in enumerate(scenes, start=1):
            if not isinstance(s, dict):
                raise RuntimeError("training script scene must be an object")
            if "narration" not in s or "on_screen_text" not in s or "visuals" not in s:
                raise RuntimeError(f"training script scene {i} missing required fields")
        return obj
