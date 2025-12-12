"""Interactive Q&A to capture user's creative vision."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Optional, Dict, Any
import os

from schemas import Vision
from prompt_llm import PromptLLMClient


VISION_QUESTIONS = [
    {
        "key": "tone",
        "question": "What emotional tone should the video have?",
        "examples": "intense and dramatic, peaceful and meditative, playful and energetic, melancholic and reflective",
    },
    {
        "key": "style",
        "question": "What visual style are you aiming for?",
        "examples": "gritty handheld realism, polished cinematic, documentary-style, anime-inspired, vintage film",
    },
    {
        "key": "pacing",
        "question": "How should the pacing feel?",
        "examples": "fast-paced action, slow contemplative, steady and rhythmic, building intensity, relaxed flow",
    },
    {
        "key": "key_moment",
        "question": "What's the most important moment to capture?",
        "examples": "the peak of the jump, the character's expression, the environmental reveal, the climactic action",
    },
    {
        "key": "atmosphere",
        "question": "What atmosphere or mood should pervade the video?",
        "examples": "tense anticipation, serene beauty, chaotic energy, mysterious intrigue, triumphant achievement",
    },
]


def conduct_vision_interview(topic: str, skip_interview: bool = False) -> Vision:
    """
    Ask the user 5 questions to capture their creative vision.
    Uses LLM to generate contextual questions and smart defaults.
    
    Args:
        topic: The video topic/goal
        skip_interview: If True, use LLM-generated defaults based on topic
    
    Returns:
        Vision object containing user's creative direction
    """
    print(f"\n{'='*60}")
    print(f"Vision Interview for: {topic}")
    print(f"{'='*60}\n")
    
    # Generate contextual questions and defaults using LLM
    questions_and_defaults = _generate_questions_and_defaults(topic)
    
    if skip_interview:
        print("Using AI-generated default vision settings...\n")
        return Vision(
            tone=questions_and_defaults["tone"]["default"],
            style=questions_and_defaults["style"]["default"],
            pacing=questions_and_defaults["pacing"]["default"],
            key_moment=questions_and_defaults["key_moment"]["default"],
            atmosphere=questions_and_defaults["atmosphere"]["default"],
        )
    
    print("Answer 5 quick questions to guide the creative direction.")
    print("(Press Enter to use the AI-suggested default)\n")
    
    answers = {}
    
    for key in ["tone", "style", "pacing", "key_moment", "atmosphere"]:
        q_data = questions_and_defaults[key]
        print(f"Q: {q_data['question']}")
        print(f"   Examples: {q_data['examples']}")
        
        default = q_data['default']
        user_input = input(f"   Your answer [{default}]: ").strip()
        
        answers[key] = user_input if user_input else default
        print()
    
    vision = Vision(**answers)
    
    print(f"\n{'='*60}")
    print("Vision Summary:")
    print(f"{'='*60}")
    for key, value in vision.to_dict().items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print(f"{'='*60}\n")
    
    return vision


def _generate_questions_and_defaults(topic: str) -> Dict[str, Dict[str, str]]:
    """
    Use LLM to generate contextual questions and smart defaults based on the topic.
    Falls back to static questions if LLM is unavailable.
    
    Args:
        topic: The video topic/goal
    
    Returns:
        Dictionary with keys: tone, style, pacing, key_moment, atmosphere
        Each containing: question, examples, default
    """
    try:
        client = PromptLLMClient()
        
        system = (
            "You are a cinematographer helping users define their creative vision for video generation.\n"
            "Given a video topic, generate 5 contextual questions with relevant examples and a smart default answer.\n\n"
            "The 5 aspects to cover:\n"
            "1. TONE: emotional feeling (dramatic, peaceful, playful, melancholic, etc.)\n"
            "2. STYLE: visual aesthetic (cinematic, documentary, anime, vintage, gritty, etc.)\n"
            "3. PACING: rhythm and speed (fast action, slow contemplative, steady, building, etc.)\n"
            "4. KEY_MOMENT: most important moment to capture (specific to the topic)\n"
            "5. ATMOSPHERE: overall mood (tense, serene, chaotic, mysterious, triumphant, etc.)\n\n"
            "For each aspect:\n"
            "- Write a clear question tailored to the topic\n"
            "- Provide 3-5 relevant examples\n"
            "- Suggest ONE best default that fits the topic\n\n"
            "Output ONLY valid JSON with this structure:\n"
            "{\n"
            '  "tone": {"question": "...", "examples": "...", "default": "..."},\n'
            '  "style": {"question": "...", "examples": "...", "default": "..."},\n'
            '  "pacing": {"question": "...", "examples": "...", "default": "..."},\n'
            '  "key_moment": {"question": "...", "examples": "...", "default": "..."},\n'
            '  "atmosphere": {"question": "...", "examples": "...", "default": "..."}\n'
            "}"
        )
        
        user_prompt = f"Topic: {topic}\n\nGenerate contextual vision interview questions with smart defaults."
        
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
        }
        
        resp = client.session.post(client._chat_url(), headers=client._headers(), json=payload, timeout=30)
        resp.raise_for_status()
        
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Try to extract JSON from markdown fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # Validate structure
        required_keys = ["tone", "style", "pacing", "key_moment", "atmosphere"]
        if all(k in result for k in required_keys):
            for k in required_keys:
                if not all(field in result[k] for field in ["question", "examples", "default"]):
                    raise ValueError(f"Missing fields in {k}")
            return result
        else:
            raise ValueError("Missing required vision keys")
    
    except Exception as e:
        # Fallback to static questions
        print(f"Note: Using default questions (LLM unavailable: {e})\n")
        return _get_static_questions_and_defaults(topic)


def _get_static_questions_and_defaults(topic: str) -> Dict[str, Dict[str, str]]:
    """
    Fallback static questions with basic keyword-based defaults.
    Used when LLM is unavailable.
    """
    topic_lower = topic.lower()
    
    # Basic tone default
    if any(word in topic_lower for word in ["battle", "fight", "race", "intense"]):
        tone_default = "intense and dramatic"
    elif any(word in topic_lower for word in ["peaceful", "calm", "serene", "cooking", "meditate"]):
        tone_default = "warm and inviting"
    elif any(word in topic_lower for word in ["fun", "play", "dance", "celebrate"]):
        tone_default = "playful and energetic"
    else:
        tone_default = "cinematic and engaging"
    
    # Basic style default
    if any(word in topic_lower for word in ["documentary", "real", "authentic"]):
        style_default = "documentary-style realism"
    elif any(word in topic_lower for word in ["anime", "cartoon", "animated"]):
        style_default = "anime-inspired"
    else:
        style_default = "polished cinematic with high detail"
    
    # Basic pacing default
    if any(word in topic_lower for word in ["fast", "quick", "rapid", "race"]):
        pacing_default = "fast-paced action"
    elif any(word in topic_lower for word in ["slow", "calm", "peaceful", "cooking"]):
        pacing_default = "slow and methodical"
    else:
        pacing_default = "steady and rhythmic"
    
    # Key moment based on topic
    if "cooking" in topic_lower or "recipe" in topic_lower:
        key_moment_default = "the moment of plating the finished dish"
    elif any(word in topic_lower for word in ["jump", "flip", "leap"]):
        key_moment_default = "the peak of the aerial trick"
    else:
        key_moment_default = "the main action sequence"
    
    # Atmosphere based on topic
    if "cooking" in topic_lower or "kitchen" in topic_lower:
        atmosphere_default = "cozy and comforting"
    elif any(word in topic_lower for word in ["scenic", "landscape", "nature"]):
        atmosphere_default = "serene beauty"
    else:
        atmosphere_default = "visually compelling"
    
    return {
        "tone": {
            "question": "What emotional tone should the video have?",
            "examples": "intense and dramatic, peaceful and meditative, playful and energetic, warm and inviting",
            "default": tone_default,
        },
        "style": {
            "question": "What visual style are you aiming for?",
            "examples": "gritty handheld realism, polished cinematic, documentary-style, anime-inspired, vintage film",
            "default": style_default,
        },
        "pacing": {
            "question": "How should the pacing feel?",
            "examples": "fast-paced action, slow contemplative, steady and rhythmic, building intensity, relaxed flow",
            "default": pacing_default,
        },
        "key_moment": {
            "question": "What's the most important moment to capture?",
            "examples": "the peak of the action, the character's expression, the environmental reveal, the climactic moment",
            "default": key_moment_default,
        },
        "atmosphere": {
            "question": "What atmosphere or mood should pervade the video?",
            "examples": "tense anticipation, serene beauty, chaotic energy, mysterious intrigue, cozy and comforting",
            "default": atmosphere_default,
        },
    }


def save_vision(vision: Vision, run_dir: Path) -> None:
    """Save vision to vision.json in the run directory."""
    vision_path = run_dir / "vision.json"
    with vision_path.open("w", encoding="utf-8") as f:
        json.dump(vision.to_dict(), f, indent=2)
    print(f"Vision saved to {vision_path}")


def load_vision(run_dir: Path) -> Optional[Vision]:
    """Load vision from vision.json if it exists."""
    vision_path = run_dir / "vision.json"
    if not vision_path.exists():
        return None
    
    with vision_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return Vision.from_dict(data)
