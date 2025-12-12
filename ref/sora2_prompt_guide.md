
# Sora 2 Prompt Engineering Guide  
### *Reddit‑Validated & Agent‑Ready (for Automated Prompt Builders)*  
Version: 1.0  
Author: ChatGPT (validated via Reddit & OpenAI Cookbook references)

---

## 📌 Overview
This document is designed for **agents**, **LLM chains**, and **chatbots** that need to automatically craft *high‑quality*, *continuous*, and *structured* Sora 2 prompts.  
It includes:

- Reddit‑validated prompting lessons  
- Structured templates  
- Negative prompting patterns  
- JSON/YAML output formats for agents  
- Continuity threading best‑practices  
- A reusable library of motion, style, and camera fragments

---

# 1. Reddit‑Validated Principles

## 1.1 “Specificity beats creativity”  
Reddit consensus: **Vague prompts result in chaotic Sora outputs.**  
→ Always define:  
- Camera  
- Motion  
- Beats  
- Lighting  
- Scene details  
- Character attributes

---

## 1.2 “Sora adds unwanted interpretations unless constrained”
Reddit users report Sora spontaneously creates:
- Random zooms  
- Random cut transitions  
- New objects  
- Unrequested actions  

→ Mitigation: **Add negative constraints** (see templates below).

---

## 1.3 “Sora follows camera instructions very reliably”
Users note that when camera moves are defined, Sora behaves consistently.

→ Always specify: “slow dolly‑in at constant speed”, “tracking left”, etc.

---

## 1.4 “Continuity requires repeating key descriptors”
Reddit experience shows:
- Sora will alter clothes, lighting, angles, if descriptions are omitted.  
→ Repeat these in each clip.

---

## 1.5 “AI‑assisted prompt builders outperform manual writing”
Multiple users mention using ChatGPT / Claude to refine prompts.  
→ This guide is optimized for **agent‑generation**.

---

# 2. Agent Prompt‑Building Workflow

```
1. Ask structured questions (setting, characters, mood, style).
2. Generate Clip 1 prompt.
3. For each additional clip:
      - Include continuity phrasing
      - Repeat scene descriptors
      - Insert negative constraints
4. Output prompts as JSON array.
```

---

# 3. Required Elements for Every Clip
Agents must enforce:

- **Scene description**
- **Characters + attire**
- **Camera movement**
- **Action beats**
- **Lighting**
- **Visual style**
- **Negative constraints**
- **Duration**

These elements appear in all templates below.

---

# 4. Prompt Fragment Library (Reusable Blocks)

## 4.1 Camera Motions
```
slow dolly-in
slow dolly-out
steady tracking left
steady tracking right
static tripod shot
orbiting clockwise
orbiting counterclockwise
crane shot descending
gentle push-in at constant rate
```

## 4.2 Lighting/Mood
```
golden hour warm glow
neon cyberpunk ambient light
overcast diffused gray light
soft cinematic rim light
moody chiaroscuro contrast
```

## 4.3 Negative Constraints
(derived from Reddit user complaints)

```
No sudden zooms.
No random cuts or transitions.
Do not change character outfits.
Do not change the environment.
Do not add new characters.
Keep full subject in frame.
No fisheye or distorted lenses.
No jitter or abrupt camera motion.
```

---

# 5. Official Prompt Structure for Agents

## 5.1 Clip 1 Template (YAML)

```yaml
clip_1:
  scene: "<describe environment, lighting, mood>"
  characters: "<physical description + attire>"
  action: "<what happens from t=0 to t=15>"
  camera: "<define motion precisely>"
  style: "<cinematic, anime, photorealistic, etc>"
  negatives:
    - "No sudden zooms"
    - "No random cuts"
    - "No changes to character appearance"
  duration: "15 seconds"
```

---

## 5.2 Continuity Clip Template (YAML)

```yaml
clip_n:
  continuation: "Continue the same shot from the previous frame/image."
  maintain:
    camera: "<repeat motion>"
    lighting: "<repeat lighting>"
    characters: "<repeat full description>"
    environment: "<repeat>"
  action: "<next beat of the story>"
  style: "<match prior clip>"
  negatives:
    - "Maintain exact style continuity"
    - "No new camera angles"
    - "No added characters"
  duration: "15 seconds"
```

---

# 6. JSON Output Format for Agents

```json
{
  "seed": 12345,
  "global_style": "cinematic warm tones, shallow depth of field",
  "clips": [
    {
      "index": 1,
      "prompt": "..."
    },
    {
      "index": 2,
      "prompt": "..."
    }
  ]
}
```

---

# 7. Example Full Prompt (Reddit‑Validated)

## Clip 1 Example
```
Scene: A neon-lit Tokyo side street at night. Soft rain hits the pavement, creating glowing reflections. 
Characters: A courier wearing a blue tech-jacket with LED stripes and a small visor helmet. 
Camera: Slow tracking shot from left to right at a constant pace. 
Action: Courier glides on an electric board through the alley, dodging puddles. 
Style: Cyberpunk cinematic, shallow depth-of-field, moody neon color palette. 
Negatives: No new characters, no sudden zooms, no camera shaking, do not change outfit. 
Duration: 15s.
```

## Clip 2 Example
```
Continue the same tracking motion from the previous frame. 
Maintain: same neon lighting, same speed, same character attire, same environment details. 
Action: Courier reaches a wider street; large holograms appear overhead, flickering. 
Style: Same cyberpunk cinematic palette. 
Negatives: No perspective change, no added cuts, no new characters. 
Duration: 15s.
```

---

# 8. Agent‑Ready Prompt Builder Specification

```yaml
agent_behavior:
  name: "sora_prompt_builder"
  goals:
    - "Generate structured Sora 2 prompts."
    - "Maintain continuity for multi-clip videos."
    - "Ask clarifying questions to users."
    - "Output standardized YAML/JSON prompts."
  rules:
    - "Always include camera movement."
    - "Always include negative constraints."
    - "Always repeat character/environment descriptors for continuity."
    - "Always produce durations."
  output_formats:
    - "yaml"
    - "json"
    - "plain text prompts"
```

---

# 9. Final Notes
This guide reflects:
- Reddit consensus on Sora’s weaknesses  
- Prompt structures that reduce randomness  
- Best practices for multi‑clip continuity  
- Practical tools for agents that need consistency and automation  

This document can be safely embedded into your agent’s memory or RAG store.

---

*End of Document*
