# Video Generator with AI (Sora 2)

This tool helps you create professional-quality videos using AI. Just describe what you want to see, answer a few questions about the style, and the AI will generate the video for you—all automatically stitched together into one smooth final video.

**What you can create:**
- Action scenes (sports, adventures, dynamic movement)
- Scenic shots (nature, architecture, atmospheric moments)
- Character-focused videos (robots, animals, people in creative scenarios)
- Instructional training videos (course intros, module overviews, explainer sequences)
- Anything you can imagine and describe!

## What You Need Before Starting

**1. Software Requirements:**
- Python 3.10 or newer ([Download Python](https://www.python.org/downloads/))
- FFmpeg video tool ([Download FFmpeg](https://ffmpeg.org/download.html))
  - After installing, make sure it's accessible from command line

**2. Azure OpenAI Access:**
You'll need these credentials from your Azure account:
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI service URL
- `AZURE_OPENAI_API_KEY` - Your access key
- `AZURE_OPENAI_VIDEO_MODEL` - The name of your Sora 2 video deployment
- `AZURE_OPENAI_TEXT_MODEL` - The name of your GPT chat deployment (for prompt generation)

Don't have these? Ask your Azure administrator or [set up Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource).

## Setup (One-Time)

**Step 1: Download the code**
- Download this project and extract it to a folder on your computer

**Step 2: Install Python dependencies**
Open a terminal/command prompt in the project folder and run:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Step 3: Set up your credentials**
Create environment variables with your Azure credentials:
- Windows: Search "Environment Variables" in Start Menu
- Mac/Linux: Add to `~/.bashrc` or `~/.zshrc`

Required variables:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_VIDEO_MODEL=your-sora-deployment-name
AZURE_OPENAI_TEXT_MODEL=your-gpt-deployment-name
```

## How to Create a Video (Simple Guide)

### The 5-Step Process

**Step 1: Describe what you want**
```bash
python cli.py new --goal "a robot snowboarder doing a backflip" --total-sec 45 --clip-sec 15 --auto --interview
```
- Replace the text in quotes with your own idea
- `--total-sec 45` means 45 seconds total video length
- `--clip-sec 15` means each clip is 15 seconds (it makes 3 clips automatically)
- `--auto` tells the AI to create the prompts for you
- `--interview` asks you 5 simple questions to understand your vision

**What the interview asks:**
1. **Tone** - What emotion? (exciting, peaceful, dramatic, playful)
2. **Style** - What look? (cinematic, realistic, anime-style, documentary)
3. **Pacing** - How fast? (fast action, slow and smooth, steady rhythm)
4. **Key moment** - What's most important? (the peak of action, facial expression, environment)
5. **Atmosphere** - What mood? (tense, serene, energetic, mysterious)

Just type your answer or press Enter to use the suggested default!

**Step 2: Preview what will be created**
```bash
python cli.py preview
```
This shows you the plan before generating expensive video. Review it to make sure it looks right!

**Step 3: Generate the video**
```bash
python cli.py generate
```
This sends your prompts to Azure's Sora 2 AI to create the actual video clips. **This takes several minutes per clip** (be patient!). The tool automatically saves progress, so if something fails, it can continue where it left off.

Optional: Seed clip 1 with a reference image (logo-safe, no human faces):
```bash
python cli.py generate --reference-image path/to/reference.png
```
The tool will auto-resize and crop the image to match the target video size and use it as the first frame.

**Step 4: Combine clips into final video**
```bash
python cli.py stitch
```
This smoothly blends all clips together with crossfade transitions (like a professional video editor).

**Step 5: Watch your video!**
```bash
python cli.py open
```
This opens your final video. Share it, use it in presentations, or just enjoy what you created!

### If Something Goes Wrong

**A clip failed to generate?**
```bash
# Regenerate just the failed clip (e.g., clip 2 and 3)
python cli.py generate --regenerate "2,3"
```

**Video doesn't look right? Give AI-powered feedback!**
```bash
# The AI will update prompts based on your feedback
python cli.py feedback "The character needs to hold the tools properly with both hands"

# Or be more specific about what you want
python cli.py feedback "Add more dramatic lighting from the left side and increase color saturation"
python cli.py feedback "The robot should move more smoothly and gracefully, not jerky"
python cli.py feedback "Show the character's face more clearly in every shot"

# Preview the updated prompts
python cli.py preview

# If you like the changes, regenerate
python cli.py generate
```

**Want to change the style without regenerating?**
```bash
# Answer the 5 vision questions again
python cli.py refine --interview

# Preview the updated prompts
python cli.py preview

# If you like it, generate the clips
python cli.py generate
```

**Made a mistake and want to start over?**
Just run step 1 again with a new description!

### Customizing Your Video

**Change video length and format:**
```bash
python cli.py new --goal "your idea here" --total-sec 60 --clip-sec 20 --aspect 16:9 --auto --interview
```
- `--total-sec 60` = 1 minute total video
- `--clip-sec 20` = each clip is 20 seconds (makes 3 clips)
- `--aspect 16:9` = widescreen format (also supports `1:1` for square, `9:16` for vertical)

Note: Sora video durations are currently supported in fixed increments. If you request `--clip-sec 10`, the backend may produce 8s or 12s clips. For a clean “~60s” run today, a reliable option is `--total-sec 60 --clip-sec 12` (5 clips).

## Make Instructional Training Videos (Course Intros)

This workflow is great for instructor-led training intros, module overviews, and internal enablement videos.

### Example: 60-second course intro (avatar presenter)

Run this (recommended for ~60s: 5 clips × 12s):
```bash
python cli.py new --template training-intro --goal "Create a one-minute instructor-led training course introduction video for: 'ILT Course Title: M3 Cloud: Food and Beverage Foundation – Instructor-Led Training' (Learning Level: Intermediate). Use a non-real, stylized 3D avatar presenter (no real people/faces) speaking to camera in a clean studio with subtle UI motion-graphics. Training description: This instructor-led course provides a foundational understanding of the M3 Cloud Food and Beverage solution. It introduces participants to system navigation, key functionalities, and industry-specific processes relevant to food and beverage operations. Learners will explore core modules, typical workflows, and how the solution supports business requirements across planning, procurement, production, inventory, distribution, order fulfillment, finance, warehouse interaction, and product quality and safety. Essential terminology, system architecture, and integration points are covered, with practical examples and real-world scenarios. The workbook emphasizes hands-on exercises and knowledge checks to reinforce understanding and prepare users for more specialized training. Goals to cover across the sequence: (1) premise of the course, (2) demand & supply planning, (3) procurement, (4) production-to-inventory, (5) distribution orders, (6) order fulfillment, (7) finance areas, (8) warehouse inventory interaction, (9) product quality & safety, (10) additional food & beverage processes. Prerequisites to mention briefly: M3 Cloud: Infor OS with M3 Overview; M3 Cloud: Infor OS Workspaces with M3 Overview; M3 Cloud: H5 Overview; M3 Cloud: H5 Overview – Advanced Topics. Modality: instructor-led training. Tone: professional, clear, encouraging. Pacing: brisk, one idea per segment with readable on-screen headings." --total-sec 60 --clip-sec 12 --aspect 16:9 --auto --interview

python cli.py preview
python cli.py generate
python cli.py stitch
python cli.py open
```

Tip: If you have an approved brand-safe reference image (e.g., background plate, UI style frame, non-logo illustration), you can seed clip 1:
```bash
python cli.py generate --reference-image path/to/reference.png
```

### Scripted training intro (recommended for training pages)

If you have a big course info block (title, description, goals, prerequisites), you can generate a more informational, scene-by-scene script with **exact on-screen text** and a friendly host.

This mode uses a friendly light-brown teddy bear host (no real people) and enforces a single consistent narrator voice in every clip.

```bash
python cli.py new --template training-script --total-sec 60 --clip-sec 12 --aspect 16:9 --goal "ILT Course Title: ...\nLearning Level: ...\n...Goals: ...\nPrerequisites: ...\nModality: Instructor-led training"

python cli.py preview
python cli.py generate
python cli.py stitch
python cli.py open
```

The generated run will include a `script.json` file alongside `prompts.json` so you can review/edit the exact narration + on-screen text per scene.

## Remix (Iterate Without Full Regeneration)

If a clip is close but needs a small change, you can remix a completed video. Remix works best for a single, well-defined adjustment (e.g., palette shift, add a title card, tighten the camera move).

This CLI stores each clip's `video_id` in `run_state.json` as it generates.

Remix the most recent clip in the latest run:
```bash
python cli.py remix "Shift the color palette to teal, sand, and rust, with a warm backlight."
```

Or remix a specific video id:
```bash
python cli.py remix "Make the on-screen title larger and increase contrast." --video-id video_abc123
```

Reference: https://platform.openai.com/docs/guides/video-generation#remix-completed-videos

**Want consistent randomness?**
```bash
python cli.py new --goal "your idea" --seed 42 --auto --interview
```
The `--seed` number ensures you get the same AI interpretation if you run it again.

**Refine specific parts without starting over:**
```bash
# Change just the lighting across all clips
python cli.py refine --field lighting

# Improve just clip 2 based on what you didn't like
python cli.py refine --segment 2

# Then preview before regenerating
python cli.py preview
```

**Work with a previous video project:**
```bash
# Your videos are saved in runs/ folder with timestamps
# To work on a specific one:
python cli.py preview --run runs/2025-12-12T16-45-05Z-fb858a
python cli.py generate --run runs/2025-12-12T16-45-05Z-fb858a
python cli.py stitch --run runs/2025-12-12T16-45-05Z-fb858a
```

**Clean up extra files:**
```bash
python cli.py clean
```
Removes intermediate files but keeps your final video.

## Understanding How It Works

**Behind the scenes:**
1. **AI Prompt Generation** - The GPT model creates detailed descriptions for each clip based on your idea
2. **Vision Guidance** - Your answers to the 5 questions guide the AI's creative choices
3. **Motion Continuity** - The AI automatically ensures smooth movement between clips (no jarring jumps)
4. **Video Generation** - Azure's Sora 2 creates actual video from the prompts
5. **Crossfade Stitching** - Clips blend together with 1.5-second transitions for professional results

**Why multiple clips?**
- Sora 2 works best with shorter clips (10-20 seconds each)
- Multiple clips give you more control and recovery options
- Crossfade blending makes them feel like one continuous video

**What gets saved:**
All your work is saved in `runs/` folder with a timestamp:
```
runs/2025-12-12T16-45-05Z-fb858a/
  storyboard.json         # The overall plan and visual style
  prompts.json            # Detailed instructions for each clip
  vision.json             # Your creative vision answers
  clip_01.mp4, clip_02.mp4, clip_03.mp4  # Individual video clips
  final_video.mp4         # Your finished video!
  logs.txt                # What happened during generation
```

You can always come back to any project later!

## Tips for Great Results

**✅ DO:**
- Be specific in your descriptions ("a red sports car" vs just "a car")
- Use the interview feature to guide the style and mood
- Preview before generating to catch issues early
- Start with shorter videos (30-45 seconds) while learning
- Describe safe, graceful actions (flowing, gliding, smooth movements)
- Use non-human subjects for action scenes when possible (robots, vehicles, animals)

**❌ AVOID:**
- Vague descriptions ("something cool")
- Extremely complex scenes with too many elements
- Skipping the preview step
- Very long individual clips (keep under 20 seconds per clip)
- Violent or dangerous-sounding language (aggressive, slamming, crashing)
- Human characters in intense action sequences (may trigger content filters)

**Common issues and fixes:**

**"Content moderation blocked my video"**
- Try softer language: "flowing" instead of "grinding", "gliding" instead of "slamming"
- Use a robot or non-human character instead of people in action scenes
- Run `refine --interview` to update the tone to something calmer

**"Generation is taking forever"**
- This is normal! Sora 2 can take 5-15 minutes per clip
- The tool saves progress automatically
- You can safely close and restart if needed

**"I don't like how one clip looks"**
- Regenerate just that clip: `python cli.py generate --regenerate "2"`
- Or refine the prompt for that segment: `python cli.py refine --segment 2`

**"Clips don't blend smoothly"**
- The AI handles motion continuity automatically
- Crossfade transitions help blend clips seamlessly
- If still jarring, try regenerating the problematic clips

## Quick Reference: All Commands

```bash
# Create new video project
python cli.py new --goal "your idea" --auto --interview

# See what will be created
python cli.py preview

# Generate the video clips
python cli.py generate

# Regenerate specific clips if needed
python cli.py generate --regenerate "2,3"

# Give feedback to improve the prompts (RECOMMENDED!)
python cli.py feedback "your specific feedback here"

# Update creative direction without regenerating
python cli.py refine --interview

# Update a specific aspect
python cli.py refine --field lighting

# Improve one clip with feedback
python cli.py refine --segment 2

# Combine clips into final video
python cli.py stitch

# Open and watch your video
python cli.py open

# Clean up intermediate files
python cli.py clean

# Work on a previous project
python cli.py preview --run runs/2025-12-12T16-45-05Z-fb858a
```

## Need Help?

**Error messages:** Check `runs/<your-project>/logs.txt` for detailed information

**Questions:** Make sure your Azure credentials are set correctly in environment variables

**Examples to try:**
- "a drone flying over autumn forest"
- "a robot chef preparing ramen"
- "ocean waves at golden hour"
- "northern lights over snowy mountains"
- "a vintage car cruising coastal highway"

---

**Ready to create?** Start with step 1 of the 5-step process above!
