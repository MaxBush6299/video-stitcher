from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

import typer

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from ffmpeg_bin import ffmpeg_exe
from schemas import ClipArtifact, PromptsFile, RunState, Storyboard
from prompt_llm import PromptLLMClient
from stitcher import StitchError, concat_videos
from utils import (
    NEGATIVE_LIBRARY,
    append_log,
    build_prompts,
    build_storyboard,
    build_training_script_fallback,
    ensure_clip_counts,
    latest_run_dir,
    load_json,
    make_job_id,
    parse_size,
    pretty_json,
    prepare_reference_image,
    render_prompts_markdown,
    runs_root,
    save_json,
    training_script_to_segments,
    training_script_to_srt,
    training_script_voiceover_text,
)
from video_client import VideoClient

app = typer.Typer(
    help="Single-machine demo CLI for sequential Sora-style video generation.",
    pretty_exceptions_show_locals=False,
)


def _prompt_list(prompt: str, default: Optional[str] = None) -> str:
    return typer.prompt(prompt, default=default or "")


def _pick_negatives(default: Optional[List[str]] = None) -> List[str]:
    typer.echo("Choose negatives (comma-separated indices, empty for defaults):")
    for idx, item in enumerate(NEGATIVE_LIBRARY, start=1):
        typer.echo(f"  {idx}) {item}")
    selection = typer.prompt("Negatives", default="")
    if not selection.strip():
        return default or [NEGATIVE_LIBRARY[0], NEGATIVE_LIBRARY[1], NEGATIVE_LIBRARY[3]]
    picks: List[str] = []
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            i = int(token)
            if 1 <= i <= len(NEGATIVE_LIBRARY):
                picks.append(NEGATIVE_LIBRARY[i - 1])
        except ValueError:
            continue
    return picks or (default or [])


def _resolve_run_dir(run: Optional[Path]) -> Path:
    root = runs_root()
    if run:
        if not run.exists():
            raise typer.BadParameter(f"Run directory not found: {run}")
        return run
    latest = latest_run_dir(root)
    if not latest:
        raise typer.BadParameter("No runs found. Create one with `demo new`. ")
    return latest


def _load_files(run_dir: Path) -> tuple[Storyboard, PromptsFile, RunState]:
    storyboard = Storyboard.from_dict(load_json(run_dir / "storyboard.json"))
    prompts = PromptsFile.from_dict(load_json(run_dir / "prompts.json"))
    state = RunState.load(run_dir / "run_state.json")
    return storyboard, prompts, state


def _extract_last_frame(clip_path: Path, output_path: Path) -> None:
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-sseof",
        "-0.5",
        "-i",
        str(clip_path),
        "-vframes",
        "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Failed to extract last frame")


@app.command()
def new(
    project: Optional[str] = typer.Option(None, help="Project or working title."),
    goal: Optional[str] = typer.Option(None, help="Goal/intention of the video."),
    total_sec: int = typer.Option(60, help="Total duration in seconds."),
    clip_sec: int = typer.Option(15, help="Clip length in seconds."),
    aspect: str = typer.Option("16:9", help="Aspect ratio, e.g., 16:9 or 9:16."),
    auto: bool = typer.Option(False, help="Use LLM to generate storyboard/prompts from topic + time."),
    mock_llm: bool = typer.Option(False, help="Mock the LLM planner (offline demo)."),
    interview: bool = typer.Option(False, help="Interactive Q&A to capture creative vision."),
    template: str = typer.Option(
        "default",
        help="Prompt template to use: default | training-intro | training-script | auto (auto-detect training from goal text).",
    ),
    style: Optional[str] = typer.Option(None, help="Global style."),
    lighting: Optional[str] = typer.Option(None, help="Lighting descriptor."),
    camera: Optional[str] = typer.Option(None, help="Camera motion descriptor."),
    characters: Optional[str] = typer.Option(None, help="Characters/attire."),
    environment: Optional[str] = typer.Option(None, help="Environment/setting."),
    seed: Optional[int] = typer.Option(None, help="Seed for reproducibility."),
) -> None:
    """Create a new run with storyboard and prompts."""

    goal = goal or _prompt_list("Topic / goal")
    aspect = aspect or _prompt_list("Aspect ratio", "16:9")

    template_norm = (template or "default").strip().lower()
    if template_norm not in {"default", "training-intro", "training-script", "auto"}:
        raise typer.BadParameter("--template must be one of: default, training-intro, training-script, auto")

    total_clips, normalized_total = ensure_clip_counts(total_sec, clip_sec)

    # Vision Q&A
    vision_obj = None
    if auto and interview:
        from vision_qa import conduct_vision_interview
        vision_obj = conduct_vision_interview(goal, skip_interview=False)
    elif auto:
        # Use defaults
        from vision_qa import conduct_vision_interview
        vision_obj = conduct_vision_interview(goal, skip_interview=True)

    segments = None
    training_script = None
    if auto:
        planner = PromptLLMClient(mock=mock_llm)
        vision_dict = vision_obj.to_dict() if vision_obj else None
        if template_norm == "training-script":
            try:
                training_script = planner.generate_training_script(
                    course_block=goal,
                    total_clips=total_clips,
                    clip_seconds=clip_sec,
                    aspect_ratio=aspect,
                    seed=seed,
                    style_hint=style,
                )
            except Exception:
                training_script = build_training_script_fallback(goal, total_clips, clip_sec)

            segments = training_script_to_segments(training_script, total_clips)

            # Non-interactive defaults for training scripts
            style = style or "Modern enterprise training aesthetic, clean typography, subtle motion graphics, warm approachable tone, high-quality 3D animation"
            lighting = lighting or "Soft diffused studio lighting at 5000K with gentle wrap-around key and subtle rim light, minimal shadows"
            camera = camera or "Steady slow dolly-in with subtle micro-arcs, centered medium shot, no cuts"
            characters = characters or "Friendly light-brown teddy bear host, plush texture, small teal scarf as a consistent anchor, expressive but gentle gestures, no clothing changes"
            environment = environment or "Clean minimal studio with soft neutral background, subtle floating UI silhouettes and labeled icon tiles, light motion graphics, brand-safe"
            negatives = ["No sudden zooms", "No random cuts", "No camera shake", "Do not change character outfits", "Do not add new characters", "Do not change environment"]
        else:
            plan = planner.generate_plan(
                topic=goal,
                total_clips=total_clips,
                clip_seconds=clip_sec,
                aspect_ratio=aspect,
                negative_library=NEGATIVE_LIBRARY,
                seed=seed,
                vision=vision_dict,
            )
            style = plan.global_style
            lighting = plan.lighting
            camera = plan.camera
            characters = plan.characters
            environment = plan.environment
            negatives = plan.negatives
            segments = plan.segments
    else:
        if template_norm == "training-script":
            # Non-interactive defaults for training scripts
            style = style or "Modern enterprise training aesthetic, clean typography, subtle motion graphics, warm approachable tone, high-quality 3D animation"
            lighting = lighting or "Soft diffused studio lighting at 5000K with gentle wrap-around key and subtle rim light, minimal shadows"
            camera = camera or "Steady slow dolly-in with subtle micro-arcs, centered medium shot, no cuts"
            characters = characters or "Friendly light-brown teddy bear host, plush texture, small teal scarf as a consistent anchor, expressive but gentle gestures, no clothing changes"
            environment = environment or "Clean minimal studio with soft neutral background, subtle floating UI silhouettes and labeled icon tiles, light motion graphics, brand-safe"
            negatives = ["No sudden zooms", "No random cuts", "No camera shake", "Do not change character outfits", "Do not add new characters", "Do not change environment"]

            planner = PromptLLMClient(mock=mock_llm)
            try:
                training_script = planner.generate_training_script(
                    course_block=goal,
                    total_clips=total_clips,
                    clip_seconds=clip_sec,
                    aspect_ratio=aspect,
                    seed=seed,
                    style_hint=style,
                )
            except Exception:
                training_script = build_training_script_fallback(goal, total_clips, clip_sec)

            segments = training_script_to_segments(training_script, total_clips)
        else:
            style = style or _prompt_list("Style", "cinematic warm tones, shallow depth of field")
            lighting = lighting or _prompt_list("Lighting", "golden hour warm glow")
            camera = camera or _prompt_list("Camera motion", "slow dolly-in at constant speed")
            characters = characters or _prompt_list("Characters", "Protagonist courier in neon jacket with visor helmet")
            environment = environment or _prompt_list("Environment", "Rainy neon alley with reflective puddles")
            negatives = _pick_negatives()

    job_id = make_job_id(project)
    run_dir = runs_root() / job_id
    run_dir.mkdir(parents=True, exist_ok=False)

    storyboard = build_storyboard(
        job_id=job_id,
        goal=goal,
        total_seconds=normalized_total,
        clip_seconds=clip_sec,
        aspect_ratio=aspect,
        seed=seed,
        style=style,
        camera=camera,
        lighting=lighting,
        negatives=negatives,
        characters=characters,
        environment=environment,
        template=template_norm,
    )

    if training_script:
        save_json(run_dir / "script.json", training_script)
        (run_dir / "voiceover.txt").write_text(training_script_voiceover_text(training_script), encoding="utf-8")
        (run_dir / "voiceover.srt").write_text(training_script_to_srt(training_script), encoding="utf-8")
        append_log(run_dir, "Wrote training script exports: script.json, voiceover.txt, voiceover.srt")

    prompts_file = build_prompts(storyboard, total_clips, segments=segments)

    save_json(run_dir / "storyboard.json", storyboard.to_dict())
    save_json(run_dir / "prompts.json", prompts_file.to_dict())
    (run_dir / "prompts_readable.md").write_text(render_prompts_markdown(storyboard, prompts_file), encoding="utf-8")
    
    # Save vision if captured
    if vision_obj:
        from vision_qa import save_vision
        save_vision(vision_obj, run_dir)
    
    RunState(status="new", current_clip=1, total_clips=total_clips).save(run_dir / "run_state.json")
    append_log(run_dir, "Initialized run and wrote storyboard/prompts.")

    typer.echo(f"Run created at {run_dir}")
    typer.echo("Files: storyboard.json, prompts.json, run_state.json")


@app.command()
def preview(run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest).")) -> None:
    """Pretty-print storyboard and prompts then ask for approval."""

    run_dir = _resolve_run_dir(run)
    storyboard, prompts, state = _load_files(run_dir)

    readable_path = run_dir / "prompts_readable.md"
    if readable_path.exists():
        typer.echo(readable_path.read_text(encoding="utf-8"))
    else:
        readable = render_prompts_markdown(storyboard, prompts)
        readable_path.write_text(readable, encoding="utf-8")
        typer.echo(readable)

    typer.echo("\n---\n")
    typer.echo("Storyboard (raw JSON):\n" + pretty_json(storyboard.to_dict()))
    approved = typer.confirm("Approve?", default=False)

    if approved:
        state.status = "approved"
        state.save(run_dir / "run_state.json")
        append_log(run_dir, "Storyboard/prompts approved.")
        typer.secho("Approved.", fg=typer.colors.GREEN)
    else:
        typer.secho("Not approved. You can edit JSON manually or rerun `demo new`.", fg=typer.colors.YELLOW)


@app.command()
def generate(
    run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest)."),
    mock: bool = typer.Option(False, help="Use mock clips instead of calling Azure."),
    regenerate: Optional[str] = typer.Option(None, help="Comma-separated clip indices to regenerate (e.g., '3,4')."),
    reference_image: Optional[Path] = typer.Option(
        None,
        help="Optional reference image to seed clip 1 only (will be auto-resized to match the target size).",
    ),
) -> None:
    """Generate sequential clips, extracting last frames and checkpointing state."""

    run_dir = _resolve_run_dir(run)
    storyboard, prompts, state = _load_files(run_dir)

    client = VideoClient(mock=mock)

    regenerate_set = set()
    if regenerate:
        try:
            regenerate_set = {int(x.strip()) for x in regenerate.split(",")}
        except ValueError:
            typer.secho("Invalid --regenerate format. Use comma-separated integers (e.g., '3,4').", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    typer.echo(f"Starting generation for run {run_dir} from clip {state.current_clip} of {state.total_clips}")
    if regenerate_set:
        typer.echo(f"Will regenerate clips: {sorted(regenerate_set)}")

    # Prepare/record clip-1 reference image (if provided)
    if reference_image:
        try:
            size_str = VideoClient._map_aspect_to_size(storyboard.aspect_ratio)
            w, h = parse_size(size_str)
            prepared = run_dir / "reference_01.jpg"
            prepare_reference_image(reference_image, prepared, w, h, mode="cover")
            state.reference_image = prepared.name
            save_json(run_dir / "run_state.json", state.to_dict())
            append_log(run_dir, f"Prepared reference image for clip 1: {prepared.name} (from {reference_image})")
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"Failed to prepare --reference-image: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    stop_flag = run_dir / "STOP_AFTER_CURRENT"
    for clip in prompts.clips:
        clip_path = run_dir / f"clip_{clip.index:02d}.mp4"
        last_frame_path = run_dir / f"last_frame_{clip.index:02d}.jpg"

        # Skip if already generated and not in regenerate set
        if clip.index < state.current_clip and clip.index not in regenerate_set:
            if clip_path.exists():
                typer.secho(f"Clip {clip.index} already exists, skipping.", fg=typer.colors.YELLOW)
                continue

        # Init image strategy:
        # - Clip 1: optional user-provided reference image (stored in run_state.json)
        # - Clip 2+: previous clip last frame (continuity)
        init_image: Optional[Path]
        if clip.index == 1 and state.reference_image:
            candidate = run_dir / state.reference_image
            init_image = candidate if candidate.exists() else None
        elif clip.index > 1:
            candidate = run_dir / f"last_frame_{clip.index - 1:02d}.jpg"
            init_image = candidate if candidate.exists() else None
        else:
            init_image = None

        typer.secho(f"Rendering clip {clip.index}/{state.total_clips}...", fg=typer.colors.BLUE)
        try:
            prompt_for_api = clip.prompt
            if clip.index == 1 and init_image is not None:
                prompt_for_api = (
                    prompt_for_api
                    + "\n\nReference image is provided as the first frame. Preserve its subject identity/appearance, environment, palette, and overall composition; only animate the described action and camera motion."  # noqa: E501
                )
            _path, video_id = client.generate_clip_with_id(
                prompt=prompt_for_api,
                output_path=clip_path,
                duration=storyboard.clip_seconds,
                aspect_ratio=storyboard.aspect_ratio,
                seed=storyboard.seed,
                init_image=init_image,
            )
            _extract_last_frame(clip_path, last_frame_path)
            state.artifacts[str(clip.index)] = ClipArtifact(
                clip=clip_path.name,
                last_frame=last_frame_path.name,
                video_id=video_id,
            )
            state.current_clip = clip.index + 1
            state.status = "running" if clip.index < state.total_clips else "complete"
            save_json(run_dir / "run_state.json", state.to_dict())
            append_log(run_dir, f"Clip {clip.index} done.")
            typer.secho(f"Clip {clip.index} complete.", fg=typer.colors.GREEN)

            # Best-effort stop: finish the current clip, then halt before starting the next.
            if stop_flag.exists() and clip.index < state.total_clips:
                state.status = "paused"
                save_json(run_dir / "run_state.json", state.to_dict())
                append_log(run_dir, "Stop requested; pausing after current clip.")
                break
        except Exception as exc:  # noqa: BLE001
            state.status = "failed"
            save_json(run_dir / "run_state.json", state.to_dict())
            append_log(run_dir, f"Clip {clip.index} failed: {exc}")
            raise typer.Exit(code=1)

    typer.secho("All clips generated.", fg=typer.colors.GREEN)


@app.command()
def remix(
    prompt: str = typer.Argument(..., help="Describe the change to apply to the existing video."),
    run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest)."),
    video_id: Optional[str] = typer.Option(None, help="Video ID to remix (defaults to latest generated clip in the run)."),
    output: Optional[Path] = typer.Option(None, help="Output MP4 path (defaults into the run folder)."),
    mock: bool = typer.Option(False, help="Use mock output instead of calling Azure/OpenAI."),
) -> None:
    """Remix a completed video to make a targeted change, preserving continuity/composition."""

    run_dir = _resolve_run_dir(run)
    storyboard, prompts, state = _load_files(run_dir)
    _ = prompts
    _ = storyboard

    resolved_video_id = video_id
    if not resolved_video_id:
        # Prefer the most recently completed clip artifact that has a video_id.
        for idx in range(state.total_clips, 0, -1):
            art = state.artifacts.get(str(idx))
            if art and getattr(art, "video_id", None):
                resolved_video_id = art.video_id
                break

    if not resolved_video_id:
        raise typer.BadParameter("No video_id provided and none found in run_state.json artifacts. Generate at least one clip first, or pass --video-id.")

    if not output:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        output = run_dir / f"remix_{ts}.mp4"

    client = VideoClient(mock=mock)
    typer.secho(f"Remixing video {resolved_video_id}...", fg=typer.colors.BLUE)
    try:
        out_path, new_id = client.remix_video(resolved_video_id, prompt=prompt, output_path=output)
        append_log(run_dir, f"Remix created: source={resolved_video_id} new={new_id} file={out_path.name}")
        typer.secho(f"Remix complete: {out_path.resolve()} (new id: {new_id})", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        append_log(run_dir, f"Remix failed: {exc}")
        raise typer.Exit(code=1)


@app.command()
def stitch(
    run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest)."),
    fast: bool = typer.Option(False, help="Use concat copy (fast path) if streams match."),
) -> None:
    """Concat clips into final_video.mp4 using ffmpeg."""

    run_dir = _resolve_run_dir(run)
    _, prompts, _ = _load_files(run_dir)
    clips: List[Path] = []
    for clip in prompts.clips:
        clip_path = run_dir / f"clip_{clip.index:02d}.mp4"
        if not clip_path.exists():
            raise typer.BadParameter(f"Missing clip: {clip_path}")
        clips.append(clip_path)

    final_path = run_dir / "final_video.mp4"
    try:
        concat_videos(clips, final_path, reencode=not fast)
        append_log(run_dir, "Stitching complete.")
        typer.secho(f"Final video at {final_path.resolve()}", fg=typer.colors.GREEN)
    except StitchError as exc:
        append_log(run_dir, f"Stitch failed: {exc}")
        raise typer.Exit(code=1)


@app.command()
def refine(
    run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest)."),
    interview: bool = typer.Option(False, help="Re-run vision interview."),
    field: Optional[str] = typer.Option(None, help="Specific field to update (e.g., 'lighting', 'camera')."),
    segment: Optional[int] = typer.Option(None, help="Specific segment index to regenerate (1-based)."),
) -> None:
    """Refine prompts without regenerating videos. Updates storyboard/prompts using LLM."""
    
    run_dir = _resolve_run_dir(run)
    storyboard, prompts, state = _load_files(run_dir)
    
    # Load or update vision
    from vision_qa import load_vision, conduct_vision_interview, save_vision
    vision_obj = load_vision(run_dir)
    
    if interview or not vision_obj:
        vision_obj = conduct_vision_interview(storyboard.goal, skip_interview=False)
        save_vision(vision_obj, run_dir)
        typer.echo("Vision updated.\n")
    
    # If specific field requested, prompt for new value
    if field:
        current_value = getattr(storyboard, field, None)
        if current_value is None:
            typer.secho(f"Field '{field}' not found in storyboard.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        typer.echo(f"Current {field}: {current_value}")
        new_value = input(f"New {field} (or press Enter to keep): ").strip()
        
        if new_value:
            setattr(storyboard, field, new_value)
            save_json(run_dir / "storyboard.json", storyboard.to_dict())
            
            # Rebuild prompts with updated storyboard
            prompts_file = build_prompts(storyboard, len(prompts.clips))
            save_json(run_dir / "prompts.json", prompts_file.to_dict())
            (run_dir / "prompts_readable.md").write_text(render_prompts_markdown(storyboard, prompts_file), encoding="utf-8")
            
            append_log(run_dir, f"Refined field '{field}'")
            typer.secho(f"Updated {field} and regenerated prompts.", fg=typer.colors.GREEN)
        else:
            typer.echo("No change.")
        
        return
    
    # If specific segment requested, regenerate just that segment
    if segment:
        if not (1 <= segment <= len(prompts.clips)):
            typer.secho(f"Segment {segment} out of range (1-{len(prompts.clips)}).", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        typer.echo(f"Current segment {segment}:")
        typer.echo(prompts.clips[segment - 1].prompt[:200] + "...")
        
        feedback = input(f"\nWhat should change about segment {segment}? ").strip()
        if not feedback:
            typer.echo("No feedback provided, exiting.")
            return
        
        # Use LLM to regenerate this specific segment with feedback
        planner = PromptLLMClient()
        typer.echo("\nRegenerating segment with LLM...")
        
        # For now, regenerate entire plan (future: segment-specific refinement)
        vision_dict = vision_obj.to_dict() if vision_obj else None
        plan = planner.generate_plan(
            topic=f"{storyboard.goal} (User feedback for segment {segment}: {feedback})",
            total_clips=len(prompts.clips),
            clip_seconds=storyboard.clip_seconds,
            aspect_ratio=storyboard.aspect_ratio,
            negative_library=storyboard.negatives,
            seed=storyboard.seed,
            vision=vision_dict,
        )
        
        # Update storyboard fields and rebuild prompts
        storyboard.global_style = plan.global_style
        storyboard.camera = plan.camera
        storyboard.lighting = plan.lighting
        storyboard.characters = plan.characters
        storyboard.environment = plan.environment
        
        prompts_file = build_prompts(storyboard, len(prompts.clips), segments=plan.segments)
        
        save_json(run_dir / "storyboard.json", storyboard.to_dict())
        save_json(run_dir / "prompts.json", prompts_file.to_dict())
        (run_dir / "prompts_readable.md").write_text(render_prompts_markdown(storyboard, prompts_file), encoding="utf-8")
        
        append_log(run_dir, f"Refined segment {segment} with feedback: {feedback}")
        typer.secho(f"Updated segment {segment}. Review with `demo preview`.", fg=typer.colors.GREEN)
        return
    
    # General refinement: regenerate everything with updated vision
    typer.echo("Regenerating all prompts with current vision...")
    
    planner = PromptLLMClient()
    vision_dict = vision_obj.to_dict() if vision_obj else None
    plan = planner.generate_plan(
        topic=storyboard.goal,
        total_clips=len(prompts.clips),
        clip_seconds=storyboard.clip_seconds,
        aspect_ratio=storyboard.aspect_ratio,
        negative_library=storyboard.negatives,
        seed=storyboard.seed,
        vision=vision_dict,
    )
    
    # Update storyboard and prompts
    storyboard.global_style = plan.global_style
    storyboard.camera = plan.camera
    storyboard.lighting = plan.lighting
    storyboard.characters = plan.characters
    storyboard.environment = plan.environment
    
    prompts_file = build_prompts(storyboard, len(prompts.clips), segments=plan.segments)
    
    save_json(run_dir / "storyboard.json", storyboard.to_dict())
    save_json(run_dir / "prompts.json", prompts_file.to_dict())
    (run_dir / "prompts_readable.md").write_text(render_prompts_markdown(storyboard, prompts_file), encoding="utf-8")
    
    append_log(run_dir, "Refined prompts with updated vision")
    typer.secho("Prompts regenerated. Review with `demo preview`.", fg=typer.colors.GREEN)


@app.command()
def feedback(
    message: str = typer.Argument(..., help="Your feedback on what to improve in the video prompts."),
    run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest)."),
) -> None:
    """Update prompts based on your feedback using AI. Describe what needs to change."""
    
    run_dir = _resolve_run_dir(run)
    storyboard, prompts, state = _load_files(run_dir)
    
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Feedback: {message}")
    typer.echo(f"{'='*60}\n")
    
    # Load vision if exists
    from vision_qa import load_vision
    vision_obj = load_vision(run_dir)
    vision_dict = vision_obj.to_dict() if vision_obj else None
    
    # Use LLM to incorporate feedback and regenerate prompts
    planner = PromptLLMClient()
    typer.echo("Regenerating prompts with your feedback...\n")
    
    # Append feedback to the topic to guide LLM
    enhanced_topic = f"{storyboard.goal}\n\nIMPORTANT USER FEEDBACK: {message}"
    
    plan = planner.generate_plan(
        topic=enhanced_topic,
        total_clips=len(prompts.clips),
        clip_seconds=storyboard.clip_seconds,
        aspect_ratio=storyboard.aspect_ratio,
        negative_library=storyboard.negatives,
        seed=storyboard.seed,
        vision=vision_dict,
    )
    
    # Update storyboard and prompts with feedback-informed plan
    storyboard.global_style = plan.global_style
    storyboard.camera = plan.camera
    storyboard.lighting = plan.lighting
    storyboard.characters = plan.characters
    storyboard.environment = plan.environment
    
    prompts_file = build_prompts(storyboard, len(prompts.clips), segments=plan.segments)
    
    # CRITICAL: Use LLM to weave feedback into each prompt, making them longer and more detailed
    typer.echo("Weaving feedback into prompts with LLM...\n")
    for clip in prompts_file.clips:
        enhanced_prompt = planner.weave_feedback_into_prompt(
            original_prompt=clip.prompt,
            feedback=message,
            clip_index=clip.index
        )
        clip.prompt = enhanced_prompt
        typer.echo(f"  ✓ Enhanced clip {clip.index} prompt ({len(enhanced_prompt)} characters)")
    
    save_json(run_dir / "storyboard.json", storyboard.to_dict())
    save_json(run_dir / "prompts.json", prompts_file.to_dict())
    (run_dir / "prompts_readable.md").write_text(render_prompts_markdown(storyboard, prompts_file), encoding="utf-8")
    
    append_log(run_dir, f"Applied feedback: {message}")
    typer.secho("\n✓ Prompts updated with your feedback!", fg=typer.colors.GREEN)
    typer.echo("\nNext steps:")
    typer.echo("  1. Review changes: python cli.py preview")
    typer.echo("  2. Regenerate clips: python cli.py generate")
    typer.echo("  3. If still not right, use 'feedback' again with more specific notes\n")


@app.command()
def open(
    run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest)."),
) -> None:
    """Open the final video or print its absolute path."""

    run_dir = _resolve_run_dir(run)
    final_path = run_dir / "final_video.mp4"
    if not final_path.exists():
        typer.secho("final_video.mp4 not found. Run `demo stitch` first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(str(final_path.resolve()))
    try:
        os.startfile(final_path)  # type: ignore[attr-defined]
    except Exception:
        typer.echo("Unable to auto-open; use the path above.")


@app.command()
def clean(
    run: Optional[Path] = typer.Option(None, help="Run directory (defaults to latest)."),
    keep_final: bool = typer.Option(True, help="Keep final_video.mp4 when cleaning."),
) -> None:
    """Delete intermediate artifacts; keeps final video by default."""

    run_dir = _resolve_run_dir(run)
    final_path = run_dir / "final_video.mp4"

    if keep_final and final_path.exists():
        for item in run_dir.iterdir():
            if item == final_path:
                continue
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        append_log(run_dir, "Cleaned intermediates; kept final video.")
    else:
        shutil.rmtree(run_dir)
        append_log(run_dir, "Deleted run directory.")
    typer.secho("Clean complete.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app(prog_name="demo")
