from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

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
    ensure_clip_counts,
    latest_run_dir,
    load_json,
    make_job_id,
    pretty_json,
    runs_root,
    save_json,
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
    if auto:
        planner = PromptLLMClient(mock=mock_llm)
        vision_dict = vision_obj.to_dict() if vision_obj else None
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
    )
    prompts_file = build_prompts(storyboard, total_clips, segments=segments)

    save_json(run_dir / "storyboard.json", storyboard.to_dict())
    save_json(run_dir / "prompts.json", prompts_file.to_dict())
    
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

    typer.echo("Storyboard:\n" + pretty_json(storyboard.to_dict()))
    typer.echo("\nPrompts:\n" + pretty_json(prompts.to_dict()))
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

    for clip in prompts.clips:
        clip_path = run_dir / f"clip_{clip.index:02d}.mp4"
        last_frame_path = run_dir / f"last_frame_{clip.index:02d}.jpg"

        # Skip if already generated and not in regenerate set
        if clip.index < state.current_clip and clip.index not in regenerate_set:
            if clip_path.exists():
                typer.secho(f"Clip {clip.index} already exists, skipping.", fg=typer.colors.YELLOW)
                continue

        init_image = run_dir / f"last_frame_{clip.index - 1:02d}.jpg" if clip.index > 1 else None
        if init_image and not init_image.exists():
            init_image = None

        typer.secho(f"Rendering clip {clip.index}/{state.total_clips}...", fg=typer.colors.BLUE)
        try:
            client.generate_clip(
                prompt=clip.prompt,
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
            )
            state.current_clip = clip.index + 1
            state.status = "running" if clip.index < state.total_clips else "complete"
            save_json(run_dir / "run_state.json", state.to_dict())
            append_log(run_dir, f"Clip {clip.index} done.")
            typer.secho(f"Clip {clip.index} complete.", fg=typer.colors.GREEN)
        except Exception as exc:  # noqa: BLE001
            state.status = "failed"
            save_json(run_dir / "run_state.json", state.to_dict())
            append_log(run_dir, f"Clip {clip.index} failed: {exc}")
            raise typer.Exit(code=1)

    typer.secho("All clips generated.", fg=typer.colors.GREEN)


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
