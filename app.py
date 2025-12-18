"""Streamlit frontend for instructional video generation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from ffmpeg_bin import ffmpeg_exe
from prompt_llm import PromptLLMClient
from schemas import ClipArtifact, PromptsFile, RunState, Storyboard
from stitcher import concat_videos
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
    prepare_reference_image,
    render_prompts_markdown,
    runs_root,
    save_json,
    training_script_to_segments,
    training_script_to_srt,
    training_script_voiceover_text,
)
from video_client import VideoClient


def extract_last_frame(clip_path: Path, output_path: Path) -> None:
    """Extract the last frame from a video clip."""
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


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "run_dir" not in st.session_state:
        st.session_state.run_dir = None
    if "storyboard" not in st.session_state:
        st.session_state.storyboard = None
    if "prompts" not in st.session_state:
        st.session_state.prompts = None
    if "state" not in st.session_state:
        st.session_state.state = None
    if "generation_started" not in st.session_state:
        st.session_state.generation_started = False
    if "gen_process" not in st.session_state:
        st.session_state.gen_process = None
    if "action_process" not in st.session_state:
        st.session_state.action_process = None
    if "action_label" not in st.session_state:
        st.session_state.action_label = ""
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False


def _stop_flag_path(run_dir: Path) -> Path:
    return run_dir / "STOP_AFTER_CURRENT"


def _start_generate_subprocess(run_dir: Path, mock: bool) -> subprocess.Popen:
    """Start CLI generation in the background (keeps Streamlit UI responsive)."""
    flag = _stop_flag_path(run_dir)
    try:
        if flag.exists():
            flag.unlink()
    except Exception:
        pass

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("cli.py")),
        "generate",
        "--run",
        str(run_dir),
    ]
    if mock:
        cmd.append("--mock")

    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _start_cli_subprocess(args: list[str]) -> subprocess.Popen:
    """Start a CLI command in the background (keeps Streamlit UI responsive)."""
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("cli.py")),
        *args,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _tail_text(path: Path, max_lines: int = 200) -> str:
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[-max_lines:]) + "\n"
    except Exception as exc:
        return f"(failed to read {path.name}: {exc})\n"


def load_run_files(run_dir: Path) -> tuple[Storyboard, PromptsFile, RunState]:
    """Load storyboard, prompts, and state from a run directory."""
    storyboard = Storyboard.from_dict(load_json(run_dir / "storyboard.json"))
    prompts = PromptsFile.from_dict(load_json(run_dir / "prompts.json"))
    state = RunState.load(run_dir / "run_state.json")
    return storyboard, prompts, state


def create_new_run(
    course_info: str,
    total_sec: int = 60,
    clip_sec: int = 12,
    aspect: str = "16:9",
    mock_llm: bool = False,
) -> Path:
    """Create a new video generation run."""
    
    # Calculate total clips
    total_clips, normalized_total = ensure_clip_counts(total_sec, clip_sec)
    
    # Default values for training videos
    style = "Modern enterprise training aesthetic, clean typography, subtle motion graphics, warm approachable tone, high-quality 3D animation"
    lighting = "Soft diffused studio lighting at 5000K with gentle wrap-around key and subtle rim light, minimal shadows"
    camera = "Steady slow dolly-in with subtle micro-arcs, centered medium shot, no cuts"
    characters = "Friendly light-brown teddy bear host, plush texture, small teal scarf as a consistent anchor, expressive but gentle gestures, no clothing changes"
    environment = "Clean minimal studio with soft neutral background, subtle floating UI silhouettes and labeled icon tiles, light motion graphics, brand-safe"
    negatives = [
        "No sudden zooms",
        "No random cuts",
        "No camera shake",
        "Do not change character outfits",
        "Do not add new characters",
        "Do not change environment",
    ]
    
    # Generate training script using LLM
    planner = PromptLLMClient(mock=mock_llm)
    try:
        training_script = planner.generate_training_script(
            course_block=course_info,
            total_clips=total_clips,
            clip_seconds=clip_sec,
            aspect_ratio=aspect,
            seed=None,
            style_hint=style,
        )
    except Exception as e:
        st.warning(
            "LLM generation failed; using deterministic fallback. "
            f"Error: {e}\n\n"
            "Tip: if this is a timeout, increase `AZURE_OPENAI_TEXT_TIMEOUT` (seconds) or set "
            "`AZURE_OPENAI_TEXT_RETRIES` in your `.env`, then retry Create Project."
        )
        training_script = build_training_script_fallback(course_info, total_clips, clip_sec)
    
    segments = training_script_to_segments(training_script, total_clips)
    
    # Create run directory
    job_id = make_job_id("streamlit")
    run_dir = runs_root() / job_id
    run_dir.mkdir(parents=True, exist_ok=False)
    
    # Build storyboard
    storyboard = build_storyboard(
        job_id=job_id,
        goal=course_info,
        total_seconds=normalized_total,
        clip_seconds=clip_sec,
        aspect_ratio=aspect,
        seed=None,
        style=style,
        camera=camera,
        lighting=lighting,
        negatives=negatives,
        characters=characters,
        environment=environment,
        template="training-script",
    )
    
    # Save training script artifacts
    save_json(run_dir / "script.json", training_script)
    (run_dir / "voiceover.txt").write_text(
        training_script_voiceover_text(training_script), encoding="utf-8"
    )
    (run_dir / "voiceover.srt").write_text(
        training_script_to_srt(training_script), encoding="utf-8"
    )
    
    # Build and save prompts
    prompts_file = build_prompts(storyboard, total_clips, segments=segments)
    save_json(run_dir / "storyboard.json", storyboard.to_dict())
    save_json(run_dir / "prompts.json", prompts_file.to_dict())
    (run_dir / "prompts_readable.md").write_text(
        render_prompts_markdown(storyboard, prompts_file), encoding="utf-8"
    )
    
    # Initialize run state
    RunState(status="new", current_clip=1, total_clips=total_clips).save(
        run_dir / "run_state.json"
    )
    append_log(run_dir, "Initialized run via Streamlit UI")
    
    return run_dir


def generate_clips(run_dir: Path, mock: bool = False, progress_callback=None) -> bool:
    """Generate video clips for a run. Returns True if successful."""
    storyboard, prompts, state = load_run_files(run_dir)
    client = VideoClient(mock=mock)
    
    for i, clip in enumerate(prompts.clips):
        clip_path = run_dir / f"clip_{clip.index:02d}.mp4"
        last_frame_path = run_dir / f"last_frame_{clip.index:02d}.jpg"
        
        # Skip if already generated
        if clip.index < state.current_clip and clip_path.exists():
            if progress_callback:
                progress_callback(i + 1, len(prompts.clips), f"Clip {clip.index} already exists")
            continue
        
        # Determine init image
        init_image: Optional[Path] = None
        if clip.index > 1:
            candidate = run_dir / f"last_frame_{clip.index - 1:02d}.jpg"
            init_image = candidate if candidate.exists() else None
        
        if progress_callback:
            progress_callback(i + 1, len(prompts.clips), f"Generating clip {clip.index}...")
        
        try:
            _path, video_id = client.generate_clip_with_id(
                prompt=clip.prompt,
                output_path=clip_path,
                duration=storyboard.clip_seconds,
                aspect_ratio=storyboard.aspect_ratio,
                seed=storyboard.seed,
                init_image=init_image,
            )
            extract_last_frame(clip_path, last_frame_path)
            
            state.artifacts[str(clip.index)] = ClipArtifact(
                clip=clip_path.name,
                last_frame=last_frame_path.name,
                video_id=video_id,
            )
            state.current_clip = clip.index + 1
            state.status = "running" if clip.index < state.total_clips else "complete"
            save_json(run_dir / "run_state.json", state.to_dict())
            append_log(run_dir, f"Clip {clip.index} done.")
        except Exception as e:
            state.status = "failed"
            save_json(run_dir / "run_state.json", state.to_dict())
            append_log(run_dir, f"Clip {clip.index} failed: {e}")
            return False
    
    return True


def stitch_clips(run_dir: Path) -> Optional[Path]:
    """Stitch all clips into final video. Returns path to final video or None if failed."""
    _, prompts, _ = load_run_files(run_dir)
    
    clips = []
    for clip in prompts.clips:
        clip_path = run_dir / f"clip_{clip.index:02d}.mp4"
        if not clip_path.exists():
            st.error(f"Missing clip: {clip_path}")
            return None
        clips.append(clip_path)
    
    final_path = run_dir / "final_video.mp4"
    try:
        concat_videos(clips, final_path, reencode=True)
        append_log(run_dir, "Stitching complete via Streamlit UI")
        return final_path
    except Exception as e:
        append_log(run_dir, f"Stitch failed: {e}")
        st.error(f"Stitching failed: {e}")
        return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Instructional Video Generator",
        page_icon="🎬",
        layout="wide",
    )
    
    st.title("🎬 Instructional Video Generator")
    st.markdown("Generate training videos from course information using AI")
    
    init_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        mock_mode = st.checkbox(
            "Mock Mode",
            value=False,
            help="Use mock video generation for testing (no API calls)",
        )
        
        total_sec = st.slider(
            "Total Duration (seconds)",
            min_value=12,
            max_value=180,
            value=60,
            step=12,
        )
        
        clip_sec = st.selectbox(
            "Clip Length (seconds)",
            options=[4, 8, 12],
            index=2,
        )
        
        aspect = st.selectbox(
            "Aspect Ratio",
            options=["16:9", "9:16", "1:1"],
            index=0,
        )
        
        st.divider()
        
        # Load existing run
        if st.button("📂 Load Latest Run"):
            latest = latest_run_dir(runs_root())
            if latest:
                st.session_state.run_dir = latest
                try:
                    storyboard, prompts, state = load_run_files(latest)
                    st.session_state.storyboard = storyboard
                    st.session_state.prompts = prompts
                    st.session_state.state = state
                    st.success(f"Loaded: {latest.name}")
                except Exception as e:
                    st.error(f"Failed to load run: {e}")
            else:
                st.info("No existing runs found")
    
    # Main content area
    tab1, tab2, tab_iter, tab3 = st.tabs(["📝 Create", "🎥 Generate", "🧪 Iterate", "📹 Results"])
    
    with tab1:
        st.header("Create New Video Project")
        
        course_info = st.text_area(
            "Course Information",
            height=200,
            placeholder="Enter the course content, learning objectives, and key points you want to cover in the training video...",
            help="Provide detailed information about what you want to teach",
        )
        
        if st.button("🚀 Create Project", type="primary", disabled=not course_info):
            with st.spinner("Creating project and generating script..."):
                try:
                    run_dir = create_new_run(
                        course_info=course_info,
                        total_sec=total_sec,
                        clip_sec=clip_sec,
                        aspect=aspect,
                        mock_llm=mock_mode,
                    )
                    st.session_state.run_dir = run_dir
                    storyboard, prompts, state = load_run_files(run_dir)
                    st.session_state.storyboard = storyboard
                    st.session_state.prompts = prompts
                    st.session_state.state = state
                    st.success(f"✅ Project created: {run_dir.name}")
                    st.info("Switch to the 'Generate' tab to start video generation")
                except Exception as e:
                    st.error(f"Failed to create project: {e}")
        
        # Show current project info if loaded
        if st.session_state.storyboard:
            st.divider()
            st.subheader("Current Project")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Clips", st.session_state.state.total_clips)
                st.metric("Duration", f"{st.session_state.storyboard.total_seconds}s")
            with col2:
                st.metric("Clip Length", f"{st.session_state.storyboard.clip_seconds}s")
                st.metric("Aspect Ratio", st.session_state.storyboard.aspect_ratio)
            
            # Show prompts
            with st.expander("📄 View Prompts"):
                if st.session_state.run_dir:
                    readable_path = st.session_state.run_dir / "prompts_readable.md"
                    if readable_path.exists():
                        st.markdown(readable_path.read_text(encoding="utf-8"))
    
    with tab2:
        st.header("Generate Video Clips")
        
        if not st.session_state.run_dir:
            st.info("👈 Create a project first in the 'Create' tab")
        else:
            st.write(f"**Project:** {st.session_state.run_dir.name}")

            # Manual refresh (Streamlit only re-runs on interaction)
            if st.button("Refresh status"):
                st.rerun()

            # Reload state from disk (disk is the source of truth).
            try:
                _, disk_prompts, disk_state = load_run_files(st.session_state.run_dir)
                st.session_state.state = disk_state
            except Exception:
                disk_prompts = None
                pass

            proc = st.session_state.gen_process
            running = bool(proc is not None and getattr(proc, "poll", None) and proc.poll() is None)

            st.write(f"**Status:** {st.session_state.state.status}{' (generating)' if running else ''}")
            
            # Show generation progress
            if st.session_state.state.status == "complete":
                st.success("✅ All clips generated!")
            elif st.session_state.state.status == "running":
                st.info(f"⏳ In progress: {st.session_state.state.current_clip - 1}/{st.session_state.state.total_clips} clips generated")
            
            # Generate controls (run in a subprocess so Stop works)
            can_generate = (not running) and st.session_state.state.status in ["new", "approved", "failed", "paused"]

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start generation", type="primary", disabled=not can_generate):
                    st.session_state.gen_process = _start_generate_subprocess(st.session_state.run_dir, mock=mock_mode)
                    st.session_state.stop_requested = False
                    st.info("Generation started. Use 'Stop after current clip' to pause.")
            with col_b:
                if st.button("Stop after current clip", disabled=not running):
                    try:
                        _stop_flag_path(st.session_state.run_dir).write_text("1", encoding="utf-8")
                        st.session_state.stop_requested = True
                        st.warning("Stop requested. The current clip will finish, then generation will pause.")
                    except Exception as e:
                        st.error(f"Failed to request stop: {e}")

            # Progress (best-effort; disk is truth, but run_state.json may lag or be missing artifacts)
            total = st.session_state.state.total_clips
            if disk_prompts is not None and getattr(disk_prompts, "clips", None):
                total = len(disk_prompts.clips)

            done_by_current = max(0, st.session_state.state.current_clip - 1)
            done_by_artifacts = len(getattr(st.session_state.state, "artifacts", {}) or {})
            try:
                done_by_files = len(list(st.session_state.run_dir.glob("clip_*.mp4")))
            except Exception:
                done_by_files = 0
            done = max(done_by_current, done_by_artifacts, done_by_files)
            if total > 0:
                st.progress(min(1.0, done / total))
                st.caption(f"Progress: {done}/{total} clips complete")

            if proc is not None and getattr(proc, "poll", None) and proc.poll() is not None and running is False:
                # Process finished; clear handle.
                st.session_state.gen_process = None
                st.success("Generation process finished. Refresh tabs to see updated clips.")

            st.divider()
            st.subheader("Logs")

            col_logs_a, col_logs_b = st.columns([1, 1])
            with col_logs_a:
                auto_refresh = st.checkbox(
                    "Auto-refresh logs while generating",
                    value=bool(st.session_state.get("auto_refresh_logs", False)),
                    disabled=not running,
                    help="When enabled, this page polls and re-runs periodically while generation is running.",
                )
                st.session_state.auto_refresh_logs = auto_refresh
            with col_logs_b:
                refresh_seconds = st.selectbox(
                    "Refresh every",
                    options=[2, 5, 10],
                    index=1,
                    disabled=not running or not auto_refresh,
                    format_func=lambda s: f"{s}s",
                )

            logs_path = st.session_state.run_dir / "logs.txt"
            st.code(_tail_text(logs_path, max_lines=200), language="text")

            if running and st.session_state.get("auto_refresh_logs", False):
                time.sleep(int(refresh_seconds))
                st.rerun()
            
            # Show individual clips if any exist
            if st.session_state.state.artifacts:
                st.divider()
                st.subheader("Generated Clips")
                
                for idx in sorted(st.session_state.state.artifacts.keys(), key=int):
                    artifact = st.session_state.state.artifacts[idx]
                    clip_path = st.session_state.run_dir / artifact.clip
                    
                    if clip_path.exists():
                        with st.expander(f"Clip {idx}"):
                            st.video(str(clip_path))

    with tab_iter:
        st.header("Iterate (Feedback / Remix)")

        if not st.session_state.run_dir:
            st.info("👈 Create or load a project first")
        else:
            st.write(f"**Project:** {st.session_state.run_dir.name}")

            # Refresh (Streamlit re-runs only on interaction)
            if st.button("Refresh iterate status"):
                st.rerun()

            # Update process state
            proc = st.session_state.action_process
            action_running = bool(proc is not None and getattr(proc, "poll", None) and proc.poll() is None)
            if proc is not None and getattr(proc, "poll", None) and proc.poll() is not None and action_running is False:
                st.session_state.action_process = None
                st.session_state.action_label = ""
                st.success("Action finished. Refresh to see updates.")

            # Avoid running iterate actions during clip generation.
            gen_proc = st.session_state.gen_process
            generating = bool(gen_proc is not None and getattr(gen_proc, "poll", None) and gen_proc.poll() is None)
            if generating:
                st.warning("Generation is running. Pause/finish generation before applying feedback or remix.")

            disabled_actions = action_running or generating
            if action_running and st.session_state.get("action_label"):
                st.info(f"Running: {st.session_state.action_label}")

            st.subheader("Feedback")
            feedback_text = st.text_area(
                "Describe what to change in the prompts",
                height=120,
                placeholder="Example: Keep the same narrator voice across clips; slow down pacing; make on-screen text larger.",
                disabled=disabled_actions,
            )
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                if st.button("Apply feedback to prompts", type="primary", disabled=disabled_actions or not feedback_text.strip()):
                    st.session_state.action_label = "Applying feedback"
                    st.session_state.action_process = _start_cli_subprocess(
                        ["feedback", feedback_text.strip(), "--run", str(st.session_state.run_dir)]
                    )
                    st.info("Feedback started. Watch logs for progress, then refresh prompts.")
            with col_f2:
                st.caption("This regenerates prompts (does not regenerate videos). After this, re-run generation for the clips you want to update.")

            st.divider()
            st.subheader("Remix")
            # Load state to find available video_ids
            try:
                _, _, disk_state = load_run_files(st.session_state.run_dir)
            except Exception:
                disk_state = None

            video_choices: list[tuple[str, Optional[str]]] = [("Latest generated clip", None)]
            if disk_state is not None:
                for k in sorted((disk_state.artifacts or {}).keys(), key=lambda x: int(x)):
                    art = disk_state.artifacts.get(k)
                    vid = getattr(art, "video_id", None) if art else None
                    if vid:
                        video_choices.append((f"Clip {k} (video_id={vid})", vid))

            choice_labels = [c[0] for c in video_choices]
            selected_label = st.selectbox("Source video", options=choice_labels, disabled=disabled_actions)
            selected_video_id = None
            for lbl, vid in video_choices:
                if lbl == selected_label:
                    selected_video_id = vid
                    break

            remix_prompt = st.text_area(
                "Describe the change for remix",
                height=100,
                placeholder="Example: Same shot and lighting; increase contrast of the on-screen title and make it larger.",
                disabled=disabled_actions,
            )
            if st.button("Run remix", disabled=disabled_actions or not remix_prompt.strip()):
                args = ["remix", remix_prompt.strip(), "--run", str(st.session_state.run_dir)]
                if selected_video_id:
                    args += ["--video-id", selected_video_id]
                if mock_mode:
                    args += ["--mock"]
                st.session_state.action_label = "Remixing video"
                st.session_state.action_process = _start_cli_subprocess(args)
                st.info("Remix started. The output file will be written into the run folder; see logs for the filename.")
    
    with tab3:
        st.header("Final Video")
        
        if not st.session_state.run_dir:
            st.info("👈 Create and generate a project first")
        else:
            final_path = st.session_state.run_dir / "final_video.mp4"
            
            if final_path.exists():
                st.success("✅ Final video ready!")
                st.video(str(final_path))
                
                # Download button
                with open(final_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Video",
                        data=f,
                        file_name=f"{st.session_state.run_dir.name}_final.mp4",
                        mime="video/mp4",
                    )
            else:
                # Check if all clips are generated
                if st.session_state.state.status == "complete":
                    if st.button("🔗 Stitch Clips", type="primary"):
                        with st.spinner("Stitching clips together..."):
                            final = stitch_clips(st.session_state.run_dir)
                            if final:
                                st.success("✅ Video stitched successfully!")
                                st.rerun()
                else:
                    st.info("Complete video generation first in the 'Generate' tab")


if __name__ == "__main__":
    main()
