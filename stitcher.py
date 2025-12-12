from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List

from ffmpeg_bin import ffmpeg_exe


class StitchError(RuntimeError):
    pass


def concat_videos(clips: List[Path], output_path: Path, reencode: bool = True, loudnorm: bool = True, crossfade: float = 1.5) -> Path:
    if not clips:
        raise StitchError("No clips provided for stitching.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _run(cmd: List[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise StitchError(result.stderr or "ffmpeg concat failed")

    if reencode:
        # Using xfade filter to create smooth crossfade transitions between clips.
        # This masks any motion discontinuity at clip boundaries.
        # Trim start of subsequent clips to remove Sora "settle" frames.
        boundary_trim_s = "0.250"  # ~7-8 frames at 30fps
        target_fps = "30"
        n = len(clips)
        xfade_duration = crossfade  # crossfade duration in seconds
        # Sora clips are typically 12s, after trim clips 2+ are ~11.75s
        clip1_duration = 12.0
        clip_n_duration = 12.0 - float(boundary_trim_s)

        inputs: List[str] = []
        for clip in clips:
            inputs += ["-i", str(clip)]

        def _filter_with_audio() -> str:
            parts: List[str] = []
            
            # Step 1: Preprocess all video clips (scale, fps - NO setpts yet, xfade needs proper frame rate)
            v_labels: List[str] = []
            a_labels: List[str] = []
            for i in range(n):
                v_out = f"v{i}"
                a_out = f"a{i}"
                v_labels.append(f"[{v_out}]")
                a_labels.append(f"[{a_out}]")

                if i == 0:
                    parts.append(
                        f"[{i}:v:0]scale=iw:ih:force_original_aspect_ratio=decrease,setsar=1,fps={target_fps}[{v_out}]"
                    )
                    parts.append(f"[{i}:a:0]asetpts=PTS-STARTPTS[{a_out}]")
                else:
                    # Use select instead of trim to preserve frame rate info for xfade
                    parts.append(
                        f"[{i}:v:0]scale=iw:ih:force_original_aspect_ratio=decrease,setsar=1,fps={target_fps},"
                        f"select='gte(t\\,{boundary_trim_s})'[{v_out}]"
                    )
                    parts.append(f"[{i}:a:0]atrim=start={boundary_trim_s},asetpts=PTS-STARTPTS[{a_out}]")

            # Step 2: Build xfade chain for video (if multiple clips)
            if n == 1:
                # Single clip: no crossfade needed, just apply setpts
                final_v_label = "v0"
            else:
                # Multiple clips: chain xfade filters
                # Calculate offsets: first clip full duration, subsequent clips trimmed
                for i in range(n - 1):
                    if i == 0:
                        # First xfade: [v0][v1] -> [vx1]
                        # Clip 1 is full duration, fade starts at (duration - xfade_duration)
                        offset = clip1_duration - xfade_duration
                        parts.append(f"[v0][v1]xfade=transition=fade:duration={xfade_duration}:offset={offset}[vx1]")
                    else:
                        # Subsequent xfades: [vxi][vi+1] -> [vx(i+1)]
                        # Accumulated duration: clip1 + (i * clip_n) - (i * xfade)
                        offset = clip1_duration + (i * clip_n_duration) - (i * xfade_duration) - xfade_duration
                        parts.append(f"[vx{i}][v{i+1}]xfade=transition=fade:duration={xfade_duration}:offset={offset}[vx{i+1}]")
                
                final_v_label = f"vx{n-1}"
            
            # Apply setpts AFTER xfade to normalize timestamps
            parts.append(f"[{final_v_label}]setpts=PTS-STARTPTS[v]")

            # Step 3: Audio concat (simple approach - concat doesn't need fades since we're using xfade for video)
            if n == 1:
                final_a_label = "a0"
            else:
                # Just concat audio normally - the visual crossfade is enough
                concat_in = "".join([f"[a{i}]" for i in range(n)])
                parts.append(f"{concat_in}concat=n={n}:v=0:a=1[a]")
                final_a_label = "a"

            # Step 4: Apply loudnorm if requested
            audio_post = "loudnorm=I=-24:TP=-2:LRA=7" if loudnorm else "anull"
            parts.append(f"[{final_a_label}]{audio_post}[aout]")
            
            return ";".join(parts)

        def _filter_video_only() -> str:
            parts: List[str] = []
            v_labels: List[str] = []
            
            # Preprocess all clips (no setpts before xfade)
            for i in range(n):
                v_out = f"v{i}"
                v_labels.append(f"[{v_out}]")
                if i == 0:
                    parts.append(
                        f"[{i}:v:0]scale=iw:ih:force_original_aspect_ratio=decrease,setsar=1,fps={target_fps}[{v_out}]"
                    )
                else:
                    parts.append(
                        f"[{i}:v:0]scale=iw:ih:force_original_aspect_ratio=decrease,setsar=1,fps={target_fps},"
                        f"select='gte(t\\,{boundary_trim_s})'[{v_out}]"
                    )
            
            # Build xfade chain (or pass through for single clip)
            if n == 1:
                final_v_label = "v0"
            else:
                # Calculate offsets with proper duration accounting
                for i in range(n - 1):
                    if i == 0:
                        offset = clip1_duration - xfade_duration
                        parts.append(f"[v0][v1]xfade=transition=fade:duration={xfade_duration}:offset={offset}[vx1]")
                    else:
                        offset = clip1_duration + (i * clip_n_duration) - (i * xfade_duration) - xfade_duration
                        parts.append(f"[vx{i}][v{i+1}]xfade=transition=fade:duration={xfade_duration}:offset={offset}[vx{i+1}]")
                
                final_v_label = f"vx{n-1}"
            
            # Apply setpts after xfade
            parts.append(f"[{final_v_label}]setpts=PTS-STARTPTS[v]")
            
            return ";".join(parts)

        base_cmd = [
            ffmpeg_exe(),
            "-y",
            *inputs,
        ]

        # Try audio+video concat first; fall back to video-only if inputs lack audio.
        try:
            filter_complex = _filter_with_audio()
            cmd = [
                *base_cmd,
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_path),
            ]
            _run(cmd)
            return output_path
        except StitchError as exc:
            msg = str(exc)
            audio_missing_markers = (
                "matches no streams",
                "Stream specifier ':a'",
                "has not been used for any stream",
            )
            if any(m in msg for m in audio_missing_markers):
                filter_complex = _filter_video_only()
                cmd = [
                    *base_cmd,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[v]",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                ]
                _run(cmd)
                return output_path
            raise

    # Fast path: concat demuxer stream copy.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as filelist:
        for clip in clips:
            clip_abs = clip.resolve().as_posix()
            clip_abs = clip_abs.replace("'", r"\'")
            filelist.write(f"file '{clip_abs}'\n")
        list_path = Path(filelist.name)

    try:
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        _run(cmd)
        return output_path
    finally:
        list_path.unlink(missing_ok=True)
