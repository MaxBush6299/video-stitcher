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
        # Since Sora 2 continues from last frame, skip the duplicate first frame of subsequent clips
        target_fps = "30"
        n = len(clips)
        skip_frames = 2  # Skip first 2 frames of subsequent clips for smoother transition
        
        inputs: List[str] = []
        for clip in clips:
            inputs += ["-i", str(clip)]

        def _filter_with_audio() -> str:
            parts: List[str] = []
            
            # Preprocess all video clips - skip duplicate frames
            v_labels: List[str] = []
            a_labels: List[str] = []
            for i in range(n):
                v_out = f"v{i}"
                a_out = f"a{i}"
                v_labels.append(f"[{v_out}]")
                a_labels.append(f"[{a_out}]")

                if i == 0:
                    # First clip: just normalize
                    parts.append(
                        f"[{i}:v:0]scale=iw:ih:force_original_aspect_ratio=decrease,setsar=1,fps={target_fps}[{v_out}]"
                    )
                    parts.append(f"[{i}:a:0]asetpts=PTS-STARTPTS[{a_out}]")
                else:
                    # Subsequent clips: skip first N frames (duplicates from continuity)
                    parts.append(
                        f"[{i}:v:0]scale=iw:ih:force_original_aspect_ratio=decrease,setsar=1,fps={target_fps},"
                        f"select='gte(n\\,{skip_frames})',setpts=PTS-STARTPTS[{v_out}]"
                    )
                    parts.append(f"[{i}:a:0]atrim=start={skip_frames/30},asetpts=PTS-STARTPTS[{a_out}]")

            # Simple concat - duplicate frames already skipped
            if n == 1:
                final_v_label = "v0"
            else:
                concat_inputs = "".join([f"[v{i}]" for i in range(n)])
                parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vtmp]")
                final_v_label = "vtmp"
            
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
            
            # Preprocess all clips - skip duplicate first frames on subsequent clips
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
                        f"select='gte(n\\,{skip_frames})',setpts=PTS-STARTPTS[{v_out}]"
                    )
            
            # Simple concat - duplicates already skipped
            if n == 1:
                final_v_label = "v0"
            else:
                concat_inputs = "".join([f"[v{i}]" for i in range(n)])
                parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vtmp]")
                final_v_label = "vtmp"
            
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
