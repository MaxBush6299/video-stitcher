#!/usr/bin/env python
import subprocess
from pathlib import Path
from ffmpeg_bin import ffmpeg_exe

run_dir = Path("runs/2025-12-12T01-31-47Z-bear-v3-fbed76")
clip1 = run_dir / "clip_01.mp4"
clip2 = run_dir / "clip_02.mp4"
output = run_dir / "test_xfade.mp4"

# Try applying setpts AFTER xfade, not before
filter_complex = (
    "[0:v]fps=30[v0];"
    "[1:v]fps=30,select='gte(t\\,0.25)'[v1];"
    "[v0][v1]xfade=transition=smoothleft:duration=0.75:offset=11.25,setpts=PTS-STARTPTS[v]"
)

cmd = [
    ffmpeg_exe(),
    "-y",
    "-i", str(clip1),
    "-i", str(clip2),
    "-filter_complex", filter_complex,
    "-map", "[v]",
    "-t", "15",
    "-c:v", "libx264",
    "-preset", "veryfast",
    str(output)
]

print("Command:", " ".join(cmd))
print("\nFilter complex:", filter_complex)
print("\nRunning...")

result = subprocess.run(cmd, capture_output=True, text=True)
print("\nReturn code:", result.returncode)
if result.returncode != 0:
    print("\nSTDERR:")
    print(result.stderr[-2000:])  # Last 2000 chars
else:
    print("\nSuccess!")
