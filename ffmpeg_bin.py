from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Return an ffmpeg executable path.

    Prefers a system-installed ffmpeg if present; otherwise falls back to a bundled
    ffmpeg provided by `imageio-ffmpeg`.
    """

    system = shutil.which("ffmpeg")
    if system:
        return system

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or add `imageio-ffmpeg` and ensure it can download its bundled binary."
        ) from exc
