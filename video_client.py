from __future__ import annotations

import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from utils import normalize_azure_openai_endpoint
from ffmpeg_bin import ffmpeg_exe

DEFAULT_TIMEOUT = 10
POLL_INTERVAL = 10
SUPPORTED_SORA2_SECONDS = (4, 8, 12)


class VideoClient:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mock: bool = False,
        mock_source: Optional[Path] = None,
    ) -> None:
        import os

        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        # Prefer OpenAI's standard env var, but keep backward-compatible Azure var.
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        self.model = model or os.getenv("AZURE_OPENAI_VIDEO_MODEL")
        self.mock = mock
        self.mock_source = mock_source or Path("samples/mock_clip.mp4")

        self.base_url = os.getenv("OPENAI_BASE_URL")
        if not self.base_url and self.endpoint:
            self.base_url = normalize_azure_openai_endpoint(self.endpoint).rstrip("/") + "/openai/v1/"

        if not self.mock and not all([self.base_url, self.api_key, self.model]):
            raise ValueError(
                "Missing video configuration. Set OPENAI_BASE_URL + OPENAI_API_KEY (recommended) "
                "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY, plus AZURE_OPENAI_VIDEO_MODEL (e.g., 'sora-2')."
            )

        self._client = None

    def _client_or_raise(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Missing dependency 'openai'. Run: pip install -r requirements.txt") from exc

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # Sanity check: older clients may not have the videos API.
        if not hasattr(self._client, "videos"):
            raise RuntimeError("Your 'openai' package is too old; upgrade it (pip install --upgrade openai).")
        return self._client

    @staticmethod
    def _map_aspect_to_size(aspect_ratio: str) -> str:
        raw = (aspect_ratio or "").strip().lower().replace(" ", "")
        if raw in {"16:9", "16/9"}:
            return "1280x720"
        if raw in {"9:16", "9/16"}:
            return "720x1280"
        # Allow users to pass sizes directly.
        if "x" in raw and all(p.isdigit() for p in raw.split("x") if p):
            return raw
        # Default per docs is portrait.
        return "720x1280"

    @staticmethod
    def _map_seconds(seconds: int) -> int:
        try:
            s = int(seconds)
        except Exception:
            s = 4
        return min(SUPPORTED_SORA2_SECONDS, key=lambda v: abs(v - s))

    def generate_clip(
        self,
        prompt: str,
        output_path: Path,
        duration: int,
        aspect_ratio: str,
        seed: Optional[int] = None,
        init_image: Optional[Path] = None,
        retries: int = 2,
        backoff: float = 2.0,
    ) -> Path:
        if self.mock:
            return self._mock_clip(output_path)

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= retries:
            try:
                path, _video_id = self._generate_clip_once(prompt, output_path, duration, aspect_ratio, seed, init_image)
                return path
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                attempt += 1
                if attempt > retries:
                    break
                time.sleep(backoff * attempt)
        raise RuntimeError(f"Failed to render clip after {retries + 1} attempts: {last_err}")

    def generate_clip_with_id(
        self,
        prompt: str,
        output_path: Path,
        duration: int,
        aspect_ratio: str,
        seed: Optional[int] = None,
        init_image: Optional[Path] = None,
        retries: int = 2,
        backoff: float = 2.0,
    ) -> Tuple[Path, str]:
        """Generate a clip and return (output_path, video_id)."""
        if self.mock:
            return self._mock_clip(output_path), "mock_video"

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= retries:
            try:
                return self._generate_clip_once(prompt, output_path, duration, aspect_ratio, seed, init_image)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                attempt += 1
                if attempt > retries:
                    break
                time.sleep(backoff * attempt)
        raise RuntimeError(f"Failed to render clip after {retries + 1} attempts: {last_err}")

    def _generate_clip_once(
        self,
        prompt: str,
        output_path: Path,
        duration: int,
        aspect_ratio: str,
        seed: Optional[int],
        init_image: Optional[Path],
    ) -> Tuple[Path, str]:
        client = self._client_or_raise()
        size = self._map_aspect_to_size(aspect_ratio)
        seconds = self._map_seconds(duration)

        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "seconds": str(seconds),
        }
        # Sora 2 docs don't list seed; omit to avoid 400s.
        _ = seed

        ref_handle = None
        try:
            if init_image:
                ref_handle = init_image.open("rb")
                kwargs["input_reference"] = ref_handle

            video = client.videos.create(**kwargs)
            video_id = getattr(video, "id", None)
            if not video_id:
                raise RuntimeError("Video create did not return an id")

            # Poll until completion.
            while getattr(video, "status", None) not in {"completed", "failed", "cancelled"}:
                time.sleep(POLL_INTERVAL)
                video = client.videos.retrieve(video_id)

            if video.status != "completed":
                err = getattr(video, "error", None)
                raise RuntimeError(f"Video generation ended with status={video.status}; error={err}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            content = client.videos.download_content(video_id, variant="video")
            content.write_to_file(str(output_path))
            return output_path, str(video_id)
        finally:
            try:
                if ref_handle is not None:
                    ref_handle.close()
            except Exception:
                pass

    def remix_video(
        self,
        remix_video_id: str,
        prompt: str,
        output_path: Path,
        retries: int = 2,
        backoff: float = 2.0,
    ) -> Tuple[Path, str]:
        """Remix a completed video and return (output_path, new_video_id)."""
        if self.mock:
            return self._mock_clip(output_path), "mock_remix_video"

        client = self._client_or_raise()

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= retries:
            try:
                video = client.videos.remix(remix_video_id, prompt=prompt)
                new_id = getattr(video, "id", None)
                if not new_id:
                    raise RuntimeError("Remix did not return an id")

                while getattr(video, "status", None) not in {"completed", "failed", "cancelled"}:
                    time.sleep(POLL_INTERVAL)
                    video = client.videos.retrieve(new_id)

                if video.status != "completed":
                    err = getattr(video, "error", None)
                    raise RuntimeError(f"Video remix ended with status={video.status}; error={err}")

                output_path.parent.mkdir(parents=True, exist_ok=True)
                content = client.videos.download_content(new_id, variant="video")
                content.write_to_file(str(output_path))
                return output_path, str(new_id)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                attempt += 1
                if attempt > retries:
                    break
                time.sleep(backoff * attempt)
        raise RuntimeError(f"Failed to remix video after {retries + 1} attempts: {last_err}")

    def _mock_clip(self, output_path: Path) -> Path:
        if not self.mock_source.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Create a small, valid MP4 so ffmpeg frame extraction works.
            try:
                cmd = [
                    ffmpeg_exe(),
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=1280x720:r=30",
                    "-t",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    return output_path
            except Exception:
                pass

            # Last resort: write a tiny stub (may not be playable everywhere).
            with output_path.open("wb") as f:
                f.write(b"\x00\x00\x00\x18ftypmp42")
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.mock_source, output_path)
        return output_path
