"""
Trim screen recordings and sample frames for dataset / inference work.

Uses imageio + bundled ffmpeg (OpenCV often cannot read .mov on Windows).

Run from repo root:
    python scripts/process_video.py --info
    python scripts/process_video.py --start 2 --end 12 --every-n 37
    python scripts/process_video.py --start 0 --duration 10 --sample-fps 2 --trimmed experiments/video/clip.mp4
    python scripts/process_video.py --num-frames 30
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "experiments" / "video"


def find_default_video() -> Path:
    matches = sorted(REPO_ROOT.glob("Screen*.mov"))
    if not matches:
        raise FileNotFoundError("No Screen*.mov found in repo root.")
    return matches[0]


@dataclass
class VideoInfo:
    path: str
    fps: float
    duration_s: float
    width: int
    height: int
    frame_count_est: int


def probe_video(path: Path) -> VideoInfo:
    reader = imageio.get_reader(path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    duration = float(meta.get("duration") or 0.0)
    size = meta.get("size")
    if size is None:
        first = reader.get_data(0)
        height, width = first.shape[:2]
    else:
        width, height = int(size[0]), int(size[1])
    reader.close()
    frames = int(math.floor(duration * fps)) if duration > 0 else 0
    return VideoInfo(
        path=str(path),
        fps=fps,
        duration_s=duration,
        width=width,
        height=height,
        frame_count_est=frames,
    )


def _resolve_range(
    info: VideoInfo,
    start_s: float | None,
    end_s: float | None,
    duration_s: float | None,
) -> tuple[float, float]:
    start = 0.0 if start_s is None else max(0.0, start_s)
    if end_s is not None:
        end = min(info.duration_s, end_s)
    elif duration_s is not None:
        end = min(info.duration_s, start + duration_s)
    else:
        end = info.duration_s
    if end <= start:
        raise ValueError(f"Invalid trim range: start={start:.3f}s end={end:.3f}s")
    return start, end


def _should_keep_frame(
    frame_idx: int,
    fps: float,
    start_s: float,
    end_s: float,
    every_n: int,
    sample_fps: float | None,
    last_kept_t: float | None,
) -> tuple[bool, float | None]:
    t = frame_idx / fps
    if t < start_s or t >= end_s:
        return False, last_kept_t

    if every_n > 1 and (frame_idx % every_n) != 0:
        return False, last_kept_t

    if sample_fps is not None:
        min_gap = 1.0 / sample_fps
        if last_kept_t is not None and (t - last_kept_t) < min_gap:
            return False, last_kept_t
        return True, t

    return True, t


def equal_frame_indices(start_s: float, end_s: float, fps: float, n: int) -> list[int]:
    """Frame indices evenly spaced in time across [start_s, end_s]."""
    if n <= 0:
        return []
    if n == 1:
        return [int(round(start_s * fps))]
    times = np.linspace(start_s, end_s, n, endpoint=True)
    indices: list[int] = []
    seen: set[int] = set()
    for t in times:
        idx = int(round(t * fps))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def extract_equal_frames(
    video_path: Path,
    *,
    num_frames: int,
    start_s: float | None = None,
    end_s: float | None = None,
    duration_s: float | None = None,
    frames_dir: Path,
) -> dict:
    info = probe_video(video_path)
    start, end = _resolve_range(info, start_s, end_s, duration_s)
    targets = equal_frame_indices(start, end, info.fps, num_frames)
    frames_dir.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(video_path, "ffmpeg")
    target_set = set(targets)
    next_out = 0
    saved_times: list[float] = []

    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx not in target_set:
                continue
            out = frames_dir / f"frame_{next_out:02d}.png"
            imageio.imwrite(out, np.asarray(frame))
            saved_times.append(frame_idx / info.fps)
            next_out += 1
            if next_out >= len(targets):
                break
    finally:
        reader.close()

    return {
        "input": str(video_path),
        "trim_start_s": start,
        "trim_end_s": end,
        "num_frames": num_frames,
        "frames_saved": next_out,
        "sample_times_s": [round(t, 3) for t in saved_times],
        "frames_dir": str(frames_dir),
    }


def process_video(
    video_path: Path,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    duration_s: float | None = None,
    every_n: int = 1,
    sample_fps: float | None = None,
    frames_dir: Path | None = None,
    trimmed_path: Path | None = None,
    max_frames: int | None = None,
) -> dict:
    info = probe_video(video_path)
    start, end = _resolve_range(info, start_s, end_s, duration_s)

    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(video_path, "ffmpeg")
    writer = None
    if trimmed_path:
        trimmed_path.parent.mkdir(parents=True, exist_ok=True)
        out_fps = sample_fps or info.fps
        writer = imageio.get_writer(trimmed_path, fps=out_fps, codec="libx264", quality=8)

    saved = 0
    last_kept_t: float | None = None

    try:
        for frame_idx, frame in enumerate(reader):
            keep, last_kept_t = _should_keep_frame(
                frame_idx, info.fps, start, end, every_n, sample_fps, last_kept_t
            )
            if not keep:
                continue

            frame = np.asarray(frame)
            if frames_dir:
                out = frames_dir / f"frame_{frame_idx:06d}.png"
                imageio.imwrite(out, frame)

            if writer is not None:
                writer.append_data(frame)

            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
    finally:
        reader.close()
        if writer is not None:
            writer.close()

    return {
        "input": str(video_path),
        "trim_start_s": start,
        "trim_end_s": end,
        "every_n": every_n,
        "sample_fps": sample_fps,
        "frames_saved": saved,
        "frames_dir": str(frames_dir) if frames_dir else None,
        "trimmed_path": str(trimmed_path) if trimmed_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim video length and sample frames.")
    parser.add_argument("video", nargs="?", help="Input video path (default: Screen*.mov)")
    parser.add_argument("--info", action="store_true", help="Print metadata and exit")
    parser.add_argument("--start", type=float, default=None, help="Trim start time (seconds)")
    parser.add_argument("--end", type=float, default=None, help="Trim end time (seconds)")
    parser.add_argument("--duration", type=float, default=None, help="Trim length from --start")
    parser.add_argument("--num-frames", type=int, default=None, help="Extract N equally spaced frames")
    parser.add_argument("--every-n", type=int, default=1, help="Keep every Nth frame in range")
    parser.add_argument("--sample-fps", type=float, default=None, help="Target FPS for frame sampling")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after saving this many frames")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory for extracted PNG frames")
    parser.add_argument("--trimmed", type=Path, default=None, help="Output path for trimmed MP4")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Default output root")
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else find_default_video()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    if args.info:
        info = probe_video(video_path)
        print(json.dumps(asdict(info), indent=2))
        return

    if args.num_frames is not None:
        frames_dir = args.frames_dir or (args.out_dir / "frames")
        result = extract_equal_frames(
            video_path,
            num_frames=args.num_frames,
            start_s=args.start,
            end_s=args.end,
            duration_s=args.duration,
            frames_dir=frames_dir,
        )
        print(json.dumps(result, indent=2))
        return

    frames_dir = args.frames_dir
    trimmed = args.trimmed
    if frames_dir is None and trimmed is None:
        frames_dir = args.out_dir / "frames"
        trimmed = args.out_dir / "trimmed.mp4"

    result = process_video(
        video_path,
        start_s=args.start,
        end_s=args.end,
        duration_s=args.duration,
        every_n=max(1, args.every_n),
        sample_fps=args.sample_fps,
        frames_dir=frames_dir,
        trimmed_path=trimmed,
        max_frames=args.max_frames,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
