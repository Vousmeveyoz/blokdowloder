"""
splitter.py
Splits long audio files into multiple parts based on max duration.

Features:
  - Split by max duration (default: 6 minutes per part)
  - Uses ffmpeg segment muxer for precise, fast splitting
  - Lossless split on source (before conversion) to avoid re-encoding twice
  - Automatic part numbering: filename_part1, filename_part2, etc.
  - Skips splitting if audio is already shorter than max duration

Usage:
    splitter = AudioSplitter(temp_dir="temp")

    # Returns list of file paths (1 item if no split needed)
    parts = splitter.split("long_mix.mp3", max_duration=360)
"""

import json
import math
import subprocess
import shutil
from pathlib import Path


class AudioSplitter:
    """Splits audio files into parts based on maximum duration."""

    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("ffmpeg") is None:
            raise EnvironmentError("ffmpeg tidak ditemukan di PATH.")
        if shutil.which("ffprobe") is None:
            raise EnvironmentError("ffprobe tidak ditemukan di PATH.")

    def get_duration(self, filepath: str) -> float:
        """Get audio duration in seconds."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    def split(
        self,
        input_filepath: str,
        max_duration: int = 360,
    ) -> list[str]:
        """
        Split audio file into parts if it exceeds max_duration.

        Args:
            input_filepath:  Source audio file path
            max_duration:    Maximum duration per part in seconds (default: 360 = 6 min)

        Returns:
            List of file paths. If no split needed, returns [input_filepath].
            Split files are saved in temp_dir as: {stem}_part{N}{ext}
        """
        input_path = Path(input_filepath)
        if not input_path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {input_filepath}")

        duration = self.get_duration(input_filepath)

        # No split needed
        if duration <= max_duration:
            return [input_filepath]

        num_parts = math.ceil(duration / max_duration)
        stem = input_path.stem
        ext = input_path.suffix  # .mp3, .wav, etc.

        print(f"  ✂️  Audio {duration:.0f}s ({duration/60:.1f} min) → {num_parts} parts @ {max_duration}s max")

        # Use ffmpeg segment muxer for clean splitting
        # -c copy = no re-encoding (fast & lossless)
        # -f segment = split into segments
        # -segment_time = max duration per segment
        # -reset_timestamps 1 = each part starts at t=0
        output_pattern = str(self.temp_dir / f"{stem}_part%d{ext}")

        command = [
            "ffmpeg",
            "-i", str(input_path),
            "-f", "segment",
            "-segment_time", str(max_duration),
            "-c", "copy",
            "-reset_timestamps", "1",
            "-y",
            output_pattern,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg gagal split audio: {e}")

        # Collect output files (segment muxer uses 0-indexed numbering)
        parts = []
        for i in range(num_parts + 5):  # extra buffer in case of rounding
            part_path = self.temp_dir / f"{stem}_part{i}{ext}"
            if part_path.exists():
                part_duration = self.get_duration(str(part_path))
                parts.append(str(part_path))
                print(f"       Part {i + 1}: {part_duration:.1f}s ({part_duration/60:.1f} min)")

        if not parts:
            raise RuntimeError(
                "Split selesai tapi tidak ada file output ditemukan."
            )

        return parts

    def cleanup_parts(self, filepaths: list[str]) -> None:
        """Remove split part files after they've been processed."""
        for fp in filepaths:
            path = Path(fp)
            if path.exists():
                path.unlink()