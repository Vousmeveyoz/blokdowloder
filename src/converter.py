"""
converter.py
Handles converting audio files to OGG format using ffmpeg directly (via subprocess).
Includes audio modification and automatic splitting for long audio.
"""

import shutil
import subprocess
from pathlib import Path
from .audio_modifier import AudioModifier, ModificationProfile
from .splitter import AudioSplitter


class AudioConverter:
    """Converts audio files to OGG format with optional modification and splitting."""

    def __init__(self, output_dir: str = "output", temp_dir: str = "temp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_ffmpeg()
        self.modifier = AudioModifier()
        self.splitter = AudioSplitter(temp_dir=temp_dir)

    def _check_ffmpeg(self) -> None:
        """Ensure ffmpeg is installed and accessible in PATH."""
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError(
                "ffmpeg tidak ditemukan di PATH.\n"
                "Install ffmpeg dulu:\n"
                "  Windows : winget install ffmpeg\n"
                "  macOS   : brew install ffmpeg\n"
                "  Linux   : sudo apt install ffmpeg"
            )

    def _strip_metadata(self, filepath: str) -> None:
        """
        Strip semua metadata dari file output menggunakan ffmpeg.
        Ini memastikan tidak ada judul/artist asli yang tersisa.
        """
        path = Path(filepath)
        temp_path = path.with_suffix(".tmp" + path.suffix)

        command = [
            "ffmpeg",
            "-i", str(path),
            "-map_metadata", "-1",           # hapus semua global metadata
            "-metadata", "title=",            # kosongkan title
            "-metadata", "artist=",           # kosongkan artist
            "-metadata", "album=",            # kosongkan album
            "-metadata", "comment=",          # kosongkan comment
            "-metadata", "genre=",            # kosongkan genre
            "-metadata", "date=",             # kosongkan date
            "-metadata", "track=",            # kosongkan track number
            "-metadata", "encoder=",          # kosongkan encoder info
            "-c:a", "copy",                   # copy audio tanpa re-encode
            "-y",
            str(temp_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Replace original dengan versi bersih
            temp_path.replace(path)
        except subprocess.CalledProcessError:
            # Kalau gagal, hapus temp file dan lanjut (file tetap ada tapi metadata mungkin masih ada)
            if temp_path.exists():
                temp_path.unlink()

    def _convert_single(
        self,
        input_filepath: str,
        output_filename: str,
        modify: bool = True,
        intensity: str = "subtle",
        profile: ModificationProfile | None = None,
    ) -> tuple[str, ModificationProfile | None]:
        """Convert a single file to OGG with optional modification."""
        input_path = Path(input_filepath)

        if not input_path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {input_filepath}")

        output_path = self.output_dir / f"{output_filename}.ogg"

        if modify:
            if profile is None:
                profile = ModificationProfile.random(intensity=intensity)

            self.modifier.modify(
                input_path=str(input_path),
                output_path=str(output_path),
                profile=profile,
            )
        else:
            command = [
                "ffmpeg",
                "-i", str(input_path),
                "-map_metadata", "-1",
                "-metadata", "title=",
                "-metadata", "artist=",
                "-metadata", "album=",
                "-metadata", "comment=",
                "-metadata", "genre=",
                "-metadata", "date=",
                "-metadata", "track=",
                "-metadata", "encoder=",
                "-c:a", "libvorbis",
                "-q:a", "4",
                "-y",
                str(output_path),
            ]

            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg gagal convert: {e}")

        # FIX: Selalu strip metadata setelah convert (baik modify maupun plain)
        # untuk memastikan tidak ada judul/artist asli yang tersisa
        self._strip_metadata(str(output_path))

        return str(output_path), profile

    def to_ogg(
        self,
        input_filepath: str,
        output_filename: str,
        modify: bool = True,
        intensity: str = "subtle",
        profile: ModificationProfile | None = None,
        max_duration: int = 360,
    ) -> tuple[list[str], ModificationProfile | None]:
        """
        Convert audio to OGG, with optional modification and splitting.

        If audio exceeds max_duration, it's split into parts first,
        then each part is converted + modified individually.

        Args:
            input_filepath:  Source audio file path
            output_filename: Base output filename (without extension)
            modify:          Whether to apply audio modifications
            intensity:       Modification intensity: "subtle", "moderate", "strong"
            profile:         Custom ModificationProfile, or None for random
            max_duration:    Max seconds per part (default: 360 = 6 min)
                             Set to 0 to disable splitting

        Returns:
            tuple of (list_of_output_paths, applied_profile or None)
        """
        input_path = Path(input_filepath)
        if not input_path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {input_filepath}")

        # Check if splitting is needed
        if max_duration > 0:
            parts = self.splitter.split(input_filepath, max_duration=max_duration)
        else:
            parts = [input_filepath]

        is_split = len(parts) > 1

        # Generate one profile for all parts (consistency across parts)
        if modify and profile is None:
            profile = ModificationProfile.random(intensity=intensity)

        output_paths = []

        for i, part_path in enumerate(parts):
            if is_split:
                part_filename = f"{output_filename}_part{i + 1}"
            else:
                part_filename = output_filename

            output_path, _ = self._convert_single(
                input_filepath=part_path,
                output_filename=part_filename,
                modify=modify,
                intensity=intensity,
                profile=profile,
            )
            output_paths.append(output_path)

        # Cleanup split temp files (not the original source)
        if is_split:
            self.splitter.cleanup_parts(parts)

        return output_paths, profile

    def cleanup_source(self, filepath: str) -> None:
        """Remove the source file after conversion."""
        path = Path(filepath)
        if path.exists():
            path.unlink()
