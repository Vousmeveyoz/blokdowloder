"""
downloader.py
Handles downloading audio from SoundCloud, YouTube, and Spotify.

- SoundCloud & YouTube : via yt-dlp (direct audio stream)
- Spotify              : via spotdl (matches Spotify track → YouTube → download)

Install requirements:
    pip install yt-dlp spotdl
    
Optional (for richer Spotify metadata):
    pip install spotipy
"""

import sys
import re
import subprocess
import shutil
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Base Downloader
# ═════════════════════════════════════════════════════════════════════════════


class BaseDownloader:
    """Base class for all audio downloaders."""

    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> dict:
        """
        Download audio from URL.

        Args:
            url: Track/video URL

        Returns:
            dict with 'title', 'filepath', and optional 'metadata' keys

        Raises:
            ValueError: If download fails
        """
        raise NotImplementedError

    def download_many(self, urls: list[str]) -> list[dict]:
        """
        Download multiple URLs sequentially.

        Args:
            urls: List of track URLs

        Returns:
            List of result dicts; failed downloads include an 'error' key instead of 'filepath'.
        """
        results = []
        for url in urls:
            try:
                result = self.download(url)
                results.append(result)
            except ValueError as e:
                results.append({"url": url, "error": str(e)})
        return results


# ═════════════════════════════════════════════════════════════════════════════
# YtdlpDownloader (SoundCloud + YouTube)
# ═════════════════════════════════════════════════════════════════════════════


class YtdlpDownloader(BaseDownloader):
    """Downloads audio from SoundCloud and YouTube URLs using yt-dlp."""

    def _progress_hook(self, d: dict) -> None:
        """Display live download progress in terminal."""
        try:
            from .ui import print_progress, print_progress_done
        except ImportError:
            # Fallback if UI module is not available
            if d["status"] == "downloading":
                downloaded = d.get("downloaded_bytes", 0)
                total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
                if total:
                    pct = downloaded / total * 100
                    speed = d.get("speed", 0) or 0
                    speed_kb = speed / 1024
                    print(f"\r  {pct:5.1f}%  {speed_kb:.0f} KB/s", end="", flush=True)
            elif d["status"] == "finished":
                print()
            return

        if d["status"] == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
            speed = d.get("speed", 0) or 0
            eta = d.get("eta", 0) or 0
            print_progress(downloaded, total, speed, eta)
        elif d["status"] == "finished":
            print_progress_done()

    def download(self, url: str) -> dict:
        """
        Download audio from a SoundCloud or YouTube URL.

        Returns:
            dict with 'title', 'filepath', and 'metadata' keys
        """
        import yt_dlp

        before_files = set(self.temp_dir.glob("*.mp3"))

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.temp_dir / "%(title)s.%(ext)s"),
            "restrictfilenames": False,
            "windowsfilenames": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [self._progress_hook],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "Unknown Title")
                metadata = {
                    "uploader": info.get("uploader"),
                    "duration": info.get("duration"),
                    "thumbnail": info.get("thumbnail"),
                    "upload_date": info.get("upload_date"),
                    "view_count": info.get("view_count"),
                }
        except yt_dlp.utils.DownloadError as e:
            raise ValueError(f"Gagal download: {e}")

        after_files = set(self.temp_dir.glob("*.mp3"))
        new_files = after_files - before_files

        if not new_files:
            # Fallback: ambil file MP3 paling baru di temp dir
            all_mp3 = list(self.temp_dir.glob("*.mp3"))
            if not all_mp3:
                raise ValueError(
                    "Download selesai tapi file MP3 tidak ditemukan di folder temp. "
                    "Pastikan ffmpeg terinstall dengan benar."
                )
            # Gunakan file yang paling baru dimodifikasi
            new_files = {max(all_mp3, key=lambda f: f.stat().st_mtime)}

        downloaded_file = max(new_files, key=lambda f: f.stat().st_mtime)

        return {
            "title": title,
            "filepath": str(downloaded_file),
            "metadata": metadata,
        }


# Backward compatibility alias
SoundCloudDownloader = YtdlpDownloader


# ═════════════════════════════════════════════════════════════════════════════
# SpotifyDownloader
# ═════════════════════════════════════════════════════════════════════════════


class SpotifyDownloader(BaseDownloader):
    """
    Downloads audio from Spotify URLs using spotdl.

    spotdl matches the Spotify track metadata (title, artist) to a
    YouTube video, then downloads the audio from YouTube.

    Install: pip install spotdl

    Supports:
      - Single tracks    : https://open.spotify.com/track/...
      - Albums           : https://open.spotify.com/album/...
      - Playlists        : https://open.spotify.com/playlist/...

    Note:
      Albums/playlists download ALL tracks. Use download_many() for
      multiple URLs, or download() for a single track URL.
    """

    # Spotify URL type detection
    _URL_PATTERNS = {
        "track":    re.compile(r"spotify\.com/track/"),
        "album":    re.compile(r"spotify\.com/album/"),
        "playlist": re.compile(r"spotify\.com/playlist/"),
    }

    def __init__(
        self,
        temp_dir: str = "temp",
        bitrate: str = "192k",
        audio_format: str = "mp3",
        max_retries: int = 2,
        timeout: int = 300,
    ):
        """
        Args:
            temp_dir:     Directory for downloaded files.
            bitrate:      Audio bitrate (e.g. '128k', '192k', '320k').
            audio_format: Output format ('mp3', 'm4a', 'flac', 'ogg').
            max_retries:  Number of retry attempts on failure.
            timeout:      Process timeout in seconds.
        """
        super().__init__(temp_dir)
        self.bitrate = bitrate
        self.audio_format = audio_format
        self.max_retries = max_retries
        self.timeout = timeout

    # ── helpers ──────────────────────────────────────────────────────────────

    def _check_spotdl(self) -> None:
        """Ensure spotdl is installed. Raises ValueError if not found."""
        if shutil.which("spotdl") is None:
            raise ValueError(
                "spotdl tidak ditemukan.\n"
                "Install dulu:\n"
                "  pip install spotdl\n\n"
                "spotdl juga butuh ffmpeg terinstall."
            )

    def _check_ffmpeg(self) -> None:
        """Ensure ffmpeg is available. Raises ValueError if not found."""
        if shutil.which("ffmpeg") is None:
            raise ValueError(
                "ffmpeg tidak ditemukan.\n"
                "Install ffmpeg:\n"
                "  Windows : https://ffmpeg.org/download.html\n"
                "  macOS   : brew install ffmpeg\n"
                "  Linux   : sudo apt install ffmpeg"
            )

    def _detect_url_type(self, url: str) -> str:
        """
        Detect Spotify URL type.

        Returns:
            'track', 'album', 'playlist', or 'unknown'
        """
        for url_type, pattern in self._URL_PATTERNS.items():
            if pattern.search(url):
                return url_type
        return "unknown"

    def _build_command(self, url: str) -> list[str]:
        """Build the spotdl subprocess command."""
        return [
            "spotdl", "download", url,
            "--output", str(self.temp_dir),
            "--format", self.audio_format,
            "--bitrate", self.bitrate,
            "--print-errors",
        ]

    def _run_spotdl(self, url: str) -> subprocess.CompletedProcess:
        """
        Run spotdl with retry logic.

        Returns:
            CompletedProcess result

        Raises:
            ValueError on timeout or if all retries fail
        """
        command = self._build_command(url)
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                print(f"  Retry {attempt}/{self.max_retries}...")

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                raise ValueError(
                    f"spotdl timeout setelah {self.timeout} detik. "
                    "Coba lagi atau periksa koneksi internet."
                )
            except FileNotFoundError:
                raise ValueError("spotdl tidak ditemukan. Install: pip install spotdl")

            if result.returncode == 0:
                return result

            last_error = result.stderr.strip() or result.stdout.strip()
            logger.warning("spotdl attempt %d failed: %s", attempt, last_error[:200])

        # All retries exhausted
        if len(last_error) > 400:
            last_error = last_error[:400] + "..."
        raise ValueError(f"spotdl gagal setelah {self.max_retries} percobaan:\n{last_error}")

    def _collect_new_files(self, before_files: set[Path]) -> list[Path]:
        """Return list of new audio files created since before_files snapshot."""
        pattern = f"*.{self.audio_format}"
        after_files = set(self.temp_dir.glob(pattern))
        new_files = sorted(after_files - before_files, key=lambda f: f.stat().st_mtime)
        return new_files

    @staticmethod
    def _parse_title(filepath: Path) -> str:
        """
        Extract a clean title from filename.

        spotdl naming format: "Artist - Title.mp3"
        Returns the full stem as title (preserves artist info).
        """
        return filepath.stem

    # ── public interface ──────────────────────────────────────────────────────

    def download(self, url: str) -> dict:
        """
        Download a single Spotify track.

        For albums/playlists, only the first downloaded track is returned.
        Use download_playlist() for full album/playlist support.

        Args:
            url: Spotify track/album/playlist URL

        Returns:
            dict with keys:
              - title    (str)  : track title (from filename)
              - filepath (str)  : absolute path to downloaded file
              - url_type (str)  : 'track', 'album', 'playlist', or 'unknown'

        Raises:
            ValueError: On download failure or dependency missing
        """
        self._check_spotdl()
        self._check_ffmpeg()

        url_type = self._detect_url_type(url)
        pattern = f"*.{self.audio_format}"
        before_files = set(self.temp_dir.glob(pattern))

        print(f"  spotdl: fetching [{url_type}] from Spotify + matching YouTube...")

        self._run_spotdl(url)

        new_files = self._collect_new_files(before_files)

        if not new_files:
            raise ValueError(
                "spotdl selesai tapi file audio tidak ditemukan.\n"
                "Kemungkinan penyebab:\n"
                "  - Track tidak tersedia di YouTube\n"
                "  - Spotify API rate limit\n"
                "  - URL tidak valid atau konten dihapus"
            )

        downloaded_file = new_files[-1]  # most recent
        title = self._parse_title(downloaded_file)

        print(f"  [OK] {title}")

        return {
            "title": title,
            "filepath": str(downloaded_file),
            "url_type": url_type,
        }

    def download_playlist(self, url: str) -> list[dict]:
        """
        Download all tracks from a Spotify album or playlist.

        Args:
            url: Spotify album or playlist URL

        Returns:
            List of dicts, each with 'title' and 'filepath' keys.
            On partial failure, returns whatever was successfully downloaded.

        Raises:
            ValueError: If no tracks were downloaded at all
        """
        self._check_spotdl()
        self._check_ffmpeg()

        url_type = self._detect_url_type(url)
        if url_type == "track":
            # Gracefully handle single tracks passed to this method
            return [self.download(url)]

        pattern = f"*.{self.audio_format}"
        before_files = set(self.temp_dir.glob(pattern))

        print(f"  spotdl: downloading full [{url_type}]...")

        self._run_spotdl(url)

        new_files = self._collect_new_files(before_files)

        if not new_files:
            raise ValueError(
                f"spotdl selesai tapi tidak ada file audio yang diunduh dari {url_type}."
            )

        results = []
        for f in new_files:
            title = self._parse_title(f)
            print(f"  [OK] {title}")
            results.append({
                "title": title,
                "filepath": str(f),
                "url_type": url_type,
            })

        print(f"\n  Total: {len(results)} track(s) downloaded.")
        return results


# ═════════════════════════════════════════════════════════════════════════════
# Factory function
# ═════════════════════════════════════════════════════════════════════════════


def get_downloader(
    url: str,
    temp_dir: str = "temp",
    spotify_bitrate: str = "192k",
    spotify_format: str = "mp3",
) -> BaseDownloader:
    """
    Return the appropriate downloader based on URL.

    Args:
        url:              Track URL
        temp_dir:         Temporary directory for downloads
        spotify_bitrate:  Bitrate for Spotify downloads (default '192k')
        spotify_format:   Audio format for Spotify downloads (default 'mp3')

    Returns:
        BaseDownloader instance (YtdlpDownloader or SpotifyDownloader)

    Raises:
        ValueError: If URL is not from a supported platform
    """
    url_lower = url.lower()

    if "open.spotify.com" in url_lower or "spotify.link" in url_lower:
        return SpotifyDownloader(
            temp_dir=temp_dir,
            bitrate=spotify_bitrate,
            audio_format=spotify_format,
        )

    if any(domain in url_lower for domain in ("soundcloud.com", "youtube.com", "youtu.be")):
        return YtdlpDownloader(temp_dir=temp_dir)

    raise ValueError(
        "URL tidak didukung.\n"
        "Platform yang didukung: SoundCloud, YouTube, Spotify\n\n"
        f"URL diterima: {url}"
    )
