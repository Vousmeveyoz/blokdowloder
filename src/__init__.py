"""
src/__init__.py
BLOKMARKET AUDIO — SoundCloud, YouTube & Spotify -> OGG converter.
"""

from .downloader import (
    YtdlpDownloader,
    SoundCloudDownloader,
    SpotifyDownloader,
    get_downloader,
)
from .converter import AudioConverter
from .filename_generator import build_filename
from .audio_modifier import AudioModifier, ModificationProfile
from .splitter import AudioSplitter
from . import ui

__all__ = [
    "YtdlpDownloader",
    "SoundCloudDownloader",
    "SpotifyDownloader",
    "get_downloader",
    "AudioConverter",
    "AudioModifier",
    "ModificationProfile",
    "AudioSplitter",
    "build_filename",
    "ui",
]