"""
main.py
Entry point for BLOKMARKET AUDIO.

Download audio from SoundCloud, YouTube, or Spotify and convert to OGG
with audio modifications, fingerprint removal, and auto-splitting.

Usage:
    python main.py <url>
    python main.py <url> --no-modify
    python main.py <url> --intensity moderate
    python main.py <url> --max-duration 300
    python main.py <url> --no-split
    python main.py <url> --keep-source
"""

import sys
import argparse
from src.downloader import get_downloader
from src.converter import AudioConverter
from src.filename_generator import build_filename
from src import ui

SUPPORTED_DOMAINS = (
    "soundcloud.com",
    "youtube.com",
    "youtu.be",
    "open.spotify.com",
    "spotify.link",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BLOKMARKET AUDIO — Download & convert audio to OGG."
    )
    parser.add_argument(
        "url",
        type=str,
        help="SoundCloud, YouTube, or Spotify URL",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the temp MP3 file after conversion",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for OGG files (default: ./output)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="temp",
        help="Temp directory for intermediate files (default: ./temp)",
    )
    parser.add_argument(
        "--no-modify",
        action="store_true",
        help="Skip audio modification (plain convert)",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        choices=["subtle", "moderate", "strong"],
        default="subtle",
        help="Modification intensity (default: subtle)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=360,
        help="Max seconds per split part (default: 360 = 6min)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable auto-splitting for long audio",
    )
    return parser.parse_args()


def detect_source(url: str) -> str:
    url_lower = url.lower()
    if "soundcloud.com" in url_lower:
        return "SoundCloud"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "YouTube"
    if "open.spotify.com" in url_lower or "spotify.link" in url_lower:
        return "Spotify"
    return "Unknown"


def main() -> None:
    args = parse_args()
    url = args.url.strip()

    # ── Validate URL ──
    if not any(domain in url for domain in SUPPORTED_DOMAINS):
        ui.print_banner()
        ui.print_fatal(
            "URL not supported. "
            "Use a URL from SoundCloud, YouTube, or Spotify."
        )
        sys.exit(1)

    source = detect_source(url)
    modify = not args.no_modify
    max_dur = 0 if args.no_split else args.max_duration

    # ── Banner + Config ──
    ui.print_banner()
    ui.print_config(
        source=source,
        url=url,
        output_dir=args.output_dir,
        modify=modify,
        intensity=args.intensity,
        max_dur=max_dur,
    )

    # ── Step 1: Download ──
    ui.print_step(1, 3, f"Downloading from {source}")

    try:
        downloader = get_downloader(url, temp_dir=args.temp_dir)
        download_result = downloader.download(url)
    except (ValueError, EnvironmentError) as e:
        ui.print_fatal(str(e))
        sys.exit(1)

    title = download_result["title"]
    source_filepath = download_result["filepath"]
    ui.print_info("TRACK", title)

    # ── Step 2: Generate filename ──
    ui.print_step(2, 3, "Generating filename")

    output_filename = build_filename(title)
    ui.print_info("FILENAME", f"{output_filename}.ogg")

    # ── Step 3: Convert + Modify + Split ──
    label = "Converting"
    if modify:
        label += " + modifying"
    if max_dur > 0:
        label += " + auto-split"

    ui.print_step(3, 3, label)

    converter = AudioConverter(output_dir=args.output_dir, temp_dir=args.temp_dir)

    try:
        output_paths, applied_profile = converter.to_ogg(
            source_filepath,
            output_filename,
            modify=modify,
            intensity=args.intensity,
            max_duration=max_dur,
        )
    except (FileNotFoundError, RuntimeError) as e:
        ui.print_fatal(str(e))
        sys.exit(1)

    if applied_profile:
        ui.print_mods(applied_profile.describe())

    if not args.keep_source:
        converter.cleanup_source(source_filepath)

    # ── Results ──
    ui.print_results(output_paths)
    ui.print_done()


if __name__ == "__main__":
    main()