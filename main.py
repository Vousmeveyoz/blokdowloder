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
    python main.py <url> --upload-roblox
"""

import sys
import argparse
from src.downloader import get_downloader
from src.converter import AudioConverter
from src.filename_generator import build_filename
from src.roblox_uploader import RobloxUploader
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
    parser.add_argument(
        "--upload-roblox",
        action="store_true",
        help="Prompt to upload result to Roblox after conversion",
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


def prompt_roblox_upload(output_paths: list[str], track_title: str) -> None:
    """
    Interactively ask user which files to upload to Roblox,
    then collect API key + userId and upload.
    """
    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │         UPLOAD KE ROBLOX                │")
    print("  └─────────────────────────────────────────┘")

    # Show file list
    print()
    print("  File yang tersedia:")
    for i, path in enumerate(output_paths, 1):
        from pathlib import Path
        print(f"    [{i}] {Path(path).name}")

    print()

    # Ask which files to upload
    if len(output_paths) == 1:
        choice_input = input("  Upload file ini ke Roblox? (y/n): ").strip().lower()
        if choice_input != "y":
            print("  [SKIP] Upload dibatalkan.")
            return
        selected_paths = output_paths
    else:
        print("  Masukkan nomor file yang mau diupload (pisah koma, atau 'all'):")
        choice_input = input("  Pilihan: ").strip().lower()

        if choice_input == "all":
            selected_paths = output_paths
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice_input.split(",")]
                selected_paths = [output_paths[i] for i in indices if 0 <= i < len(output_paths)]
            except (ValueError, IndexError):
                print("  [ERR] Input tidak valid. Upload dibatalkan.")
                return

        if not selected_paths:
            print("  [SKIP] Tidak ada file dipilih.")
            return

    # Collect credentials
    import getpass
    print()
    api_key = getpass.getpass("  Masukkan Roblox API Key: ").strip()
    if not api_key:
        print("  [ERR] API key tidak boleh kosong.")
        return

    user_id = input("  Masukkan Roblox User ID: ").strip()
    if not user_id:
        print("  [ERR] User ID tidak boleh kosong.")
        return

    # Mask API key confirmation
    masked = api_key[:6] + "*" * max(0, len(api_key) - 6)
    print(f"  API Key  : {masked}")
    print(f"  User ID  : {user_id}")

    # Upload each selected file
    uploader = RobloxUploader(api_key=api_key, user_id=user_id)

    print()
    print(f"  Mengupload {len(selected_paths)} file...")
    print()

    success_count = 0
    for i, filepath in enumerate(selected_paths, 1):
        from pathlib import Path

        if len(selected_paths) == 1:
            display_name = track_title[:50]
        else:
            display_name = f"{track_title[:44]} pt{i}"

        print(f"  [{i}/{len(selected_paths)}] Uploading: {display_name}")

        # Poll terus sampai status final (approved/rejected) — timeout 24 jam
        result = uploader.upload(
            filepath=filepath,
            display_name=display_name,
            wait_moderation=True,
            moderation_timeout=86400,
        )

        if result["success"]:
            asset_id  = result.get("asset_id") or "N/A"
            asset_url = result.get("asset_url") or ""
            mod_state = result.get("moderation_state") or "Reviewing"
            mod_note  = result.get("moderation_note")

            if mod_state == "Approved":
                success_count += 1
                print(f"  [OK] Asset ID   : {asset_id}")
                if asset_url:
                    print(f"       rbxassetid  : {asset_url}")
                print(f"       Moderasi    : [APPROVED] Siap dipakai!")
            elif mod_state == "Rejected":
                print(f"  [REJECTED] Asset ID : {asset_id}")
                print(f"             Moderasi : [REJECTED] Asset ditolak Roblox.")
                if mod_note:
                    print(f"             Alasan   : {mod_note}")
            else:
                success_count += 1
                print(f"  [?] Asset ID    : {asset_id}")
                print(f"      Moderasi    : [REVIEWING] Cek manual di Creator Dashboard.")
        else:
            print(f"  [ERR] {result['error']}")

    # Summary
    print()
    print(f"  Upload selesai: {success_count}/{len(selected_paths)} berhasil.")
    print()


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

    # ── Step 4: Upload ke Roblox (manual prompt) ──
    if args.upload_roblox:
        prompt_roblox_upload(output_paths, title)

    ui.print_done()


if __name__ == "__main__":
    main()
