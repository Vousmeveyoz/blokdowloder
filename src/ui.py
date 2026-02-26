"""
ui.py
Centralized terminal UI for BLOKMARKET AUDIO.

All terminal output goes through this module for consistent styling.
No emojis — uses ASCII/Unicode box-drawing characters only.
"""

import sys
import shutil

# ── Theme constants ─────────────────────────────────────────────────────

# Try to detect terminal width, fallback to 62
try:
    TERM_WIDTH = min(shutil.get_terminal_size().columns, 72)
except Exception:
    TERM_WIDTH = 62

W = TERM_WIDTH

# Box drawing chars
TL = "+"   # top-left
TR = "+"   # top-right
BL = "+"   # bottom-left
BR = "+"   # bottom-right
H  = "-"   # horizontal
V  = "|"   # vertical
DH = "="   # double horizontal


# ── ASCII Banner ────────────────────────────────────────────────────────

BANNER = r"""
██████╗ ██╗      ██████╗ ██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗████████╗    
██╔══██╗██║     ██╔═══██╗██║ ██╔╝████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝    
██████╔╝██║     ██║   ██║█████╔╝ ██╔████╔██║███████║██████╔╝█████╔╝ █████╗     ██║       
██╔══██╗██║     ██║   ██║██╔═██╗ ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝     ██║       
██████╔╝███████╗╚██████╔╝██║  ██╗██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████╗   ██║       
╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝       
                                                                                         
                         █████╗ ██╗   ██╗██████╗ ██╗ ██████╗                             
                        ██╔══██╗██║   ██║██╔══██╗██║██╔═══██╗                            
                        ███████║██║   ██║██║  ██║██║██║   ██║                            
                        ██╔══██║██║   ██║██║  ██║██║██║   ██║                            
                        ██║  ██║╚██████╔╝██████╔╝██║╚██████╔╝                            
                        ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝                             
                                                                                                       
"""

BANNER_SLIM = r"""
██████╗ ██╗      ██████╗ ██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗████████╗    
██╔══██╗██║     ██╔═══██╗██║ ██╔╝████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝    
██████╔╝██║     ██║   ██║█████╔╝ ██╔████╔██║███████║██████╔╝█████╔╝ █████╗     ██║       
██╔══██╗██║     ██║   ██║██╔═██╗ ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝     ██║       
██████╔╝███████╗╚██████╔╝██║  ██╗██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████╗   ██║       
╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝       
                                                                                         
                         █████╗ ██╗   ██╗██████╗ ██╗ ██████╗                             
                        ██╔══██╗██║   ██║██╔══██╗██║██╔═══██╗                            
                        ███████║██║   ██║██║  ██║██║██║   ██║                            
                        ██╔══██║██║   ██║██║  ██║██║██║   ██║                            
                        ██║  ██║╚██████╔╝██████╔╝██║╚██████╔╝                            
                        ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝                             
                                                                 
"""


# ── Helpers ─────────────────────────────────────────────────────────────

def _box_line(text: str = "", pad: int = 2) -> str:
    """Create a line inside a box: | text          |"""
    inner = W - 2 - (pad * 2)
    content = text[:inner].ljust(inner)
    return f"{V}{' ' * pad}{content}{' ' * pad}{V}"


def _box_top() -> str:
    return f"{TL}{H * (W - 2)}{TR}"


def _box_bottom() -> str:
    return f"{BL}{H * (W - 2)}{BR}"


def _box_divider(char: str = "-") -> str:
    return f"{V}{char * (W - 2)}{V}"


def _center(text: str) -> str:
    return text.center(W)


# ── Public API ──────────────────────────────────────────────────────────

def print_banner():
    """Print the main ASCII art banner."""
    print()
    for line in BANNER_SLIM.strip().splitlines():
        print(_center(line))
    print(_center("[ SoundCloud / YouTube / Spotify -> OGG ]"))
    print(_center(f"v2.0"))
    print()


def print_config(source: str, url: str, output_dir: str,
                 modify: bool, intensity: str,
                 max_dur: int):
    """Print the configuration box."""
    modify_str = f"ON ({intensity})" if modify else "OFF"
    split_str = f"max {max_dur}s per part" if max_dur > 0 else "OFF"

    # Truncate URL if too long
    max_url = W - 20
    url_display = url if len(url) <= max_url else url[:max_url - 3] + "..."

    print(_box_top())
    print(_box_line(f"SOURCE    : {source}"))
    print(_box_line(f"URL       : {url_display}"))
    print(_box_line(f"OUTPUT    : {output_dir}/"))
    print(_box_line(f"MODIFY    : {modify_str}"))
    print(_box_line(f"SPLIT     : {split_str}"))
    print(_box_bottom())
    print()


def print_step(step: int, total: int, label: str):
    """Print a step header like: --- [1/3] DOWNLOADING FROM YOUTUBE ---"""
    tag = f"[{step}/{total}] {label.upper()}"
    pad = (W - len(tag) - 6) // 2
    line = f"{'─' * pad} {tag} {'─' * pad}"
    # Fix odd width
    if len(line) < W:
        line += "─"
    print(f"\n{line}")


def print_info(label: str, value: str):
    """Print an info line: TRACK : Some Title"""
    print(f"  {label:<10}: {value}")


def print_sub(text: str):
    """Print a sub-line (indented)."""
    print(f"  {text}")


def print_ok(text: str):
    """Print a success line: [OK] something"""
    print(f"  [OK] {text}")


def print_err(text: str):
    """Print an error line: [ERR] something"""
    print(f"  [ERR] {text}", file=sys.stderr)


def print_warn(text: str):
    """Print a warning line: [!] something"""
    print(f"  [!] {text}")


def print_split_info(duration: float, num_parts: int, max_duration: int):
    """Print split information."""
    print(f"  SPLITTING : {duration:.0f}s ({duration/60:.1f}min) -> {num_parts} parts @ {max_duration}s max")


def print_split_part(part_num: int, duration: float):
    """Print individual split part info."""
    print(f"    part {part_num:>2}  : {duration:.1f}s ({duration/60:.1f}min)")


def print_mods(description: str):
    """Print modification profile description."""
    # Break long mod descriptions into multiple lines
    if len(description) > W - 16:
        parts = description.split(" | ")
        print(f"  MODS      : {parts[0]}")
        for part in parts[1:]:
            print(f"            : {part}")
    else:
        print(f"  MODS      : {description}")


def print_progress(downloaded: int, total: int, speed: float, eta: int):
    """Print download progress bar."""
    bar_width = 25
    if total > 0:
        percent = downloaded / total * 100
        filled = int(percent / (100 / bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        size_mb = total / 1_048_576
        speed_kb = speed / 1024
        sys.stdout.write(
            f"\r  [{bar}] {percent:5.1f}%  "
            f"{size_mb:.1f}MB  "
            f"{speed_kb:.0f}KB/s  "
            f"ETA {eta}s   "
        )
    else:
        dl_mb = downloaded / 1_048_576
        sys.stdout.write(f"\r  downloading... {dl_mb:.1f}MB")
    sys.stdout.flush()


def print_progress_done():
    """Clear progress line and print done."""
    sys.stdout.write(f"\r  [{'#' * 25}] 100.0%  done.{' ' * 20}\n")
    sys.stdout.flush()


def print_results(output_paths: list[str]):
    """Print the final results box."""
    print()
    print(_box_top())
    print(_box_line("OUTPUT FILES"))
    print(_box_line())

    if len(output_paths) == 1:
        # Single file — show full path
        path = output_paths[0]
        if len(path) > W - 10:
            path = "..." + path[-(W - 13):]
        print(_box_line(path))
    else:
        for i, path in enumerate(output_paths, 1):
            fname = Path(path).name if len(path) > W - 18 else path
            if len(fname) > W - 18:
                fname = fname[:W - 21] + "..."
            print(_box_line(f"part {i:>2} : {fname}"))

    print(_box_line())
    print(_box_line(f"{len(output_paths)} file(s) created."))
    print(_box_bottom())
    print()


def print_done():
    """Print the final completion line."""
    tag = "COMPLETE"
    pad = (W - len(tag) - 4) // 2
    print(f"{'=' * pad} {tag} {'=' * pad}")
    print()


def print_fatal(message: str):
    """Print a fatal error and nothing else."""
    print()
    print(_box_top())
    print(_box_line("ERROR"))
    print(_box_line())
    # Wrap long error messages
    max_line = W - 8
    words = str(message).split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > max_line:
            print(_box_line(line))
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        print(_box_line(line))
    print(_box_bottom())
    print()


# Need Path for print_results
from pathlib import Path