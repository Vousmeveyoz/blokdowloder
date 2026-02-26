"""
filename_generator.py
Generates unique filenames in the format: {Title}_{random_string}

Pipeline:
  1. Sanitize karakter invalid untuk filename
  2. SELALU buang noise words (DJ, FULL BASS, VIRAL, dll)
  3. Jika masih panjang (> MAX_TITLE_LENGTH), shorten lebih lanjut
  4. Tambahkan random suffix
"""

import random
import re
import string

# Batas panjang judul sebelum diperpendek
MAX_TITLE_LENGTH = 50

# ── Noise patterns ──────────────────────────────────────────────────────

# Multi-word noise phrases (harus dicek SEBELUM single words)
_NOISE_PHRASES = re.compile(
    r"(?i)\b(?:"
    r"full\s*bass|bass\s*boosted|bass\s*boost|"
    r"full\s*version|full\s*album|full\s*song|"
    r"slowed\s*\+?\s*reverb|sped\s*up|"
    r"speed\s*up|slow\s*motion|"
    r"no\s*copyright|copyright\s*free|royalty\s*free|"
    r"lirik\s*lagu|lirik\s*video|lyrics?\s*video|"
    r"musik\s*tiktok|lagu\s*tiktok|"
    r"dj\s*remix|dj\s*viral|dj\s*tiktok|dj\s*full\s*bass|dj\s*full|"
    r"nonstop\s*mix|mega\s*mix|jedag\s*jedug"
    r")\b"
)

# Single noise words (word-boundary safe)
_NOISE_WORDS = re.compile(
    r"\b("
    r"viral|tiktok|trending|official|music|video|audio|lyrics?|"
    r"song|bass|slow[s]?|slowed|reverb|"
    r"hd|hq|mv|cover|remix|remaster(?:ed)?|"
    r"vj|ft\.?|feat\.?|prod\.?|live|session|karaoke|"
    r"best|top|hits?|new|latest|terbaru|"
    r"dj|nonstop|dangdut|koplo|"
    r"subscribe|like|share|channel|now|here|"
    r"full|jedag|jedug|"
    r"1080p|720p|480p|4k|8k"
    r")\b",
    flags=re.IGNORECASE,
)

# Tahun (2000–2099)
_YEAR_RES = re.compile(r"\b20\d{2}\b")

# Konten dalam kurung/bracket yang isinya noise (e.g., "(Official Music Video)", "[DJ Remix]")
_BRACKET_NOISE = re.compile(
    r"\([^)]*\b(?:official|music|video|lyric|audio|mv|hd|hq|remix|cover|live|karaoke|tiktok|viral|dj|bass)\b[^)]*\)"
    r"|"
    r"\[[^\]]*\b(?:official|music|video|lyric|audio|mv|hd|hq|remix|cover|live|karaoke|tiktok|viral|dj|bass)\b[^\]]*\]",
    flags=re.IGNORECASE,
)

# ── Functions ───────────────────────────────────────────────────────────


def generate_random_string(length: int = 26) -> str:
    """Generate a random alphanumeric string (lowercase)."""
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choices(characters, k=length))


def sanitize_title(title: str) -> str:
    """Remove characters invalid in filenames and normalize whitespace."""
    sanitized = re.sub(r'[\\/*?:"<>|#]', " ", title)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def clean_noise(title: str) -> str:
    """
    Remove ALL noise words and phrases from title.
    Always runs regardless of title length.

    Pipeline:
      1. Strip bracket sections containing noise (e.g., "(Official MV)")
      2. Strip multi-word noise phrases (e.g., "Full Bass", "Slowed + Reverb")
      3. Strip single noise words (e.g., "DJ", "Viral", "Remix")
      4. Strip standalone years (e.g., "2024")
      5. Clean up leftover punctuation and whitespace
    """
    # Step 1 — Bracket noise
    cleaned = _BRACKET_NOISE.sub(" ", title)

    # Step 2 — Multi-word phrases first (before splitting into single words)
    cleaned = _NOISE_PHRASES.sub(" ", cleaned)

    # Step 3 — Single noise words
    cleaned = _NOISE_WORDS.sub(" ", cleaned)

    # Step 4 — Years
    cleaned = _YEAR_RES.sub(" ", cleaned)

    # Step 5 — Cleanup
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", " ", cleaned)  # empty brackets
    cleaned = re.sub(r"[\-_.\s]+$", "", cleaned)         # trailing junk
    cleaned = re.sub(r"^[\-_.\s]+", "", cleaned)          # leading junk
    cleaned = re.sub(r"\s*[-–—]\s*$", "", cleaned)        # trailing dashes
    cleaned = re.sub(r"\s+", " ", cleaned).strip()        # normalize spaces

    # Fallback: jika cleaning terlalu agresif (semua noise),
    # kembalikan "Audio" daripada mengembalikan judul penuh noise
    if len(cleaned) < 3:
        return "Audio"

    return cleaned


def shorten_title(title: str, max_length: int = MAX_TITLE_LENGTH) -> str:
    """
    Shorten a title that's still too long after noise cleaning.

    Args:
        title: Already cleaned title
        max_length: Max characters allowed

    Returns:
        str: Shortened title
    """
    if len(title) <= max_length:
        return title

    # Ambil segmen pertama sebelum separator utama (|, —, //)
    first_segment = re.split(r"\s*[|—–]{1,2}\s*|\s*//\s*", title)[0].strip()
    if 5 <= len(first_segment) <= max_length:
        title = first_segment

    # Masih panjang? Potong di batas kata
    if len(title) > max_length:
        truncated = title[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            truncated = truncated[:last_space]
        title = truncated.strip()

    return title


def build_filename(title: str, random_length: int = 26) -> str:
    """
    Build the final output filename.

    Format : {Clean Title}_{random_string}
    Example: "DJ Menerima Luka Full Bass 2024" → "Menerima Luka_abc123..."

    Args:
        title: Track title from SoundCloud / YouTube
        random_length: Length of the random suffix (default: 26)

    Returns:
        str: Final filename (without extension)
    """
    sanitized = sanitize_title(title)
    cleaned = clean_noise(sanitized)      # ← SELALU clean noise
    shortened = shorten_title(cleaned)    # ← shorten kalau masih panjang
    random_suffix = generate_random_string(random_length)
    return f"{shortened}_{random_suffix}"