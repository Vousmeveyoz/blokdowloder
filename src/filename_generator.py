"""
filename_generator.py
Generates unique filenames in the format: {Title}_{random_string}

Jika judul terlalu panjang (> MAX_TITLE_LENGTH karakter), otomatis
diperpendek dengan strategi rule-based:
  1. Ambil segmen pertama sebelum separator utama (|, —, //)
  2. Buang kata-kata noise (VIRAL, TIKTOK, tahun, dll)
  3. Buang tanda kurung kosong / sisa noise
  4. Potong di batas kata jika masih panjang
"""

import random
import re
import string

# Batas panjang judul sebelum diperpendek
MAX_TITLE_LENGTH = 50

# Kata-kata noise (word-boundary safe)
_NOISE_WORDS = re.compile(
    r"\b("
    r"viral|tiktok|trending|official|music|video|audio|lyric|SONG|song|BASS|SLOW[s]?|"
    r"full\s*version|hd|hq|mv|cover|remix|remaster(?:ed)?|"
    r"vj|ft\.?|feat\.?|prod\.?|live|session|"
    r"best|top|hits?|new|latest|terbaru|DJ|"
    r"subscribe|like|share|channel|now|here|"
    r"1080p|720p|480p"
    r")\b",
    flags=re.IGNORECASE,
)

# Tahun (2000–2099) dan resolusi seperti 4K, 8K
_YEAR_RES = re.compile(r"\b20\d{2}\b|\s+\d+[kK]\b")


def generate_random_string(length: int = 26) -> str:
    """Generate a random alphanumeric string (lowercase)."""
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choices(characters, k=length))


def sanitize_title(title: str) -> str:
    """Remove characters invalid in filenames and normalize whitespace."""
    sanitized = re.sub(r'[\\/*?:"<>|#]', " ", title)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def shorten_title(title: str, max_length: int = MAX_TITLE_LENGTH) -> str:
    """
    Shorten a long title intelligently.

    Args:
        title: Raw or sanitized title
        max_length: Max characters allowed

    Returns:
        str: Shortened title
    """
    if len(title) <= max_length:
        return title

    # Step 1 — Ambil segmen pertama sebelum separator utama (|, —, //)
    first_segment = re.split(r"\s*[|—–]{1,2}\s*|\s*//\s*", title)[0].strip()
    if 5 <= len(first_segment) <= max_length:
        title = first_segment

    # Step 2 — Buang noise words + tahun + resolusi
    cleaned = _NOISE_WORDS.sub(" ", title)
    cleaned = _YEAR_RES.sub(" ", cleaned)

    # Step 3 — Buang kurung kosong & trailing noise
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", " ", cleaned)
    cleaned = re.sub(r"[\-_.\s]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) >= 5:
        title = cleaned

    # Step 4 — Masih panjang? Potong di batas kata
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

    Format : {Shortened Sanitized Title}_{random_string}
    Example: DJ Menerima Luka_cmm3napev200ctumheg2y4cdl

    Args:
        title: Track title from SoundCloud / YouTube
        random_length: Length of the random suffix (default: 26)

    Returns:
        str: Final filename (without extension)
    """
    sanitized = sanitize_title(title)
    shortened = shorten_title(sanitized)
    random_suffix = generate_random_string(random_length)
    return f"{shortened}_{random_suffix}"
