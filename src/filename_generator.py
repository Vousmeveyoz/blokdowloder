"""
filename_generator.py
Generates unique filenames in the format: {random_string}

Pipeline:
  1. Generate random alphanumeric filename
  2. Tidak menggunakan judul asli sama sekali untuk menghindari
     bocornya metadata/nama lagu ke output file dan Roblox upload
"""

import random
import string


def generate_random_string(length: int = 26) -> str:
    """Generate a random alphanumeric string (lowercase)."""
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choices(characters, k=length))


def build_filename(title: str = "", random_length: int = 26) -> str:
    """
    Build the final output filename.

    Format : audio_{random_string}
    Example: "DJ Menerima Luka Full Bass 2024" → "audio_k3m9x2ab..."

    Args:
        title: Track title (ignored — kept for backward compatibility)
        random_length: Length of the random suffix (default: 26)

    Returns:
        str: Final filename (without extension)
    """
    random_suffix = generate_random_string(random_length)
    return f"audio_{random_suffix}"
