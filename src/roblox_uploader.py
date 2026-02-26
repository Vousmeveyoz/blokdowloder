"""
roblox_uploader.py
Handles uploading OGG audio files to Roblox via the Assets API.

API Docs:
    https://apis.roblox.com/assets/v1/assets

Usage:
    uploader = RobloxUploader(api_key="...", user_id="...")
    result = uploader.upload("path/to/audio.ogg", display_name="My Track")
"""

import requests
import json
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# RobloxUploader
# ═════════════════════════════════════════════════════════════════════════════


class RobloxUploader:
    """Uploads audio files to Roblox using the Assets API."""

    API_URL = "https://apis.roblox.com/assets/v1/assets"

    def __init__(self, api_key: str, user_id: str):
        """
        Args:
            api_key: Roblox Open Cloud API key
            user_id: Roblox user ID (found in profile URL)
        """
        self.api_key = api_key.strip()
        self.user_id = str(user_id).strip()

    def upload(
        self,
        filepath: str,
        display_name: str,
        description: str = "Uploaded via BLOKMARKET AUDIO",
    ) -> dict:
        """
        Upload an audio file to Roblox.

        Args:
            filepath:     Path to the audio file (.ogg or .mp3)
            display_name: Display name for the asset on Roblox
            description:  Asset description

        Returns:
            dict with keys:
              - success    (bool)
              - asset_id   (str | None)
              - asset_url  (str | None)
              - error      (str | None)
        """
        path = Path(filepath)
        if not path.exists():
            return {
                "success": False,
                "asset_id": None,
                "asset_url": None,
                "error": f"File tidak ditemukan: {filepath}",
            }

        # Determine content type
        ext = path.suffix.lower()
        content_type_map = {
            ".ogg": "audio/ogg",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
        }
        content_type = content_type_map.get(ext, "audio/ogg")

        # Build request payload
        request_body = {
            "assetType": "Audio",
            "displayName": display_name[:50],  # Roblox max 50 chars
            "description": description[:1000],
            "creationContext": {
                "creator": {
                    "userId": self.user_id
                }
            }
        }

        try:
            with open(path, "rb") as f:
                response = requests.post(
                    self.API_URL,
                    headers={"x-api-key": self.api_key},
                    data={"request": json.dumps(request_body)},
                    files={"fileContent": (path.name, f, content_type)},
                    timeout=60,
                )
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "asset_id": None,
                "asset_url": None,
                "error": "Request timeout. Coba lagi.",
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "asset_id": None,
                "asset_url": None,
                "error": "Koneksi gagal. Periksa internet.",
            }

        # Handle response
        if response.status_code in (200, 201):
            try:
                data = response.json()
                # Roblox returns assetId or operationId
                asset_id = (
                    data.get("assetId")
                    or data.get("operationId")
                    or data.get("id")
                )
                return {
                    "success": True,
                    "asset_id": str(asset_id) if asset_id else None,
                    "asset_url": f"rbxassetid://{asset_id}" if asset_id else None,
                    "error": None,
                }
            except Exception:
                return {
                    "success": True,
                    "asset_id": None,
                    "asset_url": None,
                    "error": None,
                }

        # Error responses
        error_map = {
            400: "Request tidak valid. Periksa format file.",
            401: "API key tidak valid atau tidak punya permission Assets.",
            403: "Akses ditolak. Pastikan API key punya permission Write Assets.",
            429: "Rate limit. Tunggu sebentar lalu coba lagi.",
            500: "Server Roblox error. Coba lagi nanti.",
        }
        error_msg = error_map.get(
            response.status_code,
            f"HTTP {response.status_code}: {response.text[:200]}"
        )

        return {
            "success": False,
            "asset_id": None,
            "asset_url": None,
            "error": error_msg,
        }
