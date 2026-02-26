"""
roblox_uploader.py
Handles uploading OGG audio files to Roblox via the Assets API.
Includes operation polling to check approve/reject moderation status.
"""

import time
import requests
import json
from pathlib import Path


class RobloxUploader:
    API_URL       = "https://apis.roblox.com/assets/v1/assets"
    OPERATION_URL = "https://apis.roblox.com/assets/v1/operations/{operation_id}"

    MODERATION_APPROVED  = "Approved"
    MODERATION_REJECTED  = "Rejected"
    MODERATION_REVIEWING = "Reviewing"

    def __init__(self, api_key: str, user_id: str):
        self.api_key = api_key.strip()
        self.user_id = str(user_id).strip()

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key}

    @staticmethod
    def _fail(error: str) -> dict:
        return {
            "success": False,
            "asset_id": None,
            "asset_url": None,
            "moderation_state": None,
            "moderation_note": None,
            "error": error,
        }

    def _poll_operation(self, operation_id: str, max_wait: int = 120, interval: int = 5) -> dict:
        url = self.OPERATION_URL.format(operation_id=operation_id)
        elapsed = 0

        while elapsed < max_wait:
            try:
                resp = requests.get(url, headers=self._headers(), timeout=30)
            except requests.exceptions.RequestException as e:
                return {"done": False, "asset_id": None, "moderation_state": None, "moderation_note": None, "error": str(e)}

            if resp.status_code != 200:
                return {"done": False, "asset_id": None, "moderation_state": None, "moderation_note": None, "error": f"HTTP {resp.status_code}"}

            data = resp.json()
            if data.get("done", False):
                response_obj = data.get("response", {})
                asset_id   = response_obj.get("assetId")
                moderation = response_obj.get("moderationResult", {})
                mod_state  = moderation.get("moderationState")
                mod_note   = moderation.get("comment") or moderation.get("reason")
                return {"done": True, "asset_id": str(asset_id) if asset_id else None, "moderation_state": mod_state, "moderation_note": mod_note, "error": None}

            time.sleep(interval)
            elapsed += interval
            print(f"  [~] Menunggu moderasi Roblox... ({elapsed}s)", end="\r", flush=True)

        return {"done": False, "asset_id": None, "moderation_state": self.MODERATION_REVIEWING, "moderation_note": None, "error": f"Timeout {max_wait}s"}

    def upload(self, filepath: str, display_name: str, description: str = "Uploaded via BLOKMARKET AUDIO", wait_moderation: bool = True, moderation_timeout: int = 120) -> dict:
        path = Path(filepath)
        if not path.exists():
            return self._fail(f"File tidak ditemukan: {filepath}")

        content_type_map = {".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}
        content_type = content_type_map.get(path.suffix.lower(), "audio/ogg")

        request_body = {
            "assetType": "Audio",
            "displayName": display_name[:50],
            "description": description[:1000],
            "creationContext": {"creator": {"userId": self.user_id}}
        }

        try:
            with open(path, "rb") as f:
                response = requests.post(
                    self.API_URL,
                    headers=self._headers(),
                    data={"request": json.dumps(request_body)},
                    files={"fileContent": (path.name, f, content_type)},
                    timeout=60,
                )
        except requests.exceptions.Timeout:
            return self._fail("Request timeout. Coba lagi.")
        except requests.exceptions.ConnectionError:
            return self._fail("Koneksi gagal. Periksa internet.")

        if response.status_code not in (200, 201):
            error_map = {
                400: "Request tidak valid. Periksa format file.",
                401: "API key tidak valid atau tidak punya permission Assets.",
                403: "Akses ditolak. Pastikan API key punya permission Write Assets.",
                429: "Rate limit. Tunggu sebentar lalu coba lagi.",
                500: "Server Roblox error. Coba lagi nanti.",
            }
            return self._fail(error_map.get(response.status_code, f"HTTP {response.status_code}: {response.text[:200]}"))

        try:
            data = response.json()
        except Exception:
            return self._fail("Gagal parse response dari Roblox.")

        # Check if assetId returned directly
        direct_asset_id = data.get("assetId") or data.get("id")
        if direct_asset_id:
            return {"success": True, "asset_id": str(direct_asset_id), "asset_url": f"rbxassetid://{direct_asset_id}", "moderation_state": self.MODERATION_APPROVED, "moderation_note": None, "error": None}

        # Get operationId for polling
        operation_path = data.get("path", "")
        operation_id = operation_path.split("/")[-1] if operation_path else None

        if not operation_id:
            return self._fail("Tidak dapat operationId dari response Roblox.")

        if not wait_moderation:
            return {"success": True, "asset_id": None, "asset_url": None, "moderation_state": self.MODERATION_REVIEWING, "moderation_note": None, "error": None, "operation_id": operation_id}

        print(f"  [~] Upload diterima, menunggu moderasi Roblox...")
        poll = self._poll_operation(operation_id, max_wait=moderation_timeout)
        print(" " * 60, end="\r")

        asset_id = poll["asset_id"]
        return {
            "success": True,
            "asset_id": asset_id,
            "asset_url": f"rbxassetid://{asset_id}" if asset_id else None,
            "moderation_state": poll["moderation_state"],
            "moderation_note": poll["moderation_note"],
            "error": poll.get("error"),
        }
