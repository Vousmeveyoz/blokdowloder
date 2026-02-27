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

    def _poll_operation(
        self,
        operation_id: str,
        max_wait: int = 86400,
        interval: int = 15,
    ) -> dict:
        """
        Poll operation endpoint until done=True (Approved/Rejected).

        - interval: 15 detik antar poll (Roblox moderasi butuh 1-5 menit)
        - max_wait: default 86400 (24 jam) — practically infinite
        - Saat network error: skip + retry, TIDAK langsung return
        - Hanya berhenti saat: done=True ATAU max_wait habis
        """
        url = self.OPERATION_URL.format(operation_id=operation_id)
        elapsed = 0
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10  # toleransi 10 error berturut-turut

        while elapsed < max_wait:
            try:
                resp = requests.get(url, headers=self._headers(), timeout=30)
                consecutive_errors = 0  # reset error counter on success

            except requests.exceptions.RequestException as e:
                # Network error — jangan return, cukup skip dan coba lagi
                consecutive_errors += 1
                print(f"  [~] Network error ({consecutive_errors}), retry... ({elapsed}s)", end="\r", flush=True)

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return {
                        "done": False,
                        "asset_id": None,
                        "moderation_state": self.MODERATION_REVIEWING,
                        "moderation_note": None,
                        "error": f"Gagal poll setelah {MAX_CONSECUTIVE_ERRORS} error berturut-turut: {e}",
                    }

                time.sleep(interval)
                elapsed += interval
                continue

            if resp.status_code != 200:
                # HTTP error — skip dan retry, bukan langsung return
                consecutive_errors += 1
                print(f"  [~] HTTP {resp.status_code}, retry... ({elapsed}s)", end="\r", flush=True)

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return {
                        "done": False,
                        "asset_id": None,
                        "moderation_state": self.MODERATION_REVIEWING,
                        "moderation_note": None,
                        "error": f"HTTP {resp.status_code} terlalu sering.",
                    }

                time.sleep(interval)
                elapsed += interval
                continue

            # Parse response
            try:
                data = resp.json()
            except Exception:
                time.sleep(interval)
                elapsed += interval
                continue

            if data.get("done", False):
                # Moderasi selesai — ambil result
                response_obj = data.get("response", {})
                asset_id   = response_obj.get("assetId")
                moderation = response_obj.get("moderationResult", {})
                mod_state  = moderation.get("moderationState")
                mod_note   = moderation.get("comment") or moderation.get("reason")

                return {
                    "done": True,
                    "asset_id": str(asset_id) if asset_id else None,
                    "moderation_state": mod_state,
                    "moderation_note": mod_note,
                    "error": None,
                }

            # done=False — belum selesai, tunggu lagi
            time.sleep(interval)
            elapsed += interval
            mins = elapsed // 60
            secs = elapsed % 60
            print(f"  [~] Menunggu moderasi Roblox... ({mins}m {secs}s)", end="\r", flush=True)

        # Max wait habis (24 jam) — sangat jarang terjadi
        return {
            "done": False,
            "asset_id": None,
            "moderation_state": self.MODERATION_REVIEWING,
            "moderation_note": None,
            "error": f"Timeout setelah {max_wait // 60} menit. Cek di Creator Dashboard.",
        }

    def upload(
        self,
        filepath: str,
        display_name: str,
        description: str = "Audio",
        wait_moderation: bool = True,
        moderation_timeout: int = 86400,
    ) -> dict:
        path = Path(filepath)
        if not path.exists():
            return self._fail(f"File tidak ditemukan: {filepath}")

        content_type_map = {
            ".ogg": "audio/ogg",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
        }
        content_type = content_type_map.get(path.suffix.lower(), "audio/ogg")

        request_body = {
            "assetType": "Audio",
            "displayName": display_name[:50],
            "description": description[:1000],
            "creationContext": {"creator": {"userId": self.user_id}}
        }

        # FIX: Gunakan nama file generic supaya tidak bocor judul asli
        # via multipart filename header ke Roblox API
        safe_filename = f"audio{path.suffix.lower()}"

        try:
            with open(path, "rb") as f:
                response = requests.post(
                    self.API_URL,
                    headers=self._headers(),
                    data={"request": json.dumps(request_body)},
                    files={"fileContent": (safe_filename, f, content_type)},
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
            return self._fail(error_map.get(
                response.status_code,
                f"HTTP {response.status_code}: {response.text[:200]}"
            ))

        try:
            data = response.json()
        except Exception:
            return self._fail("Gagal parse response dari Roblox.")

        # Kadang Roblox langsung return assetId (sudah done)
        direct_asset_id = data.get("assetId") or data.get("id")
        if direct_asset_id:
            return {
                "success": True,
                "asset_id": str(direct_asset_id),
                "asset_url": f"rbxassetid://{direct_asset_id}",
                "moderation_state": self.MODERATION_APPROVED,
                "moderation_note": None,
                "error": None,
            }

        # Async operation — perlu polling
        operation_path = data.get("path", "")
        operation_id = operation_path.split("/")[-1] if operation_path else None

        if not operation_id:
            return self._fail("Tidak dapat operationId dari response Roblox.")

        if not wait_moderation:
            return {
                "success": True,
                "asset_id": None,
                "asset_url": None,
                "moderation_state": self.MODERATION_REVIEWING,
                "moderation_note": None,
                "error": None,
                "operation_id": operation_id,
            }

        print(f"  [~] Upload diterima, menunggu moderasi Roblox (max 7 menit)...")
        poll = self._poll_operation(operation_id, max_wait=moderation_timeout, interval=10)
        print(" " * 70, end="\r")  # clear progress line

        asset_id = poll["asset_id"]
        return {
            "success": True,
            "asset_id": asset_id,
            "asset_url": f"rbxassetid://{asset_id}" if asset_id else None,
            "moderation_state": poll["moderation_state"],
            "moderation_note": poll["moderation_note"],
            "error": poll.get("error"),
        }
