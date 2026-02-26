"""
audio_modifier.py
Applies subtle audio modifications so the output isn't identical to the original.

All processing is done via ffmpeg filters — no extra Python audio libs needed.

═══════════════════════════════════════════════════════════════════════════════
MODIFICATION LAYERS (10 total):
═══════════════════════════════════════════════════════════════════════════════

Layer 1 — Audio Tweaks:
  - Pitch shift        : ±0.5–2 semitones
  - Tempo change       : ±1–5% speed adjustment
  - Bass/Treble EQ     : slight boost/cut
  - Stereo widening    : subtle width adjustment
  - Layered EQ         : multi-band micro-nudges across spectrum
  - Micro-cuts         : remove tiny 10–50ms segments with smooth gap-fill

Layer 2 — Fingerprint Removal:
  - Metadata stripping : remove ALL metadata (ID3, encoder info, comments)
  - Watermark removal  : low-pass filter to kill high-freq watermarks (>17.5kHz)

Layer 3 — Human-Like Imperfections:
  - Micro-dynamics     : subtle random volume fluctuations over time
  - Phase jitter       : tiny L/R timing offset to break digital stereo symmetry
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import random
import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass


# ═════════════════════════════════════════════════════════════════════════════
# ModificationProfile
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ModificationProfile:
    """
    Configuration for all audio modifications.

    Set any value to None to skip that modification.
    Use ModificationProfile.random() for auto-generated subtle changes.
    """

    # ── Layer 1: Audio Tweaks ──
    pitch_semitones: float | None = None
    speed_factor: float | None = None
    bass_db: float | None = None
    treble_db: float | None = None
    stereo_width: float | None = None
    micro_cuts: int | None = None
    eq_layers: list[tuple[float, float, float]] | None = None

    # ── Layer 2: Fingerprint Removal ──
    # Strip all metadata from output (ID3, encoder, comments, etc)
    strip_metadata: bool = True

    # Low-pass cutoff frequency to remove high-freq watermarks
    # None = skip, typical value: 17500–18500 Hz
    # Human hearing tops out ~18kHz, watermarks often hide above that
    watermark_cutoff_hz: int | None = None

    # ── Layer 3: Human-Like Imperfections ──
    # Micro-dynamics: subtle volume fluctuation via tremolo
    # (rate_hz, depth_percent) — e.g. (0.3, 1.5) = 0.3Hz wobble, 1.5% depth
    micro_dynamics: tuple[float, float] | None = None

    # Phase jitter: tiny delay on one channel (in milliseconds)
    # Breaks perfect digital stereo symmetry. Range: 0.1–0.8ms
    phase_jitter_ms: float | None = None

    @classmethod
    def random(cls, intensity: str = "subtle") -> "ModificationProfile":
        """
        Generate a random modification profile.

        Args:
            intensity: "subtle", "moderate", or "strong"
        """
        presets = {
            "subtle": {
                "pitch": (-0.7, 0.7),
                "speed": (0.98, 1.02),
                "bass": (-1.5, 1.5),
                "treble": (-1.5, 1.5),
                "stereo": (0.9, 1.1),
                "cuts": (3, 8),
                "eq_layers_count": (2, 4),
                "eq_gain": (-1.0, 1.0),
                "watermark_cutoff": (17500, 18500),
                "dynamics_rate": (0.15, 0.5),
                "dynamics_depth": (0.5, 2.0),
                "phase_jitter": (0.1, 0.4),
            },
            "moderate": {
                "pitch": (-1.5, 1.5),
                "speed": (0.96, 1.04),
                "bass": (-3.0, 3.0),
                "treble": (-3.0, 3.0),
                "stereo": (0.8, 1.2),
                "cuts": (5, 15),
                "eq_layers_count": (3, 6),
                "eq_gain": (-2.0, 2.0),
                "watermark_cutoff": (16500, 18000),
                "dynamics_rate": (0.2, 0.8),
                "dynamics_depth": (1.0, 3.5),
                "phase_jitter": (0.2, 0.6),
            },
            "strong": {
                "pitch": (-2.0, 2.0),
                "speed": (0.93, 1.07),
                "bass": (-5.0, 5.0),
                "treble": (-5.0, 5.0),
                "stereo": (0.7, 1.3),
                "cuts": (10, 25),
                "eq_layers_count": (4, 8),
                "eq_gain": (-3.5, 3.5),
                "watermark_cutoff": (15500, 17500),
                "dynamics_rate": (0.3, 1.2),
                "dynamics_depth": (2.0, 5.0),
                "phase_jitter": (0.3, 0.8),
            },
        }

        p = presets.get(intensity, presets["subtle"])

        # ── Pick random Layer 1 mods (3–5 of 7) ──
        layer1_mods = ["pitch", "speed", "bass", "treble", "stereo", "cuts", "eq_layers"]
        chosen_l1 = random.sample(layer1_mods, k=random.randint(3, min(5, len(layer1_mods))))

        # ── Layer 2: always strip metadata, randomly enable watermark cut ──
        enable_watermark = random.random() < 0.7  # 70% chance

        # ── Layer 3: pick 1–2 imperfections ──
        layer3_mods = ["dynamics", "phase_jitter"]
        chosen_l3 = random.sample(layer3_mods, k=random.randint(1, 2))

        # Generate layered EQ
        eq_layers = None
        if "eq_layers" in chosen_l1:
            n = random.randint(*p["eq_layers_count"])
            freq_zones = [
                (60, 250),
                (250, 1000),
                (1000, 4000),
                (4000, 8000),
                (8000, 16000),
            ]
            chosen_zones = random.sample(freq_zones, k=min(n, len(freq_zones)))
            eq_layers = []
            for low, high in chosen_zones:
                freq = random.uniform(low, high)
                gain = round(random.uniform(*p["eq_gain"]), 2)
                bandwidth = round(freq * random.uniform(0.3, 0.8), 1)
                eq_layers.append((round(freq, 1), gain, bandwidth))

        # Generate micro-dynamics
        micro_dynamics = None
        if "dynamics" in chosen_l3:
            rate = round(random.uniform(*p["dynamics_rate"]), 2)
            depth = round(random.uniform(*p["dynamics_depth"]), 2)
            micro_dynamics = (rate, depth)

        return cls(
            # Layer 1
            pitch_semitones=(
                round(random.uniform(*p["pitch"]), 2) if "pitch" in chosen_l1 else None
            ),
            speed_factor=(
                round(random.uniform(*p["speed"]), 4) if "speed" in chosen_l1 else None
            ),
            bass_db=(
                round(random.uniform(*p["bass"]), 1) if "bass" in chosen_l1 else None
            ),
            treble_db=(
                round(random.uniform(*p["treble"]), 1) if "treble" in chosen_l1 else None
            ),
            stereo_width=(
                round(random.uniform(*p["stereo"]), 2) if "stereo" in chosen_l1 else None
            ),
            micro_cuts=(
                random.randint(*p["cuts"]) if "cuts" in chosen_l1 else None
            ),
            eq_layers=eq_layers,
            # Layer 2
            strip_metadata=True,
            watermark_cutoff_hz=(
                random.randint(*p["watermark_cutoff"]) if enable_watermark else None
            ),
            # Layer 3
            micro_dynamics=micro_dynamics,
            phase_jitter_ms=(
                round(random.uniform(*p["phase_jitter"]), 2)
                if "phase_jitter" in chosen_l3 else None
            ),
        )

    def describe(self) -> str:
        """Human-readable summary of active modifications."""
        parts = []

        # Layer 1
        if self.pitch_semitones is not None:
            d = "↑" if self.pitch_semitones > 0 else "↓"
            parts.append(f"Pitch {d}{abs(self.pitch_semitones):.2f}st")
        if self.speed_factor is not None:
            pct = (self.speed_factor - 1.0) * 100
            d = "↑" if pct > 0 else "↓"
            parts.append(f"Speed {d}{abs(pct):.1f}%")
        if self.bass_db is not None:
            s = "+" if self.bass_db > 0 else ""
            parts.append(f"Bass {s}{self.bass_db:.1f}dB")
        if self.treble_db is not None:
            s = "+" if self.treble_db > 0 else ""
            parts.append(f"Treble {s}{self.treble_db:.1f}dB")
        if self.stereo_width is not None:
            parts.append(f"Stereo {self.stereo_width:.2f}x")
        if self.micro_cuts is not None:
            parts.append(f"Cuts ×{self.micro_cuts}")
        if self.eq_layers is not None:
            parts.append(f"EQ {len(self.eq_layers)} bands")

        # Layer 2
        if self.strip_metadata:
            parts.append("Meta stripped")
        if self.watermark_cutoff_hz is not None:
            parts.append(f"LP {self.watermark_cutoff_hz}Hz")

        # Layer 3
        if self.micro_dynamics is not None:
            rate, depth = self.micro_dynamics
            parts.append(f"Dynamics {rate}Hz/{depth}%")
        if self.phase_jitter_ms is not None:
            parts.append(f"Phase ±{self.phase_jitter_ms}ms")

        return " | ".join(parts) if parts else "No modifications"


# ═════════════════════════════════════════════════════════════════════════════
# AudioModifier
# ═════════════════════════════════════════════════════════════════════════════


class AudioModifier:
    """
    Applies multi-layer audio modifications using ffmpeg filters.

    Pipeline:
      1. Audio tweaks (EQ, pitch, speed, stereo, layered EQ)
      2. Fingerprint removal (metadata strip, watermark low-pass)
      3. Human-like imperfections (micro-dynamics, phase jitter)
      4. Micro-cuts (tiny segment removal with smooth gap-fill)

    Usage:
        modifier = AudioModifier()
        output = modifier.modify("input.mp3", "output.ogg")
    """

    def __init__(self):
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError("ffmpeg tidak ditemukan di PATH.")
        if shutil.which("ffprobe") is None:
            raise EnvironmentError("ffprobe tidak ditemukan di PATH.")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_duration(self, filepath: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    def _get_channels(self, filepath: str) -> int:
        """Get number of audio channels using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a:0",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        if info.get("streams"):
            return int(info["streams"][0].get("channels", 2))
        return 2

    def _run_ffmpeg(self, command: list[str]) -> None:
        """Run an ffmpeg command, raising RuntimeError on failure."""
        try:
            subprocess.run(
                command, check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg error: {e}")

    # ── Filter chain builder ────────────────────────────────────────────

    def _build_filter_chain(self, profile: ModificationProfile, channels: int = 2) -> str:
        """
        Build the complete ffmpeg -af filter chain.

        Order (optimized for quality):
          1. Watermark removal (low-pass) — kill hidden high-freq markers first
          2. Layered EQ nudges
          3. Bass/treble broad EQ
          4. Micro-dynamics (tremolo)
          5. Pitch shift (asetrate + aresample)
          6. Speed change (atempo)
          7. Stereo widening (extrastereo) — stereo only
          8. Phase jitter (adelay on one channel) — stereo only
        """
        filters = []

        # ── Layer 2: Watermark removal ──
        if profile.watermark_cutoff_hz is not None:
            filters.append(
                f"lowpass=f={profile.watermark_cutoff_hz}:p=2"
            )

        # ── Layer 1: Layered EQ tweaks ──
        if profile.eq_layers:
            for freq, gain, bw in profile.eq_layers:
                if gain != 0:
                    filters.append(f"equalizer=f={freq}:t=h:w={bw}:g={gain}")

        # ── Layer 1: Bass & treble ──
        if profile.bass_db is not None and profile.bass_db != 0:
            filters.append(f"equalizer=f=100:t=h:w=200:g={profile.bass_db}")
        if profile.treble_db is not None and profile.treble_db != 0:
            filters.append(f"equalizer=f=8000:t=h:w=4000:g={profile.treble_db}")

        # ── Layer 3: Micro-dynamics ──
        if profile.micro_dynamics is not None:
            rate, depth = profile.micro_dynamics
            depth_normalized = depth / 100.0
            filters.append(f"tremolo=f={rate}:d={depth_normalized}")

        # ── Layer 1: Pitch shift ──
        if profile.pitch_semitones is not None and profile.pitch_semitones != 0:
            ratio = 2 ** (profile.pitch_semitones / 12.0)
            new_rate = int(44100 * ratio)
            filters.append(f"asetrate={new_rate}")
            filters.append("aresample=44100")

        # ── Layer 1: Speed ──
        if profile.speed_factor is not None and profile.speed_factor != 1.0:
            factor = max(0.5, min(2.0, profile.speed_factor))
            filters.append(f"atempo={factor}")

        # ── Layer 1: Stereo widening (stereo only) ──
        if channels >= 2:
            if profile.stereo_width is not None and profile.stereo_width != 1.0:
                filters.append(f"extrastereo=m={profile.stereo_width}")

        # ── Layer 3: Phase jitter (stereo only) ──
        if channels >= 2 and profile.phase_jitter_ms is not None:
            delay_ms = profile.phase_jitter_ms
            filters.append(f"adelay={delay_ms}|0")

        return ",".join(filters)

    # ── Micro-cuts ──────────────────────────────────────────────────────

    def _generate_cut_points(
        self, duration: float, num_cuts: int
    ) -> list[tuple[float, float]]:
        """
        Generate random micro-cut positions.

        Each cut removes 10–50ms. Avoids first/last 2 seconds.
        Cuts are non-overlapping with a 20ms safety buffer.
        """
        if duration < 10:
            return []

        safe_start = 2.0
        safe_end = duration - 2.0

        if safe_end - safe_start < 1.0:
            return []

        cuts = []
        attempts = 0

        while len(cuts) < num_cuts and attempts < num_cuts * 15:
            attempts += 1
            start = round(random.uniform(safe_start, safe_end - 0.06), 4)
            cut_len = round(random.uniform(0.010, 0.050), 4)
            end = round(start + cut_len, 4)

            if end >= safe_end:
                continue

            buffer = 0.020
            if any(
                not (end + buffer < cs or start - buffer > ce)
                for cs, ce in cuts
            ):
                continue

            cuts.append((start, end))

        cuts.sort()
        return cuts

    def _apply_micro_cuts(
        self,
        input_path: str,
        output_path: str,
        num_cuts: int,
        strip_metadata: bool = True,
        codec: str = "libvorbis",
        quality: str = "4",
    ) -> str:
        """
        Remove tiny segments and smoothly close the gaps.

        Uses aselect to exclude micro-ranges, then aresample async=1
        to seamlessly fill gaps without clicks or pops.
        """
        duration = self._get_duration(input_path)
        cuts = self._generate_cut_points(duration, num_cuts)

        if not cuts:
            return input_path

        exclude_parts = [f"between(t,{s},{e})" for s, e in cuts]
        select_expr = "not(" + "+".join(exclude_parts) + ")"

        filter_chain = (
            f"aselect='{select_expr}',"
            f"aresample=async=1:first_pts=0"
        )

        command = [
            "ffmpeg", "-i", str(input_path),
            "-af", filter_chain,
        ]

        if strip_metadata:
            command.extend(["-map_metadata", "-1"])

        command.extend([
            "-c:a", codec, "-q:a", quality,
            "-y", str(output_path),
        ])

        self._run_ffmpeg(command)
        return str(output_path)

    # ── Main entry point ────────────────────────────────────────────────

    def modify(
        self,
        input_path: str,
        output_path: str,
        profile: ModificationProfile | None = None,
        codec: str = "libvorbis",
        quality: str = "4",
    ) -> str:
        """
        Apply all audio modifications and save to output file.

        Full pipeline:
          Pass 1: All filter tweaks → intermediate WAV (lossless)
                  - Watermark removal (low-pass)
                  - EQ layers + bass/treble
                  - Micro-dynamics (tremolo)
                  - Pitch shift + speed change
                  - Stereo widening + phase jitter
                  - Metadata stripped
          Pass 2: Micro-cuts → final OGG (only compression here)
                  - Also strips metadata via -map_metadata -1

        If no micro-cuts, single pass with metadata stripping.
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

        if profile is None:
            profile = ModificationProfile.random(intensity="subtle")

        channels = self._get_channels(str(input_file))
        filter_chain = self._build_filter_chain(profile, channels)
        has_tweaks = bool(filter_chain)
        has_cuts = profile.micro_cuts is not None and profile.micro_cuts > 0
        strip = profile.strip_metadata

        # ── No modifications at all → plain convert + strip metadata ──
        if not has_tweaks and not has_cuts:
            cmd = ["ffmpeg", "-i", str(input_file)]
            if strip:
                cmd.extend(["-map_metadata", "-1"])
            cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
            self._run_ffmpeg(cmd)
            return str(output_path)

        # ── Both tweaks + cuts → 2-pass via lossless WAV ──
        if has_tweaks and has_cuts:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Pass 1: all tweaks → lossless WAV
                cmd1 = [
                    "ffmpeg", "-i", str(input_file),
                    "-af", filter_chain,
                    "-map_metadata", "-1",
                    "-c:a", "pcm_s16le",
                    "-y", tmp_path,
                ]
                self._run_ffmpeg(cmd1)

                # Pass 2: micro-cuts → final OGG
                self._apply_micro_cuts(
                    tmp_path, str(output_path),
                    num_cuts=profile.micro_cuts,
                    strip_metadata=strip,
                    codec=codec, quality=quality,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            return str(output_path)

        # ── Only tweaks (no cuts) → single pass ──
        if has_tweaks:
            cmd = [
                "ffmpeg", "-i", str(input_file),
                "-af", filter_chain,
            ]
            if strip:
                cmd.extend(["-map_metadata", "-1"])
            cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
            self._run_ffmpeg(cmd)
            return str(output_path)

        # ── Only cuts (no tweaks) ──
        self._apply_micro_cuts(
            str(input_file), str(output_path),
            num_cuts=profile.micro_cuts,
            strip_metadata=strip,
            codec=codec, quality=quality,
        )
        return str(output_path)