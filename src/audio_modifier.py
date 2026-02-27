"""
audio_modifier.py
Applies multi-layer audio modifications for Roblox upload compatibility.

All processing is done via ffmpeg filters — no extra Python audio libs needed.

═══════════════════════════════════════════════════════════════════════════════
MODIFICATION LAYERS (14 total — optimized for Roblox):
═══════════════════════════════════════════════════════════════════════════════

Layer 1 — Audio Tweaks:
  - Pitch shift        : ±0.5–2 semitones
  - Tempo change       : ±1–5% speed adjustment
  - Bass/Treble EQ     : slight boost/cut
  - Stereo widening    : subtle width adjustment
  - Layered EQ         : multi-band micro-nudges across spectrum
  - Micro-cuts         : remove tiny 10–50ms segments with smooth gap-fill

Layer 2 — Fingerprint Removal (Roblox optimized):
  - Metadata stripping : remove ALL metadata (ID3, encoder info, comments)
  - Watermark removal  : low-pass filter to kill high-freq watermarks (>17.5kHz)
  - Noise floor inject : add very low-level white noise to break hash matching
  - Silence padding    : add random 50–300ms silence at start/end
  - Frequency micro-shift: slight shift across spectrum to break spectral fingerprint

Layer 3 — Human-Like Imperfections:
  - Micro-dynamics     : subtle random volume fluctuations over time
  - Phase jitter       : tiny L/R timing offset to break digital stereo symmetry
  - Volume normalize   : slight gain adjustment to change waveform amplitude
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import random
import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field


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
    strip_metadata: bool = True
    watermark_cutoff_hz: int | None = None

    # Noise floor: volume of injected white noise (0.0–1.0)
    # Very low values (0.001–0.005) are inaudible but break hash matching
    noise_floor_volume: float | None = None

    # Silence padding: milliseconds of silence added to start and end
    # Changes total duration → breaks duration-based fingerprinting
    silence_pad_start_ms: int | None = None
    silence_pad_end_ms: int | None = None

    # Frequency micro-shift: slight allpass filter to shift phase across spectrum
    # Breaks spectral fingerprinting without audible change
    freq_micro_shift: float | None = None

    # ── Layer 3: Human-Like Imperfections ──
    micro_dynamics: tuple[float, float] | None = None
    phase_jitter_ms: float | None = None

    # Volume normalize: slight gain change in dB (-2 to +2)
    # Changes overall amplitude → different waveform hash
    volume_adjust_db: float | None = None

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
                "noise_floor": (0.001, 0.003),
                "silence_pad": (50, 200),
                "freq_shift": (0.01, 0.05),
                "volume_adjust": (-1.0, 1.0),
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
                "noise_floor": (0.002, 0.005),
                "silence_pad": (100, 300),
                "freq_shift": (0.03, 0.08),
                "volume_adjust": (-1.5, 1.5),
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
                "noise_floor": (0.003, 0.008),
                "silence_pad": (150, 400),
                "freq_shift": (0.05, 0.12),
                "volume_adjust": (-2.0, 2.0),
            },
        }

        p = presets.get(intensity, presets["subtle"])

        # ── Layer 1: pick 3–5 mods ──
        layer1_mods = ["pitch", "speed", "bass", "treble", "stereo", "cuts", "eq_layers"]
        chosen_l1 = random.sample(layer1_mods, k=random.randint(3, min(5, len(layer1_mods))))

        # ── Layer 2: always strip metadata + always enable Roblox fingerprint countermeasures ──
        enable_watermark = random.random() < 0.7
        # Roblox optimizations: ALWAYS enable these for best bypass chance
        enable_noise_floor = True
        enable_silence_pad = True
        enable_freq_shift = random.random() < 0.8  # 80% chance

        # ── Layer 3: pick 1–3 imperfections ──
        layer3_mods = ["dynamics", "phase_jitter", "volume_adjust"]
        chosen_l3 = random.sample(layer3_mods, k=random.randint(1, 3))

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
            noise_floor_volume=(
                round(random.uniform(*p["noise_floor"]), 4) if enable_noise_floor else None
            ),
            silence_pad_start_ms=(
                random.randint(*p["silence_pad"]) if enable_silence_pad else None
            ),
            silence_pad_end_ms=(
                random.randint(*p["silence_pad"]) if enable_silence_pad else None
            ),
            freq_micro_shift=(
                round(random.uniform(*p["freq_shift"]), 3) if enable_freq_shift else None
            ),
            # Layer 3
            micro_dynamics=micro_dynamics,
            phase_jitter_ms=(
                round(random.uniform(*p["phase_jitter"]), 2)
                if "phase_jitter" in chosen_l3 else None
            ),
            volume_adjust_db=(
                round(random.uniform(*p["volume_adjust"]), 2)
                if "volume_adjust" in chosen_l3 else None
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
        if self.noise_floor_volume is not None:
            parts.append(f"Noise {self.noise_floor_volume:.4f}")
        if self.silence_pad_start_ms is not None:
            parts.append(f"Pad +{self.silence_pad_start_ms}ms/{self.silence_pad_end_ms}ms")
        if self.freq_micro_shift is not None:
            parts.append(f"FShift {self.freq_micro_shift:.3f}")

        # Layer 3
        if self.micro_dynamics is not None:
            rate, depth = self.micro_dynamics
            parts.append(f"Dynamics {rate}Hz/{depth}%")
        if self.phase_jitter_ms is not None:
            parts.append(f"Phase ±{self.phase_jitter_ms}ms")
        if self.volume_adjust_db is not None:
            s = "+" if self.volume_adjust_db > 0 else ""
            parts.append(f"Vol {s}{self.volume_adjust_db:.2f}dB")

        return " | ".join(parts) if parts else "No modifications"


# ═════════════════════════════════════════════════════════════════════════════
# AudioModifier
# ═════════════════════════════════════════════════════════════════════════════


class AudioModifier:
    """
    Applies multi-layer audio modifications using ffmpeg filters.
    Optimized for Roblox content detection bypass.

    Pipeline:
      1. Audio tweaks (EQ, pitch, speed, stereo, layered EQ)
      2. Fingerprint removal (metadata strip, watermark low-pass, noise floor,
         silence padding, frequency micro-shift)
      3. Human-like imperfections (micro-dynamics, phase jitter, volume adjust)
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

    def _get_sample_rate(self, filepath: str) -> int:
        """Get audio sample rate using ffprobe."""
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
            return int(info["streams"][0].get("sample_rate", 44100))
        return 44100

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

        Order (optimized for quality + Roblox bypass):
          1. Watermark removal (low-pass)
          2. Layered EQ nudges
          3. Bass/treble broad EQ
          4. Frequency micro-shift (allpass)
          5. Micro-dynamics (tremolo)
          6. Pitch shift (asetrate + aresample)
          7. Speed change (atempo)
          8. Volume adjust
          9. Stereo widening (extrastereo) — stereo only
         10. Phase jitter (adelay on one channel) — stereo only
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

        # ── Layer 2: Frequency micro-shift (allpass phase rotation) ──
        if profile.freq_micro_shift is not None:
            # Apply multiple allpass filters at different frequencies
            # This rotates phase across the spectrum, changing the waveform shape
            # without audible difference — breaks spectral fingerprinting
            shift = profile.freq_micro_shift
            for freq in [200, 800, 2500, 6000, 12000]:
                adjusted_freq = freq * (1.0 + shift)
                filters.append(f"allpass=f={adjusted_freq:.1f}:w={freq * 0.5:.1f}")

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

        # ── Layer 3: Volume adjust ──
        if profile.volume_adjust_db is not None and profile.volume_adjust_db != 0:
            filters.append(f"volume={profile.volume_adjust_db}dB")

        # ── Layer 1: Stereo widening (stereo only) ──
        if channels >= 2:
            if profile.stereo_width is not None and profile.stereo_width != 1.0:
                filters.append(f"extrastereo=m={profile.stereo_width}")

        # ── Layer 3: Phase jitter (stereo only) ──
        if channels >= 2 and profile.phase_jitter_ms is not None:
            delay_ms = profile.phase_jitter_ms
            filters.append(f"adelay={delay_ms}|0")

        return ",".join(filters)

    # ── Noise floor injection ───────────────────────────────────────────

    def _build_noise_mix_command(
        self,
        input_path: str,
        output_path: str,
        noise_volume: float,
        duration: float,
        sample_rate: int,
        channels: int,
        filter_chain: str = "",
        strip_metadata: bool = True,
        codec: str = "libvorbis",
        quality: str = "4",
    ) -> list[str]:
        """
        Build ffmpeg command that mixes white noise into audio.

        Uses anoisesrc to generate noise, then amix to blend it
        at very low volume with the original audio.
        """
        # Build complex filter graph:
        # [0:a] = input audio (with optional filter chain)
        # [noise] = generated white noise
        # Mix them together
        
        audio_filters = filter_chain if filter_chain else "anull"
        
        complex_filter = (
            f"[0:a]{audio_filters}[modified];"
            f"anoisesrc=d={duration}:c=white:r={sample_rate}:a={noise_volume},"
            f"aformat=channel_layouts={'stereo' if channels >= 2 else 'mono'}[noise];"
            f"[modified][noise]amix=inputs=2:duration=first:dropout_transition=0[out]"
        )

        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-filter_complex", complex_filter,
            "-map", "[out]",
        ]

        if strip_metadata:
            cmd.extend(["-map_metadata", "-1"])
            # Explicitly clear all common metadata fields
            for tag in ["title", "artist", "album", "comment", "genre", "date", "track", "encoder"]:
                cmd.extend(["-metadata", f"{tag}="])

        cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
        return cmd

    # ── Silence padding ─────────────────────────────────────────────────

    def _apply_silence_padding(
        self,
        input_path: str,
        output_path: str,
        pad_start_ms: int,
        pad_end_ms: int,
        strip_metadata: bool = True,
        codec: str = "libvorbis",
        quality: str = "4",
    ) -> str:
        """
        Add silence padding at start and/or end of audio.
        Changes total duration → breaks duration-based fingerprinting.
        """
        pad_start_s = pad_start_ms / 1000.0
        pad_end_s = pad_end_ms / 1000.0

        # adelay for start padding, apad for end padding
        filters = []
        if pad_start_ms > 0:
            filters.append(f"adelay={pad_start_ms}|{pad_start_ms}")
        if pad_end_ms > 0:
            filters.append(f"apad=pad_dur={pad_end_s}")

        if not filters:
            return input_path

        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-af", ",".join(filters),
        ]

        if strip_metadata:
            cmd.extend(["-map_metadata", "-1"])
            for tag in ["title", "artist", "album", "comment", "genre", "date", "track", "encoder"]:
                cmd.extend(["-metadata", f"{tag}="])

        cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
        self._run_ffmpeg(cmd)
        return str(output_path)

    # ── Micro-cuts ──────────────────────────────────────────────────────

    def _generate_cut_points(
        self, duration: float, num_cuts: int
    ) -> list[tuple[float, float]]:
        """
        Generate random micro-cut positions.
        Each cut removes 10–50ms. Avoids first/last 2 seconds.
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
            for tag in ["title", "artist", "album", "comment", "genre", "date", "track", "encoder"]:
                command.extend(["-metadata", f"{tag}="])

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
        roblox_trick: bool = False,
    ) -> str:
        """
        Apply all audio modifications and save to output file.
        Optimized for Roblox content detection bypass.

        If roblox_trick=True, applies speed 2.3x + volume -8dB as final step.
        Restore in Roblox game with: sound.PlaybackSpeed = 0.4348

        Full pipeline:
          Pass 1: All filter tweaks + noise injection → intermediate WAV
          Pass 2: Micro-cuts → intermediate WAV
          Pass 3: Silence padding → intermediate WAV
          Pass 4: Roblox trick (speed 2.3x + vol -8dB) → final OGG
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

        if profile is None:
            profile = ModificationProfile.random(intensity="subtle")

        channels = self._get_channels(str(input_file))
        sample_rate = self._get_sample_rate(str(input_file))
        filter_chain = self._build_filter_chain(profile, channels)
        has_tweaks = bool(filter_chain)
        has_cuts = profile.micro_cuts is not None and profile.micro_cuts > 0
        has_noise = profile.noise_floor_volume is not None and profile.noise_floor_volume > 0
        has_padding = (
            (profile.silence_pad_start_ms is not None and profile.silence_pad_start_ms > 0) or
            (profile.silence_pad_end_ms is not None and profile.silence_pad_end_ms > 0)
        )
        strip = profile.strip_metadata

        temp_files = []

        try:
            current_input = str(input_file)

            # ── Pass 1: Filter tweaks + noise injection ──
            if has_tweaks or has_noise:
                tmp1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp1_path = tmp1.name
                tmp1.close()
                temp_files.append(tmp1_path)

                if has_noise:
                    duration = self._get_duration(current_input)
                    cmd = self._build_noise_mix_command(
                        input_path=current_input,
                        output_path=tmp1_path,
                        noise_volume=profile.noise_floor_volume,
                        duration=duration,
                        sample_rate=sample_rate,
                        channels=channels,
                        filter_chain=filter_chain if has_tweaks else "",
                        strip_metadata=strip,
                        codec="pcm_s16le",
                        quality="4",
                    )
                    self._run_ffmpeg(cmd)
                elif has_tweaks:
                    cmd = [
                        "ffmpeg", "-i", current_input,
                        "-af", filter_chain,
                        "-map_metadata", "-1",
                        "-c:a", "pcm_s16le",
                        "-y", tmp1_path,
                    ]
                    self._run_ffmpeg(cmd)

                current_input = tmp1_path

            # ── Pass 2: Micro-cuts ──
            if has_cuts:
                tmp2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp2_path = tmp2.name
                tmp2.close()
                temp_files.append(tmp2_path)

                self._apply_micro_cuts(
                    current_input, tmp2_path,
                    num_cuts=profile.micro_cuts,
                    strip_metadata=strip,
                    codec="pcm_s16le",
                    quality="4",
                )
                current_input = tmp2_path

            # ── Pass 3: Silence padding → final output ──
            if has_padding:
                tmp3 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp3_path = tmp3.name
                tmp3.close()
                temp_files.append(tmp3_path)

                self._apply_silence_padding(
                    current_input, tmp3_path,
                    pad_start_ms=profile.silence_pad_start_ms or 0,
                    pad_end_ms=profile.silence_pad_end_ms or 0,
                    strip_metadata=strip,
                    codec="pcm_s16le",
                    quality="4",
                )
                current_input = tmp3_path

            # ── Pass 4: Roblox trick (speed 2.3x + volume -8dB) ──
            if roblox_trick:
                tmp4 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp4_path = tmp4.name
                tmp4.close()
                temp_files.append(tmp4_path)

                # atempo max = 2.0, jadi chain 2x: atempo=2.0,atempo=1.15 = 2.3x
                trick_filter = "atempo=2.0,atempo=1.15,volume=-8dB"
                cmd_trick = [
                    "ffmpeg", "-i", current_input,
                    "-af", trick_filter,
                    "-map_metadata", "-1",
                    "-c:a", "pcm_s16le",
                    "-y", tmp4_path,
                ]
                self._run_ffmpeg(cmd_trick)
                current_input = tmp4_path

            # ── Final encode to OGG ──
            cmd_final = [
                "ffmpeg", "-i", current_input,
                "-map_metadata", "-1",
            ]
            # Strip all metadata tags explicitly
            for tag in ["title", "artist", "album", "comment", "genre", "date", "track", "encoder"]:
                cmd_final.extend(["-metadata", f"{tag}="])
            cmd_final.extend([
                "-c:a", codec, "-q:a", quality,
                "-y", str(output_path),
            ])
            self._run_ffmpeg(cmd_final)

        finally:
            # Cleanup all temp files
            for tmp in temp_files:
                Path(tmp).unlink(missing_ok=True)

        return str(output_path)
