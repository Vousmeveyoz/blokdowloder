"""
audio_modifier.py
Applies subtle audio modifications so the output isn't identical to the original.

All processing is done via ffmpeg filters — no extra Python audio libs needed.

═══════════════════════════════════════════════════════════════════════════════
MODIFICATION LAYERS (16 total):
═══════════════════════════════════════════════════════════════════════════════

Layer 1 — Audio Tweaks:
  - Pitch shift        : ±0.5–2 semitones
  - Tempo change       : ±1–5% speed adjustment
  - Speed 2.3x         : aggressive speed-up via chained atempo (2.0 × 1.15)
  - Amplify            : volume adjustment in dB (e.g. -8dB)
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

Layer 4 — Anti-Fingerprint (Advanced):
  - Noise injection    : white/pink noise at -60 to -50dB to alter noise floor
  - Silence padding    : random ms silence at start/end to shift all timestamps
  - Subtle reverb      : very short aecho to add room signature
  - Time stretch       : rubberband-style stretch via atempo without pitch change
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
    speed_2x: bool | None = None
    amplify_db: float | None = None
    bass_db: float | None = None
    treble_db: float | None = None
    stereo_width: float | None = None
    micro_cuts: int | None = None
    eq_layers: list[tuple[float, float, float]] | None = None

    # ── Layer 2: Fingerprint Removal ──
    strip_metadata: bool = True
    watermark_cutoff_hz: int | None = None

    # ── Layer 3: Human-Like Imperfections ──
    micro_dynamics: tuple[float, float] | None = None
    phase_jitter_ms: float | None = None

    # ── Layer 4: Anti-Fingerprint (Advanced) ──

    # Noise injection: mix white noise at very low volume
    # (noise_db) — e.g. -55.0 means noise at -55dB (barely audible)
    # noise_type: 'white' or 'pink'
    noise_db: float | None = None
    noise_type: str = "white"

    # Silence padding: add random silence at start and/or end (in milliseconds)
    # Shifts all timestamps, breaks beat-alignment fingerprints
    # (pad_start_ms, pad_end_ms)
    silence_pad_ms: tuple[int, int] | None = None

    # Subtle reverb: very short echo/room effect using aecho
    # (in_gain, out_gain, delay_ms, decay)
    # e.g. (0.8, 0.6, 25, 0.2) = very subtle room feel
    reverb: tuple[float, float, int, float] | None = None

    # Time stretch: change duration slightly without pitch change
    # Independent from speed_factor — uses atempo in 0.95–1.05 range
    # Combined with pitch correction to keep pitch stable
    time_stretch: float | None = None

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
                "amplify": (-2.0, 2.0),
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
                "speed_2x_chance": 0.0,
                # Layer 4
                "noise_db": (-62.0, -55.0),
                "noise_chance": 0.7,
                "silence_pad": (5, 30),       # ms range per side
                "silence_chance": 0.8,
                "reverb_delay": (15, 35),     # ms
                "reverb_decay": (0.1, 0.25),
                "reverb_chance": 0.5,
                "time_stretch": (0.97, 1.03),
                "time_stretch_chance": 0.6,
            },
            "moderate": {
                "pitch": (-1.5, 1.5),
                "speed": (0.96, 1.04),
                "amplify": (-5.0, 3.0),
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
                "speed_2x_chance": 0.3,
                # Layer 4
                "noise_db": (-58.0, -50.0),
                "noise_chance": 0.85,
                "silence_pad": (10, 80),
                "silence_chance": 0.9,
                "reverb_delay": (20, 60),
                "reverb_decay": (0.15, 0.4),
                "reverb_chance": 0.65,
                "time_stretch": (0.94, 1.06),
                "time_stretch_chance": 0.75,
            },
            "strong": {
                "pitch": (-2.0, 2.0),
                "speed": (0.93, 1.07),
                "amplify": (-8.0, 5.0),
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
                "speed_2x_chance": 0.6,
                # Layer 4
                "noise_db": (-55.0, -45.0),
                "noise_chance": 1.0,
                "silence_pad": (20, 150),
                "silence_chance": 1.0,
                "reverb_delay": (25, 80),
                "reverb_decay": (0.2, 0.55),
                "reverb_chance": 0.8,
                "time_stretch": (0.91, 1.09),
                "time_stretch_chance": 0.9,
            },
        }

        p = presets.get(intensity, presets["subtle"])

        layer1_mods = ["pitch", "speed", "amplify", "bass", "treble", "stereo", "cuts", "eq_layers"]
        chosen_l1 = random.sample(layer1_mods, k=random.randint(3, min(6, len(layer1_mods))))

        enable_watermark    = random.random() < 0.7
        enable_speed_2x     = random.random() < p["speed_2x_chance"]
        enable_noise        = random.random() < p["noise_chance"]
        enable_silence      = random.random() < p["silence_chance"]
        enable_reverb       = random.random() < p["reverb_chance"]
        enable_time_stretch = random.random() < p["time_stretch_chance"]

        layer3_mods = ["dynamics", "phase_jitter"]
        chosen_l3 = random.sample(layer3_mods, k=random.randint(1, 2))

        # Generate layered EQ
        eq_layers = None
        if "eq_layers" in chosen_l1:
            n = random.randint(*p["eq_layers_count"])
            freq_zones = [
                (60, 250), (250, 1000), (1000, 4000),
                (4000, 8000), (8000, 16000),
            ]
            chosen_zones = random.sample(freq_zones, k=min(n, len(freq_zones)))
            eq_layers = []
            for low, high in chosen_zones:
                freq = random.uniform(low, high)
                gain = round(random.uniform(*p["eq_gain"]), 2)
                bw   = round(freq * random.uniform(0.3, 0.8), 1)
                eq_layers.append((round(freq, 1), gain, bw))

        micro_dynamics = None
        if "dynamics" in chosen_l3:
            rate  = round(random.uniform(*p["dynamics_rate"]), 2)
            depth = round(random.uniform(*p["dynamics_depth"]), 2)
            micro_dynamics = (rate, depth)

        speed_factor = None
        if "speed" in chosen_l1 and not enable_speed_2x and not enable_time_stretch:
            speed_factor = round(random.uniform(*p["speed"]), 4)

        amplify_db = None
        if "amplify" in chosen_l1:
            amplify_db = round(random.uniform(*p["amplify"]), 1)

        # Layer 4
        noise_db = None
        noise_type = "white"
        if enable_noise:
            noise_db   = round(random.uniform(*p["noise_db"]), 1)
            noise_type = random.choice(["white", "pink"])

        silence_pad_ms = None
        if enable_silence:
            pad_lo, pad_hi = p["silence_pad"]
            silence_pad_ms = (
                random.randint(pad_lo, pad_hi),
                random.randint(pad_lo, pad_hi),
            )

        reverb = None
        if enable_reverb:
            delay = random.randint(*p["reverb_delay"])
            decay = round(random.uniform(*p["reverb_decay"]), 2)
            reverb = (0.8, round(random.uniform(0.5, 0.75), 2), delay, decay)

        time_stretch = None
        if enable_time_stretch and not enable_speed_2x:
            time_stretch = round(random.uniform(*p["time_stretch"]), 4)

        return cls(
            pitch_semitones=(
                round(random.uniform(*p["pitch"]), 2) if "pitch" in chosen_l1 else None
            ),
            speed_factor=speed_factor,
            speed_2x=True if enable_speed_2x else None,
            amplify_db=amplify_db,
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
            strip_metadata=True,
            watermark_cutoff_hz=(
                random.randint(*p["watermark_cutoff"]) if enable_watermark else None
            ),
            micro_dynamics=micro_dynamics,
            phase_jitter_ms=(
                round(random.uniform(*p["phase_jitter"]), 2)
                if "phase_jitter" in chosen_l3 else None
            ),
            # Layer 4
            noise_db=noise_db,
            noise_type=noise_type,
            silence_pad_ms=silence_pad_ms,
            reverb=reverb,
            time_stretch=time_stretch,
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
        if self.speed_2x:
            parts.append("Speed 2.3x")
        if self.amplify_db is not None:
            s = "+" if self.amplify_db > 0 else ""
            parts.append(f"Amplify {s}{self.amplify_db:.1f}dB")
        if self.bass_db is not None:
            s = "+" if self.bass_db > 0 else ""
            parts.append(f"Bass {s}{self.bass_db:.1f}dB")
        if self.treble_db is not None:
            s = "+" if self.treble_db > 0 else ""
            parts.append(f"Treble {s}{self.treble_db:.1f}dB")
        if self.stereo_width is not None:
            parts.append(f"Stereo {self.stereo_width:.2f}x")
        if self.micro_cuts is not None:
            parts.append(f"Cuts x{self.micro_cuts}")
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

        # Layer 4
        if self.noise_db is not None:
            parts.append(f"Noise {self.noise_type} {self.noise_db:.1f}dB")
        if self.silence_pad_ms is not None:
            s, e = self.silence_pad_ms
            parts.append(f"Pad {s}ms/{e}ms")
        if self.reverb is not None:
            _, _, delay, decay = self.reverb
            parts.append(f"Reverb {delay}ms/{decay}")
        if self.time_stretch is not None:
            pct = (self.time_stretch - 1.0) * 100
            d = "↑" if pct > 0 else "↓"
            parts.append(f"Stretch {d}{abs(pct):.1f}%")

        return " | ".join(parts) if parts else "No modifications"


# ═════════════════════════════════════════════════════════════════════════════
# AudioModifier
# ═════════════════════════════════════════════════════════════════════════════


class AudioModifier:
    """
    Applies multi-layer audio modifications using ffmpeg filters.

    Pipeline order:
      1. Silence padding (prepend/append) — separate ffmpeg concat step
      2. Watermark removal (low-pass)
      3. Amplify / volume
      4. Layered EQ nudges
      5. Bass/treble broad EQ
      6. Micro-dynamics (tremolo)
      7. Noise injection (amix with anoisesrc)
      8. Subtle reverb (aecho)
      9. Pitch shift (asetrate + aresample)
     10. Speed change / time stretch (atempo)
     11. Speed 2.3x (chained atempo)
     12. Stereo widening (extrastereo)
     13. Phase jitter (adelay)
     14. Micro-cuts (separate pass)
    """

    def __init__(self):
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError("ffmpeg tidak ditemukan di PATH.")
        if shutil.which("ffprobe") is None:
            raise EnvironmentError("ffprobe tidak ditemukan di PATH.")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_duration(self, filepath: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json", "-show_format",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    def _get_channels(self, filepath: str) -> int:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json", "-show_streams",
            "-select_streams", "a:0", str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        if info.get("streams"):
            return int(info["streams"][0].get("channels", 2))
        return 2

    def _run_ffmpeg(self, command: list[str]) -> None:
        try:
            subprocess.run(
                command, check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg error: {e}")

    # ── Silence padding ───────────────────────────────────────────────────

    def _apply_silence_padding(
        self,
        input_path: str,
        output_path: str,
        pad_start_ms: int,
        pad_end_ms: int,
        channels: int = 2,
    ) -> None:
        """
        Add silence at start and/or end using anullsrc + concat.

        Uses ffmpeg filter_complex to prepend/append silence.
        pad_start_ms and pad_end_ms are in milliseconds.
        """
        start_s = pad_start_ms / 1000.0
        end_s   = pad_end_ms / 1000.0
        ch_layout = "stereo" if channels >= 2 else "mono"

        # Build filter_complex for concat
        # [silence_start][input][silence_end]concat=n=3:v=0:a=1
        filter_complex = (
            f"anullsrc=r=44100:cl={ch_layout}:d={start_s}[s];"
            f"anullsrc=r=44100:cl={ch_layout}:d={end_s}[e];"
            f"[s][0:a][e]concat=n=3:v=0:a=1[out]"
        )

        command = [
            "ffmpeg", "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map_metadata", "-1",
            "-c:a", "pcm_s16le",
            "-y", str(output_path),
        ]
        self._run_ffmpeg(command)

    # ── Filter chain builder ──────────────────────────────────────────────

    def _build_filter_chain(self, profile: ModificationProfile, channels: int = 2) -> str:
        """
        Build the complete ffmpeg -af filter chain.

        Note: silence_pad uses filter_complex (separate step).
              noise injection uses amix (included here as lavfi input workaround
              via aeval for simplicity without extra input).
        """
        filters = []

        # ── Layer 2: Watermark removal ──
        if profile.watermark_cutoff_hz is not None:
            filters.append(f"lowpass=f={profile.watermark_cutoff_hz}:p=2")

        # ── Layer 1: Amplify ──
        if profile.amplify_db is not None and profile.amplify_db != 0:
            filters.append(f"volume={profile.amplify_db}dB")

        # ── Layer 1: Layered EQ ──
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
            filters.append(f"tremolo=f={rate}:d={depth / 100.0}")

        # ── Layer 4: Noise injection ──
        # Uses aeval to generate noise inline — no extra input needed
        # white noise: random(), pink noise: approximated via filtered random
        if profile.noise_db is not None:
            # Convert dB to linear amplitude: A = 10^(dB/20)
            amplitude = 10 ** (profile.noise_db / 20.0)
            if profile.noise_type == "pink":
                # Pink noise: white noise + lowpass filter to roll off high freqs
                filters.append(
                    f"aeval='val(0)+{amplitude}*(random(0)-0.5)|val(1)+{amplitude}*(random(1)-0.5)',"
                    f"lowpass=f=6000"
                )
            else:
                # White noise: add flat-spectrum noise directly to signal
                filters.append(
                    f"aeval='val(0)+{amplitude}*(random(0)-0.5)|val(1)+{amplitude}*(random(1)-0.5)'"
                )

        # ── Layer 4: Subtle reverb ──
        if profile.reverb is not None:
            in_g, out_g, delay_ms, decay = profile.reverb
            filters.append(f"aecho={in_g}:{out_g}:{delay_ms}:{decay}")

        # ── Layer 1: Pitch shift ──
        if profile.pitch_semitones is not None and profile.pitch_semitones != 0:
            ratio    = 2 ** (profile.pitch_semitones / 12.0)
            new_rate = int(44100 * ratio)
            filters.append(f"asetrate={new_rate}")
            filters.append("aresample=44100")

        # ── Layer 1: Speed (subtle ±5%) ──
        if profile.speed_factor is not None and profile.speed_factor != 1.0 and not profile.speed_2x:
            factor = max(0.5, min(2.0, profile.speed_factor))
            filters.append(f"atempo={factor}")

        # ── Layer 4: Time stretch (independent tempo change) ──
        # Unlike speed_factor, time_stretch is always paired with pitch correction
        # to keep perceived pitch stable while changing duration
        if profile.time_stretch is not None and profile.time_stretch != 1.0 and not profile.speed_2x:
            factor = max(0.5, min(2.0, profile.time_stretch))
            # Apply tempo change
            filters.append(f"atempo={factor}")
            # Compensate pitch to keep it stable (inverse pitch correction)
            inv_ratio   = 1.0 / factor
            compensate  = 2 ** (12 * (inv_ratio - 1) / 12.0)  # approximation
            inv_rate    = int(44100 * compensate)
            filters.append(f"asetrate={inv_rate}")
            filters.append("aresample=44100")

        # ── Layer 1: Speed 2.3x ──
        if profile.speed_2x:
            filters.append("atempo=2.0")
            filters.append("atempo=1.15")

        # ── Layer 1: Stereo widening (stereo only) ──
        if channels >= 2 and profile.stereo_width is not None and profile.stereo_width != 1.0:
            filters.append(f"extrastereo=m={profile.stereo_width}")

        # ── Layer 3: Phase jitter (stereo only) ──
        if channels >= 2 and profile.phase_jitter_ms is not None:
            filters.append(f"adelay={profile.phase_jitter_ms}|0")

        return ",".join(filters)

    # ── Micro-cuts ────────────────────────────────────────────────────────

    def _generate_cut_points(self, duration: float, num_cuts: int) -> list[tuple[float, float]]:
        if duration < 10:
            return []
        safe_start = 2.0
        safe_end   = duration - 2.0
        if safe_end - safe_start < 1.0:
            return []

        cuts = []
        attempts = 0
        while len(cuts) < num_cuts and attempts < num_cuts * 15:
            attempts += 1
            start   = round(random.uniform(safe_start, safe_end - 0.06), 4)
            cut_len = round(random.uniform(0.010, 0.050), 4)
            end     = round(start + cut_len, 4)
            if end >= safe_end:
                continue
            buffer = 0.020
            if any(not (end + buffer < cs or start - buffer > ce) for cs, ce in cuts):
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
        duration = self._get_duration(input_path)
        cuts     = self._generate_cut_points(duration, num_cuts)
        if not cuts:
            return input_path

        exclude_parts = [f"between(t,{s},{e})" for s, e in cuts]
        select_expr   = "not(" + "+".join(exclude_parts) + ")"
        filter_chain  = f"aselect='{select_expr}',aresample=async=1:first_pts=0"

        command = ["ffmpeg", "-i", str(input_path), "-af", filter_chain]
        if strip_metadata:
            command.extend(["-map_metadata", "-1"])
        command.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
        self._run_ffmpeg(command)
        return str(output_path)

    # ── Main entry point ──────────────────────────────────────────────────

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

        Multi-pass pipeline:
          Pass 1 (optional): Silence padding → WAV
          Pass 2:            All filter tweaks → WAV
          Pass 3 (optional): Micro-cuts → final OGG
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

        if profile is None:
            profile = ModificationProfile.random(intensity="subtle")

        channels     = self._get_channels(str(input_file))
        filter_chain = self._build_filter_chain(profile, channels)
        has_tweaks   = bool(filter_chain)
        has_cuts     = profile.micro_cuts is not None and profile.micro_cuts > 0
        has_padding  = (
            profile.silence_pad_ms is not None
            and sum(profile.silence_pad_ms) > 0
        )
        strip = profile.strip_metadata

        tmp_files = []

        try:
            current = str(input_file)

            # ── Pass 1: Silence padding ──
            if has_padding:
                pad_start, pad_end = profile.silence_pad_ms
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    padded_path = f.name
                tmp_files.append(padded_path)
                self._apply_silence_padding(
                    current, padded_path,
                    pad_start_ms=pad_start,
                    pad_end_ms=pad_end,
                    channels=channels,
                )
                current = padded_path

            # ── Pass 2: Filter tweaks ──
            if has_tweaks:
                # If more passes follow, output to WAV (lossless)
                if has_cuts:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        tweaked_path = f.name
                    tmp_files.append(tweaked_path)
                    cmd = [
                        "ffmpeg", "-i", current,
                        "-af", filter_chain,
                        "-map_metadata", "-1",
                        "-c:a", "pcm_s16le",
                        "-y", tweaked_path,
                    ]
                    self._run_ffmpeg(cmd)
                    current = tweaked_path
                else:
                    # Final pass — encode directly to output format
                    cmd = ["ffmpeg", "-i", current, "-af", filter_chain]
                    if strip:
                        cmd.extend(["-map_metadata", "-1"])
                    cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
                    self._run_ffmpeg(cmd)
                    return str(output_path)

            elif not has_cuts:
                # No tweaks, no cuts — plain convert + strip
                cmd = ["ffmpeg", "-i", current]
                if strip:
                    cmd.extend(["-map_metadata", "-1"])
                cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
                self._run_ffmpeg(cmd)
                return str(output_path)

            # ── Pass 3: Micro-cuts → final output ──
            if has_cuts:
                self._apply_micro_cuts(
                    current, str(output_path),
                    num_cuts=profile.micro_cuts,
                    strip_metadata=strip,
                    codec=codec, quality=quality,
                )

        finally:
            for tmp in tmp_files:
                Path(tmp).unlink(missing_ok=True)

        return str(output_path)
