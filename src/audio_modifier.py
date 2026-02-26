"""
audio_modifier.py
Applies subtle audio modifications so the output isn't identical to the original.

All processing is done via ffmpeg filters — no extra Python audio libs needed.

═══════════════════════════════════════════════════════════════════════════════
MODIFICATION LAYERS (20 total) — Target: ~95% fingerprint bypass
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
  - Noise injection    : white/pink noise at -60 to -45dB to alter noise floor
  - Silence padding    : random ms silence at start/end to shift all timestamps
  - Subtle reverb      : very short aecho to add room signature
  - Time stretch       : tempo change with pitch compensation

Layer 5 — Perceptual Hash Breakers (New):
  - Spectral blurring  : micro-randomize FFT bins via afftfilt
  - Harmonic distortion: add subtle harmonic overtones via aeval
  - Mid-Side processing: independently process Mid and Side channels
  - Bit depth dithering: resample through lower bit depth to alter noise floor
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

    # ── Layer 4: Anti-Fingerprint ──
    noise_db: float | None = None
    noise_type: str = "white"
    silence_pad_ms: tuple[int, int] | None = None
    reverb: tuple[float, float, int, float] | None = None
    time_stretch: float | None = None

    # ── Layer 5: Perceptual Hash Breakers ──

    # Spectral blurring: micro-randomize FFT frequency bins
    # (blur_strength) — 0.001–0.02, higher = more blur
    spectral_blur: float | None = None

    # Harmonic distortion: add subtle overtones
    # (distortion_amount) — 0.002–0.015
    harmonic_distortion: float | None = None

    # Mid-Side processing: adjust mid/side balance independently
    # (mid_gain, side_gain) — e.g. (0.95, 1.05)
    mid_side: tuple[float, float] | None = None

    # Bit depth dithering: resample through 22050Hz to introduce
    # dither noise at quantization level, then back to 44100
    bit_dither: bool | None = None

    @classmethod
    def random(cls, intensity: str = "subtle") -> "ModificationProfile":
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
                "noise_db": (-62.0, -55.0),
                "noise_chance": 0.7,
                "silence_pad": (5, 30),
                "silence_chance": 0.8,
                "reverb_delay": (15, 35),
                "reverb_decay": (0.1, 0.25),
                "reverb_chance": 0.5,
                "time_stretch": (0.97, 1.03),
                "time_stretch_chance": 0.6,
                # Layer 5
                "spectral_blur": (0.001, 0.006),
                "spectral_chance": 0.7,
                "harmonic_dist": (0.002, 0.006),
                "harmonic_chance": 0.6,
                "mid_side_range": (0.92, 1.08),
                "mid_side_chance": 0.65,
                "bit_dither_chance": 0.5,
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
                "noise_db": (-58.0, -50.0),
                "noise_chance": 0.85,
                "silence_pad": (10, 80),
                "silence_chance": 0.9,
                "reverb_delay": (20, 60),
                "reverb_decay": (0.15, 0.4),
                "reverb_chance": 0.65,
                "time_stretch": (0.94, 1.06),
                "time_stretch_chance": 0.75,
                # Layer 5
                "spectral_blur": (0.004, 0.012),
                "spectral_chance": 0.85,
                "harmonic_dist": (0.005, 0.010),
                "harmonic_chance": 0.75,
                "mid_side_range": (0.85, 1.15),
                "mid_side_chance": 0.8,
                "bit_dither_chance": 0.7,
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
                "noise_db": (-55.0, -45.0),
                "noise_chance": 1.0,
                "silence_pad": (20, 150),
                "silence_chance": 1.0,
                "reverb_delay": (25, 80),
                "reverb_decay": (0.2, 0.55),
                "reverb_chance": 0.85,
                "time_stretch": (0.91, 1.09),
                "time_stretch_chance": 0.9,
                # Layer 5
                "spectral_blur": (0.008, 0.02),
                "spectral_chance": 1.0,
                "harmonic_dist": (0.008, 0.015),
                "harmonic_chance": 0.9,
                "mid_side_range": (0.78, 1.22),
                "mid_side_chance": 0.95,
                "bit_dither_chance": 0.9,
            },
        }

        p = presets.get(intensity, presets["subtle"])

        layer1_mods = ["pitch", "speed", "amplify", "bass", "treble", "stereo", "cuts", "eq_layers"]
        chosen_l1 = random.sample(layer1_mods, k=random.randint(3, min(6, len(layer1_mods))))

        enable_watermark     = random.random() < 0.7
        enable_speed_2x      = random.random() < p["speed_2x_chance"]
        enable_noise         = random.random() < p["noise_chance"]
        enable_silence       = random.random() < p["silence_chance"]
        enable_reverb        = random.random() < p["reverb_chance"]
        enable_time_stretch  = random.random() < p["time_stretch_chance"]
        enable_spectral      = random.random() < p["spectral_chance"]
        enable_harmonic      = random.random() < p["harmonic_chance"]
        enable_mid_side      = random.random() < p["mid_side_chance"]
        enable_bit_dither    = random.random() < p["bit_dither_chance"]

        layer3_mods = ["dynamics", "phase_jitter"]
        chosen_l3 = random.sample(layer3_mods, k=random.randint(1, 2))

        # Layered EQ
        eq_layers = None
        if "eq_layers" in chosen_l1:
            n = random.randint(*p["eq_layers_count"])
            freq_zones = [(60,250),(250,1000),(1000,4000),(4000,8000),(8000,16000)]
            chosen_zones = random.sample(freq_zones, k=min(n, len(freq_zones)))
            eq_layers = []
            for low, high in chosen_zones:
                freq = random.uniform(low, high)
                gain = round(random.uniform(*p["eq_gain"]), 2)
                bw   = round(freq * random.uniform(0.3, 0.8), 1)
                eq_layers.append((round(freq, 1), gain, bw))

        micro_dynamics = None
        if "dynamics" in chosen_l3:
            micro_dynamics = (
                round(random.uniform(*p["dynamics_rate"]), 2),
                round(random.uniform(*p["dynamics_depth"]), 2),
            )

        speed_factor = None
        if "speed" in chosen_l1 and not enable_speed_2x and not enable_time_stretch:
            speed_factor = round(random.uniform(*p["speed"]), 4)

        # Layer 4
        noise_db, noise_type = None, "white"
        if enable_noise:
            noise_db   = round(random.uniform(*p["noise_db"]), 1)
            noise_type = random.choice(["white", "pink"])

        silence_pad_ms = None
        if enable_silence:
            lo, hi = p["silence_pad"]
            silence_pad_ms = (random.randint(lo, hi), random.randint(lo, hi))

        reverb = None
        if enable_reverb:
            delay = random.randint(*p["reverb_delay"])
            decay = round(random.uniform(*p["reverb_decay"]), 2)
            reverb = (0.8, round(random.uniform(0.5, 0.75), 2), delay, decay)

        time_stretch = None
        if enable_time_stretch and not enable_speed_2x:
            time_stretch = round(random.uniform(*p["time_stretch"]), 4)

        # Layer 5
        spectral_blur = None
        if enable_spectral:
            spectral_blur = round(random.uniform(*p["spectral_blur"]), 4)

        harmonic_distortion = None
        if enable_harmonic:
            harmonic_distortion = round(random.uniform(*p["harmonic_dist"]), 4)

        mid_side = None
        if enable_mid_side:
            lo, hi = p["mid_side_range"]
            mid_g = round(random.uniform(lo, hi), 3)
            side_g = round(random.uniform(lo, hi), 3)
            mid_side = (mid_g, side_g)

        return cls(
            pitch_semitones=(
                round(random.uniform(*p["pitch"]), 2) if "pitch" in chosen_l1 else None
            ),
            speed_factor=speed_factor,
            speed_2x=True if enable_speed_2x else None,
            amplify_db=(
                round(random.uniform(*p["amplify"]), 1) if "amplify" in chosen_l1 else None
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
            strip_metadata=True,
            watermark_cutoff_hz=(
                random.randint(*p["watermark_cutoff"]) if enable_watermark else None
            ),
            micro_dynamics=micro_dynamics,
            phase_jitter_ms=(
                round(random.uniform(*p["phase_jitter"]), 2)
                if "phase_jitter" in chosen_l3 else None
            ),
            noise_db=noise_db,
            noise_type=noise_type,
            silence_pad_ms=silence_pad_ms,
            reverb=reverb,
            time_stretch=time_stretch,
            spectral_blur=spectral_blur,
            harmonic_distortion=harmonic_distortion,
            mid_side=mid_side,
            bit_dither=True if enable_bit_dither else None,
        )

    def describe(self) -> str:
        parts = []
        # Layer 1
        if self.pitch_semitones is not None:
            d = "↑" if self.pitch_semitones > 0 else "↓"
            parts.append(f"Pitch {d}{abs(self.pitch_semitones):.2f}st")
        if self.speed_factor is not None:
            pct = (self.speed_factor - 1.0) * 100
            parts.append(f"Speed {'↑' if pct>0 else '↓'}{abs(pct):.1f}%")
        if self.speed_2x:
            parts.append("Speed 2.3x")
        if self.amplify_db is not None:
            parts.append(f"Amplify {'+' if self.amplify_db>0 else ''}{self.amplify_db:.1f}dB")
        if self.bass_db is not None:
            parts.append(f"Bass {'+' if self.bass_db>0 else ''}{self.bass_db:.1f}dB")
        if self.treble_db is not None:
            parts.append(f"Treble {'+' if self.treble_db>0 else ''}{self.treble_db:.1f}dB")
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
            r, d = self.micro_dynamics
            parts.append(f"Dynamics {r}Hz/{d}%")
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
            parts.append(f"Stretch {'↑' if pct>0 else '↓'}{abs(pct):.1f}%")
        # Layer 5
        if self.spectral_blur is not None:
            parts.append(f"SpectralBlur {self.spectral_blur}")
        if self.harmonic_distortion is not None:
            parts.append(f"Harmonic {self.harmonic_distortion}")
        if self.mid_side is not None:
            m, s = self.mid_side
            parts.append(f"MidSide M{m}/S{s}")
        if self.bit_dither:
            parts.append("BitDither")
        return " | ".join(parts) if parts else "No modifications"


# ═════════════════════════════════════════════════════════════════════════════
# AudioModifier
# ═════════════════════════════════════════════════════════════════════════════


class AudioModifier:
    """
    Applies multi-layer audio modifications using ffmpeg filters.

    Full pipeline order:
      Pass 0 (opt): Silence padding      — shifts all timestamps
      Pass 1:       All filter tweaks    — spectral, harmonic, EQ, dynamics, etc
      Pass 2 (opt): Bit depth dithering  — resample through 22050 → 44100
      Pass 3 (opt): Micro-cuts           — remove tiny segments
    """

    def __init__(self):
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError("ffmpeg tidak ditemukan di PATH.")
        if shutil.which("ffprobe") is None:
            raise EnvironmentError("ffprobe tidak ditemukan di PATH.")

    def _get_duration(self, filepath: str) -> float:
        cmd = ["ffprobe","-v","quiet","-print_format","json","-show_format",str(filepath)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(json.loads(result.stdout)["format"]["duration"])

    def _get_channels(self, filepath: str) -> int:
        cmd = ["ffprobe","-v","quiet","-print_format","json","-show_streams","-select_streams","a:0",str(filepath)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        return int(info["streams"][0].get("channels", 2)) if info.get("streams") else 2

    def _run_ffmpeg(self, command: list[str]) -> None:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg error: {e}")

    # ── Silence padding ───────────────────────────────────────────────────────

    def _apply_silence_padding(self, input_path: str, output_path: str,
                                pad_start_ms: int, pad_end_ms: int, channels: int = 2) -> None:
        ch_layout = "stereo" if channels >= 2 else "mono"
        start_s = pad_start_ms / 1000.0
        end_s   = pad_end_ms   / 1000.0
        filter_complex = (
            f"anullsrc=r=44100:cl={ch_layout}:d={start_s}[s];"
            f"anullsrc=r=44100:cl={ch_layout}:d={end_s}[e];"
            f"[s][0:a][e]concat=n=3:v=0:a=1[out]"
        )
        self._run_ffmpeg([
            "ffmpeg", "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map_metadata", "-1",
            "-c:a", "pcm_s16le",
            "-y", str(output_path),
        ])

    # ── Bit depth dithering ───────────────────────────────────────────────────

    def _apply_bit_dither(self, input_path: str, output_path: str) -> None:
        """Resample 44100 → 22050 → 44100 with triangular dithering.
        Introduces quantization noise at a different level, altering noise floor."""
        self._run_ffmpeg([
            "ffmpeg", "-i", str(input_path),
            "-af", "aresample=22050:dither_method=triangular,aresample=44100:dither_method=triangular_hp",
            "-map_metadata", "-1",
            "-c:a", "pcm_s16le",
            "-y", str(output_path),
        ])

    # ── Filter chain builder ──────────────────────────────────────────────────

    def _build_filter_chain(self, profile: ModificationProfile, channels: int = 2) -> str:
        filters = []

        # ── Layer 2: Watermark removal ──
        if profile.watermark_cutoff_hz is not None:
            filters.append(f"lowpass=f={profile.watermark_cutoff_hz}:p=2")

        # ── Layer 5: Spectral blurring ──
        # Micro-randomize FFT real/imag bins — breaks spectral fingerprint
        if profile.spectral_blur is not None:
            s = profile.spectral_blur
            inv = round(1 - s, 6)
            filters.append(
                f"afftfilt=real=re*{inv}+{s}*(random(0)-0.5)"
                f":imag=im*{inv}+{s}*(random(1)-0.5)"
            )

        # ── Layer 5: Mid-Side processing ──
        # M = (L+R)/2, S = (L-R)/2 → adjust independently → back to L/R
        if profile.mid_side is not None and channels >= 2:
            m, s = profile.mid_side
            # pan filter: new_L = m*(L+R)/2 + s*(L-R)/2, new_R = m*(L+R)/2 - s*(L-R)/2
            # simplified: new_L = ((m+s)/2)*L + ((m-s)/2)*R
            #             new_R = ((m-s)/2)*L + ((m+s)/2)*R
            a = round((m + s) / 2, 4)
            b = round((m - s) / 2, 4)
            filters.append(f"pan=stereo|c0={a}*c0+{b}*c1|c1={b}*c0+{a}*c1")

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
        if profile.noise_db is not None:
            amp = 10 ** (profile.noise_db / 20.0)
            if profile.noise_type == "pink":
                filters.append(
                    f"aeval='val(0)+{amp}*(random(0)-0.5)|val(1)+{amp}*(random(1)-0.5)',"
                    f"lowpass=f=6000"
                )
            else:
                filters.append(
                    f"aeval='val(0)+{amp}*(random(0)-0.5)|val(1)+{amp}*(random(1)-0.5)'"
                )

        # ── Layer 5: Harmonic distortion ──
        # Add subtle 2nd and 3rd harmonic overtones
        if profile.harmonic_distortion is not None:
            d = profile.harmonic_distortion
            filters.append(
                f"aeval='val(0)+{d}*sin(val(0)*6.28318)+{d*0.5}*sin(val(0)*9.42478)"
                f"|val(1)+{d}*sin(val(1)*6.28318)+{d*0.5}*sin(val(1)*9.42478)'"
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

        # ── Layer 1: Speed ──
        if profile.speed_factor is not None and profile.speed_factor != 1.0 and not profile.speed_2x:
            filters.append(f"atempo={max(0.5, min(2.0, profile.speed_factor))}")

        # ── Layer 4: Time stretch (with pitch compensation) ──
        if profile.time_stretch is not None and profile.time_stretch != 1.0 and not profile.speed_2x:
            factor = max(0.5, min(2.0, profile.time_stretch))
            filters.append(f"atempo={factor}")
            # Compensate pitch back to original
            inv_semitones = -12 * (factor - 1.0) * 0.5  # approximation
            if abs(inv_semitones) > 0.01:
                inv_ratio = 2 ** (inv_semitones / 12.0)
                filters.append(f"asetrate={int(44100 * inv_ratio)}")
                filters.append("aresample=44100")

        # ── Layer 1: Speed 2.3x ──
        if profile.speed_2x:
            filters.append("atempo=2.0")
            filters.append("atempo=1.15")

        # ── Layer 1: Stereo widening ──
        if channels >= 2 and profile.stereo_width is not None and profile.stereo_width != 1.0:
            filters.append(f"extrastereo=m={profile.stereo_width}")

        # ── Layer 3: Phase jitter ──
        if channels >= 2 and profile.phase_jitter_ms is not None:
            filters.append(f"adelay={profile.phase_jitter_ms}|0")

        return ",".join(filters)

    # ── Micro-cuts ────────────────────────────────────────────────────────────

    def _generate_cut_points(self, duration: float, num_cuts: int) -> list[tuple[float, float]]:
        if duration < 10:
            return []
        safe_start, safe_end = 2.0, duration - 2.0
        if safe_end - safe_start < 1.0:
            return []
        cuts, attempts = [], 0
        while len(cuts) < num_cuts and attempts < num_cuts * 15:
            attempts += 1
            start   = round(random.uniform(safe_start, safe_end - 0.06), 4)
            end     = round(start + random.uniform(0.010, 0.050), 4)
            if end >= safe_end:
                continue
            if any(not (end + 0.02 < cs or start - 0.02 > ce) for cs, ce in cuts):
                continue
            cuts.append((start, end))
        return sorted(cuts)

    def _apply_micro_cuts(self, input_path: str, output_path: str,
                          num_cuts: int, strip_metadata: bool = True,
                          codec: str = "libvorbis", quality: str = "4") -> str:
        duration = self._get_duration(input_path)
        cuts = self._generate_cut_points(duration, num_cuts)
        if not cuts:
            return input_path
        exclude = [f"between(t,{s},{e})" for s, e in cuts]
        exclude_expr = "+".join(exclude)
        fc = f"aselect='not({exclude_expr})',aresample=async=1:first_pts=0"
        cmd = ["ffmpeg", "-i", str(input_path), "-af", fc]
        if strip_metadata:
            cmd.extend(["-map_metadata", "-1"])
        cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
        self._run_ffmpeg(cmd)
        return str(output_path)

    # ── Main entry point ──────────────────────────────────────────────────────

    def modify(self, input_path: str, output_path: str,
               profile: ModificationProfile | None = None,
               codec: str = "libvorbis", quality: str = "4") -> str:
        """
        Apply all audio modifications.

        Multi-pass pipeline:
          Pass 0 (opt): Silence padding      → WAV
          Pass 1:       Filter chain tweaks  → WAV or final OGG
          Pass 2 (opt): Bit depth dithering  → WAV
          Pass 3 (opt): Micro-cuts           → final OGG
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

        if profile is None:
            profile = ModificationProfile.random(intensity="subtle")

        channels     = self._get_channels(str(input_file))
        filter_chain = self._build_filter_chain(profile, channels)
        has_tweaks   = bool(filter_chain)
        has_cuts     = bool(profile.micro_cuts)
        has_padding  = bool(profile.silence_pad_ms and sum(profile.silence_pad_ms) > 0)
        has_dither   = bool(profile.bit_dither)
        strip        = profile.strip_metadata

        tmp_files = []

        try:
            current = str(input_file)

            # ── Pass 0: Silence padding ──
            if has_padding:
                pad_s, pad_e = profile.silence_pad_ms
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_files.append(tmp.name); tmp.close()
                self._apply_silence_padding(current, tmp.name, pad_s, pad_e, channels)
                current = tmp.name

            # ── Pass 1: Filter chain ──
            needs_more_passes = has_cuts or has_dither
            if has_tweaks:
                if needs_more_passes:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_files.append(tmp.name); tmp.close()
                    self._run_ffmpeg([
                        "ffmpeg", "-i", current,
                        "-af", filter_chain,
                        "-map_metadata", "-1",
                        "-c:a", "pcm_s16le",
                        "-y", tmp.name,
                    ])
                    current = tmp.name
                else:
                    cmd = ["ffmpeg", "-i", current, "-af", filter_chain]
                    if strip:
                        cmd.extend(["-map_metadata", "-1"])
                    cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
                    self._run_ffmpeg(cmd)
                    return str(output_path)
            elif not needs_more_passes:
                cmd = ["ffmpeg", "-i", current]
                if strip:
                    cmd.extend(["-map_metadata", "-1"])
                cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
                self._run_ffmpeg(cmd)
                return str(output_path)

            # ── Pass 2: Bit depth dithering ──
            if has_dither:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_files.append(tmp.name); tmp.close()
                self._apply_bit_dither(current, tmp.name)
                current = tmp.name

            # ── Pass 3: Micro-cuts → final output ──
            if has_cuts:
                self._apply_micro_cuts(
                    current, str(output_path),
                    num_cuts=profile.micro_cuts,
                    strip_metadata=strip,
                    codec=codec, quality=quality,
                )
            else:
                # Dither only, no cuts — encode to final format
                cmd = ["ffmpeg", "-i", current]
                if strip:
                    cmd.extend(["-map_metadata", "-1"])
                cmd.extend(["-c:a", codec, "-q:a", quality, "-y", str(output_path)])
                self._run_ffmpeg(cmd)

        finally:
            for tmp in tmp_files:
                Path(tmp).unlink(missing_ok=True)

        return str(output_path)
