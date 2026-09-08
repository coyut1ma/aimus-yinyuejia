from __future__ import annotations

import math
import shutil
import wave
from array import array
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import AudioInfo, EditRegion, Stem


PCM16_MIN = -32768
PCM16_MAX = 32767


def _clamp_pcm16(value: float) -> int:
    return max(PCM16_MIN, min(PCM16_MAX, round(value)))


def read_wav(path: Path) -> tuple[AudioInfo, array]:
    with wave.open(str(path), "rb") as wav:
        info = AudioInfo(
            sample_rate=wav.getframerate(),
            channels=wav.getnchannels(),
            sample_width=wav.getsampwidth(),
            frames=wav.getnframes(),
            duration_sec=wav.getnframes() / wav.getframerate(),
        )
        if info.sample_width != 2:
            raise ValueError(f"{path} must be 16-bit PCM WAV for the MVP scaffold")
        samples = array("h")
        samples.frombytes(wav.readframes(info.frames))
    return info, samples


def write_wav(path: Path, info: AudioInfo, samples: Iterable[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array("h", samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(info.channels)
        wav.setsampwidth(info.sample_width)
        wav.setframerate(info.sample_rate)
        wav.writeframes(data.tobytes())
    return path


def validate_stems(stems: dict[Stem, Path]) -> AudioInfo:
    if set(stems) != set(Stem):
        raise ValueError("vocals, drums, bass, and other WAV files are required")
    reference: AudioInfo | None = None
    for stem in Stem:
        path = stems[stem]
        if not path.is_file():
            raise FileNotFoundError(path)
        info, _ = read_wav(path)
        if reference is None:
            reference = info
        elif (
            info.sample_rate,
            info.channels,
            info.sample_width,
            info.frames,
        ) != (
            reference.sample_rate,
            reference.channels,
            reference.sample_width,
            reference.frames,
        ):
            raise ValueError(f"stem format or duration mismatch: {path}")
    assert reference is not None
    return reference


def replace_region(
    original_path: Path,
    generated_path: Path,
    output_path: Path,
    region: EditRegion,
    fade_ms: int = 200,
) -> Path:
    original_info, original = read_wav(original_path)
    generated_info, generated = read_wav(generated_path)
    if original_info != generated_info:
        raise ValueError("generated target must match the original stem format and duration")
    if region.end_sec > original_info.duration_sec:
        raise ValueError("edit region exceeds the project duration")

    channels = original_info.channels
    start_frame = round(region.start_sec * original_info.sample_rate)
    end_frame = round(region.end_sec * original_info.sample_rate)
    fade_frames = min(
        round(fade_ms * original_info.sample_rate / 1000),
        max(1, (end_frame - start_frame) // 2),
    )
    original_np = np.asarray(original, dtype=np.int16).reshape(-1, channels)
    generated_np = np.asarray(generated, dtype=np.int16).reshape(-1, channels)
    output = original_np.copy()
    region_frames = end_frame - start_frame
    alpha = np.ones(region_frames, dtype=np.float32)
    ramp = np.sin(np.arange(fade_frames, dtype=np.float32) / fade_frames * math.pi / 2) ** 2
    alpha[:fade_frames] = ramp
    alpha[-fade_frames:] = ramp[::-1]
    blended = (
        original_np[start_frame:end_frame].astype(np.float32) * (1 - alpha[:, None])
        + generated_np[start_frame:end_frame].astype(np.float32) * alpha[:, None]
    )
    output[start_frame:end_frame] = np.clip(np.rint(blended), PCM16_MIN, PCM16_MAX).astype(np.int16)
    return write_wav(output_path, original_info, output.reshape(-1))


def copy_wav(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def mix_stems(stems: dict[Stem, Path], output_path: Path) -> tuple[Path, int]:
    info = validate_stems(stems)
    tracks = [np.asarray(read_wav(stems[stem])[1], dtype=np.int16).astype(np.int32) for stem in Stem]
    summed = np.sum(np.stack(tracks), axis=0, dtype=np.int32)
    clipped = int(np.count_nonzero((summed < PCM16_MIN) | (summed > PCM16_MAX)))
    mixed = np.clip(summed, PCM16_MIN, PCM16_MAX).astype(np.int16)
    return write_wav(output_path, info, mixed), clipped


def region_rms_db(path: Path, region: EditRegion) -> float:
    info, samples = read_wav(path)
    start = round(region.start_sec * info.sample_rate) * info.channels
    end = round(region.end_sec * info.sample_rate) * info.channels
    window = np.asarray(samples[start:end], dtype=np.int16).astype(np.float64)
    if window.size == 0:
        return -120.0
    mean_square = float(np.mean(np.square(window)))
    if mean_square <= 0:
        return -120.0
    return 20 * math.log10(math.sqrt(mean_square) / PCM16_MAX)


def create_sine_wav(
    path: Path,
    duration_sec: float,
    frequency: float,
    amplitude: float = 0.08,
    sample_rate: int = 44100,
    channels: int = 2,
) -> Path:
    frames = round(duration_sec * sample_rate)
    info = AudioInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=2,
        frames=frames,
        duration_sec=frames / sample_rate,
    )
    samples = array("h")
    peak = PCM16_MAX * amplitude
    for frame in range(frames):
        value = _clamp_pcm16(peak * math.sin(2 * math.pi * frequency * frame / sample_rate))
        samples.extend([value] * channels)
    return write_wav(path, info, samples)
