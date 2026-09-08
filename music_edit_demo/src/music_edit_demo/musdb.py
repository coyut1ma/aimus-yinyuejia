from __future__ import annotations

import hashlib
import json
import re
import wave
from pathlib import Path
from typing import Iterable

import av
from pydantic import BaseModel, Field

from .audio import validate_stems
from .models import AudioInfo, MusicProject, Stem


# MUSDB18 stem container convention used by stempeg/musdb.
MUSDB_STREAMS: dict[Stem, int] = {
    Stem.DRUMS: 1,
    Stem.BASS: 2,
    Stem.OTHER: 3,
    Stem.VOCALS: 4,
}


class MusdbTrack(BaseModel):
    track_id: str
    name: str
    split: str
    container_path: str
    duration_sec: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)


def _safe_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'track'}-{digest}"


def inspect_container(path: Path, split: str | None = None) -> MusdbTrack:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with av.open(str(path)) as container:
        audio_streams = list(container.streams.audio)
        if len(audio_streams) < 5:
            raise ValueError(f"{path} does not contain the five MUSDB18 audio streams")
        first = audio_streams[0]
        duration = (
            float(first.duration * first.time_base)
            if first.duration is not None
            else float(container.duration / av.time_base)
        )
        sample_rate = int(first.codec_context.sample_rate)
        channels = len(first.codec_context.layout.channels)
        for stream in audio_streams[:5]:
            if int(stream.codec_context.sample_rate) != sample_rate:
                raise ValueError(f"audio stream sample-rate mismatch in {path}")
            if len(stream.codec_context.layout.channels) != channels:
                raise ValueError(f"audio stream channel mismatch in {path}")

    name = path.name.removesuffix(".stem.mp4")
    resolved_split = split or path.parent.name
    return MusdbTrack(
        track_id=f"musdb18_{resolved_split}_{_safe_id(name)}",
        name=name,
        split=resolved_split,
        container_path=str(path),
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
    )


def scan_musdb(root: Path, splits: Iterable[str] = ("train", "test")) -> list[MusdbTrack]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    tracks: list[MusdbTrack] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.stem.mp4"), key=lambda item: item.name.lower()):
            tracks.append(inspect_container(path, split))
    if not tracks:
        raise ValueError(f"no .stem.mp4 files found under {root}")
    return tracks


def _decode_stream(container_path: Path, stream_index: int, destination: Path) -> AudioInfo:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".wav.tmp")
    frames_written = 0
    with av.open(str(container_path)) as container:
        stream = list(container.streams.audio)[stream_index]
        sample_rate = int(stream.codec_context.sample_rate)
        channels = len(stream.codec_context.layout.channels)
        if channels != 2:
            raise ValueError("the MVP importer requires stereo MUSDB18 streams")
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=sample_rate)
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            for decoded in container.decode(stream):
                for frame in resampler.resample(decoded):
                    samples = frame.to_ndarray()
                    output.writeframesraw(samples.reshape(-1).tobytes())
                    frames_written += frame.samples
            for frame in resampler.resample(None):
                samples = frame.to_ndarray()
                output.writeframesraw(samples.reshape(-1).tobytes())
                frames_written += frame.samples
    temporary.replace(destination)
    return AudioInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=2,
        frames=frames_written,
        duration_sec=frames_written / sample_rate,
    )


def import_track(track: MusdbTrack, cache_root: Path, force: bool = False) -> MusicProject:
    source = Path(track.container_path)
    track_dir = cache_root.resolve() / track.track_id
    metadata_path = track_dir / "import.json"
    fingerprint = {
        "source": str(source.resolve()),
        "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "stream_mapping": {stem.value: index for stem, index in MUSDB_STREAMS.items()},
    }
    stems = {stem: track_dir / f"{stem.value}.wav" for stem in Stem}

    cache_valid = False
    if not force and metadata_path.is_file() and all(path.is_file() for path in stems.values()):
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_valid = cached.get("fingerprint") == fingerprint

    if not cache_valid:
        track_dir.mkdir(parents=True, exist_ok=True)
        decoded_info: AudioInfo | None = None
        for stem, index in MUSDB_STREAMS.items():
            info = _decode_stream(source, index, stems[stem])
            if decoded_info is None:
                decoded_info = info
            elif info != decoded_info:
                raise ValueError(f"decoded stem mismatch for {track.name}: {stem.value}")
        metadata_path.write_text(
            json.dumps({"track": track.model_dump(mode="json"), "fingerprint": fingerprint}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    audio = validate_stems(stems)
    return MusicProject(
        project_id=track.track_id,
        stems={stem: str(path.resolve()) for stem, path in stems.items()},
        audio=audio,
    )

