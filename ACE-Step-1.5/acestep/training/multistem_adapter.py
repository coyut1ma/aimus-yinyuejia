"""Train the first-pass ACE-Step multi-stem frontend adapter on MUSDB18."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from torch.optim import AdamW

from acestep.constants import SFT_GEN_PROMPT, TASK_INSTRUCTIONS
from acestep.handler import AceStepHandler


STEM_ORDER = ("vocals", "drums", "bass", "other")
MUSDB_STREAMS = {
    "vocals": 4,
    "drums": 1,
    "bass": 2,
    "other": 3,
}
TURBO_SHIFT3_TIMESTEPS = (
    1.0,
    0.9545454545454546,
    0.9,
    0.8333333333333334,
    0.75,
    0.6428571428571429,
    0.5,
    0.3,
)


@dataclass(frozen=True)
class MusdbTrack:
    path: str
    name: str
    split: str
    duration_sec: float
    sample_rate: int


@dataclass(frozen=True)
class SegmentBatch:
    audio: torch.Tensor
    target_stem_ids: torch.Tensor
    track_names: List[str]
    start_seconds: List[float]


def _audio_streams(path: Path):
    import av

    with av.open(str(path)) as container:
        streams = list(container.streams.audio)
        if len(streams) < 5:
            raise ValueError(f"{path} does not contain the five MUSDB18 audio streams")
        first = streams[0]
        duration = (
            float(first.duration * first.time_base)
            if first.duration is not None
            else float(container.duration / av.time_base)
        )
        sample_rate = int(first.codec_context.sample_rate)
    return duration, sample_rate


def scan_musdb_tracks(
    root: Path,
    splits: Iterable[str] = ("train",),
    max_tracks: Optional[int] = None,
    min_duration_sec: float = 0.0,
) -> List[MusdbTrack]:
    """Find valid MUSDB18 ``.stem.mp4`` files under ``root``."""
    root = root.resolve()
    tracks: List[MusdbTrack] = []
    for split in splits:
        split_dir = root / split
        candidates = sorted(split_dir.glob("*.stem.mp4")) if split_dir.is_dir() else []
        if not candidates and root.is_dir() and split == "":
            candidates = sorted(root.glob("*.stem.mp4"))
        for path in candidates:
            try:
                duration, sample_rate = _audio_streams(path)
            except Exception as exc:
                logger.warning("Skipping unreadable MUSDB container {}: {}", path, exc)
                continue
            if duration < min_duration_sec:
                continue
            tracks.append(
                MusdbTrack(
                    path=str(path),
                    name=path.name.removesuffix(".stem.mp4"),
                    split=split or path.parent.name,
                    duration_sec=duration,
                    sample_rate=sample_rate,
                )
            )
            if max_tracks is not None and len(tracks) >= max_tracks:
                return tracks
    return tracks


def _channels_first_float(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 1:
        array = array[None, :]
    elif array.shape[0] > array.shape[1] and array.shape[1] <= 8:
        array = array.T
    tensor = torch.from_numpy(array)
    if not torch.is_floating_point(tensor):
        info = np.iinfo(array.dtype)
        tensor = tensor.float() / max(abs(info.min), info.max)
    else:
        tensor = tensor.float()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(2, 1)
    return tensor[:2]


def decode_musdb_stream_segment(
    container_path: Path,
    stream_index: int,
    start_sec: float,
    duration_sec: float,
    target_sample_rate: int = 48000,
) -> torch.Tensor:
    """Decode one stereo stem segment from a MUSDB18 stem container."""
    import av

    chunks: List[torch.Tensor] = []
    expected_samples = int(round(duration_sec * target_sample_rate))
    with av.open(str(container_path)) as container:
        stream = list(container.streams.audio)[stream_index]
        source_sample_rate = int(stream.codec_context.sample_rate)
        start_sample = int(round(start_sec * source_sample_rate))
        end_sample = int(round((start_sec + duration_sec) * source_sample_rate))

        try:
            seek_offset = int(start_sec / float(stream.time_base))
            container.seek(seek_offset, backward=True, any_frame=False, stream=stream)
        except Exception as exc:
            logger.debug("Seek failed for {}; decoding from stream start: {}", container_path, exc)

        cursor: Optional[int] = None
        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    frame_start = 0 if cursor is None else cursor
                else:
                    frame_start = int(round(float(frame.pts * frame.time_base) * source_sample_rate))
                samples = _channels_first_float(frame.to_ndarray())
                frame_end = frame_start + samples.shape[-1]
                take_start = max(start_sample, frame_start)
                take_end = min(end_sample, frame_end)
                if take_end > take_start:
                    chunks.append(samples[:, take_start - frame_start : take_end - frame_start])
                if frame_end >= end_sample:
                    break
                cursor = frame_end
        except Exception:
            if not chunks:
                raise
            logger.debug("Using partial decoded audio for {} stream {}", container_path, stream_index)

    if not chunks:
        return torch.zeros(2, expected_samples)
    audio = torch.cat(chunks, dim=-1)
    if source_sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, source_sample_rate, target_sample_rate)

    if audio.shape[-1] < expected_samples:
        audio = F.pad(audio, (0, expected_samples - audio.shape[-1]))
    elif audio.shape[-1] > expected_samples:
        audio = audio[:, :expected_samples]
    return torch.clamp(audio, -1.0, 1.0)


def decode_musdb_four_stem_segment(
    container_path: Path,
    start_sec: float,
    duration_sec: float,
    target_sample_rate: int = 48000,
) -> torch.Tensor:
    stems = [
        decode_musdb_stream_segment(
            container_path,
            MUSDB_STREAMS[stem],
            start_sec,
            duration_sec,
            target_sample_rate,
        )
        for stem in STEM_ORDER
    ]
    return torch.stack(stems, dim=0)


class MusdbSegmentSampler:
    """Randomly sample aligned four-stem windows from MUSDB18 containers."""

    def __init__(
        self,
        tracks: Sequence[MusdbTrack],
        segment_seconds: float,
        sample_rate: int = 48000,
        seed: int = 42,
    ) -> None:
        if not tracks:
            raise ValueError("at least one MUSDB track is required")
        self.tracks = list(tracks)
        self.segment_seconds = float(segment_seconds)
        self.sample_rate = int(sample_rate)
        self.rng = random.Random(seed)

    def sample_one(self) -> Tuple[torch.Tensor, int, MusdbTrack, float]:
        last_error: Optional[Exception] = None
        for _ in range(12):
            track = self.rng.choice(self.tracks)
            latest_start = max(0.0, track.duration_sec - self.segment_seconds - 0.25)
            start_sec = self.rng.uniform(0.0, latest_start) if latest_start > 0 else 0.0
            target_stem_id = self.rng.randrange(len(STEM_ORDER))
            try:
                audio = decode_musdb_four_stem_segment(
                    Path(track.path),
                    start_sec,
                    self.segment_seconds,
                    self.sample_rate,
                )
                return audio, target_stem_id, track, start_sec
            except Exception as exc:
                last_error = exc
                logger.warning("Failed to decode {} at {:.2f}s: {}", track.name, start_sec, exc)
        raise RuntimeError("Could not sample a valid MUSDB segment") from last_error

    def sample_batch(self, batch_size: int) -> SegmentBatch:
        audios = []
        target_ids = []
        track_names = []
        starts = []
        for _ in range(batch_size):
            audio, target_id, track, start_sec = self.sample_one()
            audios.append(audio)
            target_ids.append(target_id)
            track_names.append(track.name)
            starts.append(start_sec)
        return SegmentBatch(
            audio=torch.stack(audios, dim=0),
            target_stem_ids=torch.tensor(target_ids, dtype=torch.long),
            track_names=track_names,
            start_seconds=starts,
        )


def _repeat_silence_latent(handler: AceStepHandler, length: int) -> torch.Tensor:
    silence = handler.silence_latent
    if silence is None:
        raise RuntimeError("handler.silence_latent is not initialized")
    silence = silence.to(device=handler.device, dtype=handler.dtype)
    if silence.shape[1] >= length:
        return silence[0, :length]
    repeats = math.ceil(length / silence.shape[1])
    return silence[0].repeat(repeats, 1)[:length]


def encode_multistem_latents(handler: AceStepHandler, audio: torch.Tensor) -> torch.Tensor:
    """VAE-encode ``[B, 4, 2, samples]`` audio to ``[B, 4, T, C]`` latents."""
    if audio.ndim != 4 or audio.shape[1] != 4:
        raise ValueError(f"audio must be shaped [B, 4, C, samples], got {tuple(audio.shape)}")
    batch_size, num_stems, channels, samples = audio.shape
    flat_audio = audio.reshape(batch_size * num_stems, channels, samples)
    flat_audio = flat_audio.to(handler.device, dtype=handler._get_vae_dtype())
    with torch.no_grad():
        with handler._load_model_context("vae"):
            latents = handler.vae.encode(flat_audio).latent_dist.sample()
    latents = latents.transpose(1, 2).to(device=handler.device, dtype=handler.dtype)
    return latents.reshape(batch_size, num_stems, latents.shape[1], latents.shape[2])


def build_repaint_masks(
    batch_size: int,
    latent_length: int,
    rng: random.Random,
    min_fraction: float = 0.25,
    max_fraction: float = 0.65,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build boolean masks where True means the local region to generate."""
    if latent_length <= 0:
        raise ValueError("latent_length must be positive")
    min_frames = max(1, int(round(latent_length * min_fraction)))
    max_frames = max(min_frames, int(round(latent_length * max_fraction)))
    masks = []
    for _ in range(batch_size):
        span = rng.randint(min_frames, min(max_frames, latent_length))
        start = rng.randint(0, latent_length - span)
        mask = torch.zeros(latent_length, dtype=torch.bool, device=device)
        mask[start : start + span] = True
        masks.append(mask)
    return torch.stack(masks, dim=0)


def gather_target_latents(multi_stem_latents: torch.Tensor, target_stem_ids: torch.Tensor) -> torch.Tensor:
    batch_size, _num_stems, latent_length, channels = multi_stem_latents.shape
    gather_idx = target_stem_ids.to(multi_stem_latents.device).view(batch_size, 1, 1, 1)
    gather_idx = gather_idx.expand(-1, 1, latent_length, channels)
    return multi_stem_latents.gather(dim=1, index=gather_idx).squeeze(1)


def build_training_context(
    handler: AceStepHandler,
    multi_stem_latents: torch.Tensor,
    target_stem_ids: torch.Tensor,
    repaint_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create target/context tensors matching the joint frontend contract."""
    model = handler.model
    target_latents = gather_target_latents(multi_stem_latents, target_stem_ids)
    batch_size, latent_length, channels = target_latents.shape
    silence = _repeat_silence_latent(handler, latent_length)
    src_latents = target_latents.clone()
    src_latents = torch.where(repaint_masks.to(src_latents.device).unsqueeze(-1), silence.unsqueeze(0), src_latents)
    chunk_masks = repaint_masks.to(device=src_latents.device, dtype=src_latents.dtype)
    context_latents = torch.cat(
        [src_latents, chunk_masks.unsqueeze(-1).expand(batch_size, latent_length, channels)],
        dim=-1,
    )
    multi_stem_chunk_masks = repaint_masks.to(multi_stem_latents.device).unsqueeze(1).expand(-1, 4, -1)
    multi_stem_context_latents = model.prepare_multi_stem_context_latents(
        multi_stem_latents,
        multi_stem_chunk_masks,
    )
    return target_latents, context_latents, multi_stem_context_latents


class ConditioningCache:
    """Cache frozen text/timbre condition tensors by batch size and duration."""

    def __init__(
        self,
        handler: AceStepHandler,
        caption: str,
        lyrics: str,
        language: str,
    ) -> None:
        self.handler = handler
        self.caption = caption
        self.lyrics = lyrics
        self.language = language
        self._cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, batch_size: int, duration_seconds: float) -> Tuple[torch.Tensor, torch.Tensor]:
        duration_key = int(round(duration_seconds))
        key = (batch_size, duration_key)
        if key not in self._cache:
            self._cache[key] = self._build(batch_size, duration_key)
        return self._cache[key]

    def _build(self, batch_size: int, duration_seconds: int) -> Tuple[torch.Tensor, torch.Tensor]:
        handler = self.handler
        model = handler.model
        meta = (
            "- bpm: N/A\n"
            "- timesignature: N/A\n"
            "- keyscale: N/A\n"
            f"- duration: {duration_seconds} seconds\n"
        )
        prompt = SFT_GEN_PROMPT.format(TASK_INSTRUCTIONS["repaint"], self.caption, meta)
        lyric_prompt = f"# Languages\n{self.language}\n\n# Lyric\n{self.lyrics}<|endoftext|>"
        text_tokens = handler.text_tokenizer(
            [prompt] * batch_size,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        lyric_tokens = handler.text_tokenizer(
            [lyric_prompt] * batch_size,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        text_ids = text_tokens.input_ids.to(handler.device)
        text_mask = text_tokens.attention_mask.to(handler.device).bool()
        lyric_ids = lyric_tokens.input_ids.to(handler.device)
        lyric_mask = lyric_tokens.attention_mask.to(handler.device).bool()

        with torch.no_grad():
            with handler._load_model_context("text_encoder"):
                try:
                    text_hidden = handler.text_encoder(input_ids=text_ids, lyric_attention_mask=None).last_hidden_state
                except TypeError:
                    text_hidden = handler.text_encoder(input_ids=text_ids).last_hidden_state
                lyric_hidden = handler.text_encoder.embed_tokens(lyric_ids)

            refer_latents = _repeat_silence_latent(handler, 750).unsqueeze(0).expand(batch_size, -1, -1)
            refer_order = torch.arange(batch_size, device=handler.device, dtype=torch.long)
            encoder_hidden_states, encoder_attention_mask = model.encoder(
                text_hidden_states=text_hidden.to(handler.dtype),
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden.to(handler.dtype),
                lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_latents.to(handler.dtype),
                refer_audio_order_mask=refer_order,
            )
        return encoder_hidden_states.detach(), encoder_attention_mask.detach()


def sample_turbo_timesteps(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    table = torch.tensor(TURBO_SHIFT3_TIMESTEPS, device=device, dtype=dtype)
    indices = torch.randint(0, table.numel(), (batch_size,), device=device)
    return table[indices]


def masked_flow_matching_loss(
    prediction: torch.Tensor,
    flow: torch.Tensor,
    repaint_masks: torch.Tensor,
    loss_region: str,
) -> torch.Tensor:
    error = (prediction - flow).pow(2)
    if loss_region == "full":
        return error.mean()
    if loss_region != "repaint":
        raise ValueError(f"unsupported loss_region: {loss_region}")
    mask = repaint_masks.to(prediction.device).unsqueeze(-1).expand_as(error)
    return error.masked_select(mask).mean()


def save_adapter_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    step: int,
    metadata: Dict[str, object],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.decoder.multi_stem_frontend.state_dict(),
        "metadata": dict(metadata, step=step, stem_order=list(STEM_ORDER)),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    path = output_dir / f"adapter_step_{step:06d}.pt"
    torch.save(checkpoint, path)
    torch.save(checkpoint, output_dir / "adapter_latest.pt")
    return path


def load_adapter_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.decoder.multi_stem_frontend.load_state_dict(state_dict, strict=True)
    return checkpoint.get("metadata", {})


def _trainable_parameters_fp32(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    params = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            with torch.no_grad():
                parameter.data = parameter.data.float()
        params.append(parameter)
    return params


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += float(grad.pow(2).sum().cpu())
    return math.sqrt(total) if total > 0 else 0.0


def train_multistem_adapter(args: argparse.Namespace) -> Dict[str, object]:
    torch.set_float32_matmul_precision("medium")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = Path(args.dataset_root)
    tracks = scan_musdb_tracks(
        dataset_root,
        splits=tuple(args.splits),
        max_tracks=args.max_tracks,
        min_duration_sec=args.segment_seconds + 0.5,
    )
    if not tracks:
        raise RuntimeError(f"No usable MUSDB18 stem files found under {dataset_root}")
    logger.info("Loaded {} MUSDB18 tracks from {}", len(tracks), dataset_root)

    if args.scan_only:
        return {"tracks": len(tracks), "output_dir": str(Path(args.output_dir))}

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=args.project_root,
        config_path=args.model_name,
        device=args.device,
        use_flash_attention=args.use_flash_attention,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        use_mlx_dit=False,
    )
    if not ok:
        raise RuntimeError(status)
    logger.info(status.splitlines()[0] if status else "Model initialized")

    model = handler.model
    trainable_names = model.enable_multi_stem_adapter_training()
    model.eval()
    model.decoder.multi_stem_frontend.train()
    if args.initial_gate is not None:
        with torch.no_grad():
            model.decoder.multi_stem_frontend.residual_gate.fill_(float(args.initial_gate))
    if args.resume:
        metadata = load_adapter_checkpoint(model, Path(args.resume))
        logger.info("Loaded adapter checkpoint {} with metadata {}", args.resume, metadata)

    trainable_params = _trainable_parameters_fp32(model)
    if not trainable_params:
        raise RuntimeError("No trainable multi-stem frontend parameters found")
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    sampler = MusdbSegmentSampler(
        tracks=tracks,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    mask_rng = random.Random(args.seed + 1000)
    conditioning = ConditioningCache(
        handler=handler,
        caption=args.caption,
        lyrics=args.lyrics,
        language=args.language,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": vars(args),
        "num_tracks": len(tracks),
        "trainable_names": trainable_names,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = torch.device(handler.device)
    dtype = handler.dtype
    device_type = device.type
    autocast_enabled = device_type in ("cuda", "xpu", "mps")
    scaler = None
    losses: List[float] = []
    last_checkpoint = None

    for step in range(1, args.max_steps + 1):
        step_start = time.time()
        batch = sampler.sample_batch(args.batch_size)
        audio = batch.audio
        target_stem_ids = batch.target_stem_ids.to(device)

        multi_stem_latents = encode_multistem_latents(handler, audio)
        latent_length = multi_stem_latents.shape[2]
        repaint_masks = build_repaint_masks(
            args.batch_size,
            latent_length,
            mask_rng,
            min_fraction=args.min_mask_fraction,
            max_fraction=args.max_mask_fraction,
            device=device,
        )
        target_latents, context_latents, multi_stem_context_latents = build_training_context(
            handler,
            multi_stem_latents,
            target_stem_ids,
            repaint_masks,
        )
        encoder_hidden_states, encoder_attention_mask = conditioning.get(
            args.batch_size,
            args.segment_seconds,
        )
        attention_mask = torch.ones(
            args.batch_size,
            latent_length,
            device=device,
            dtype=dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
            noise = torch.randn_like(target_latents)
            timesteps = sample_turbo_timesteps(args.batch_size, device, dtype)
            xt = timesteps.view(-1, 1, 1) * noise + (1.0 - timesteps.view(-1, 1, 1)) * target_latents
            prediction = model.decoder(
                hidden_states=xt,
                timestep=timesteps,
                timestep_r=timesteps,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False,
                joint_frontend=True,
                target_stem_id=target_stem_ids,
                multi_stem_context_latents=multi_stem_context_latents,
            )[0]
            flow = noise - target_latents
            loss = masked_flow_matching_loss(prediction, flow, repaint_masks, args.loss_region)

        loss = loss.float()
        loss.backward()
        grad_norm = _grad_norm(trainable_params)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        gate = float(model.decoder.multi_stem_frontend.residual_gate.detach().cpu())
        if step == 1 or step % args.log_every == 0:
            target_names = [STEM_ORDER[int(idx)] for idx in batch.target_stem_ids.tolist()]
            logger.info(
                "step {}/{} loss={:.6f} gate={:.6g} grad_norm={:.4f} target={} track={} time={:.2f}s",
                step,
                args.max_steps,
                loss_value,
                gate,
                grad_norm,
                target_names,
                batch.track_names[0] if batch.track_names else "",
                time.time() - step_start,
            )
        if args.save_every > 0 and step % args.save_every == 0:
            last_checkpoint = save_adapter_checkpoint(
                model,
                output_dir,
                step,
                metadata,
                optimizer=optimizer if args.save_optimizer else None,
            )
            logger.info("Saved adapter checkpoint to {}", last_checkpoint)

    final_checkpoint = save_adapter_checkpoint(
        model,
        output_dir,
        args.max_steps,
        dict(metadata, final=True, mean_loss=sum(losses) / max(len(losses), 1)),
        optimizer=optimizer if args.save_optimizer else None,
    )
    return {
        "tracks": len(tracks),
        "steps": args.max_steps,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "final_loss": losses[-1] if losses else None,
        "checkpoint": str(final_checkpoint),
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "output_dir": str(output_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/data_6/lcx/mus")
    parser.add_argument("--dataset-root", default="/data_6/lcx/mus/musdb18")
    parser.add_argument("--output-dir", default="/data_6/lcx/mus/outputs/multistem_adapter")
    parser.add_argument("--model-name", default="acestep-v15-turbo")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--segment-seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-mask-fraction", type=float, default=0.25)
    parser.add_argument("--max-mask-fraction", type=float, default=0.65)
    parser.add_argument("--loss-region", choices=["repaint", "full"], default="repaint")
    parser.add_argument("--caption", default="Full band multitrack music with vocals, drums, bass, and other instruments.")
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--initial-gate", type=float, default=0.0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_steps < 1 and not args.scan_only:
        parser.error("--max-steps must be >= 1")
    if args.segment_seconds <= 0:
        parser.error("--segment-seconds must be positive")
    if args.min_mask_fraction <= 0 or args.max_mask_fraction > 1 or args.min_mask_fraction > args.max_mask_fraction:
        parser.error("mask fractions must satisfy 0 < min <= max <= 1")
    result = train_multistem_adapter(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
