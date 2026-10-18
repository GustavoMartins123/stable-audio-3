"""End-to-end Stable Audio 3 text-to-audio on ONNX Runtime + DirectML."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dxgi
from .assets import Assets
from .codec import Decoder
from .dit import DiT
from .sampling import latent_length, pingpong, schedule, valid_length
from .session import DirectMLUnavailable, Placement, RuntimeConfig
from .specs import CODECS, LATENT_DIM, MODELS, SAMPLE_RATE, ModelSpec
from .text import TextEncoder


@dataclass
class Request:
    prompt: str
    duration: float
    negative_prompt: str | None = None
    steps: int = 8
    cfg_scale: float = 1.0
    apg_scale: float = 1.0
    seed: int | None = None


@dataclass
class Stage:
    seconds: float = 0.0
    vram_bytes: int = 0
    placement: Placement = field(default_factory=Placement)


@dataclass
class Result:
    audio: np.ndarray
    sample_rate: int
    seed: int
    latent_frames: int
    forwards: int
    stages: dict[str, Stage]


@dataclass
class Graphs:
    text_encoder: Path
    tokenizer: Path
    dit: Path
    decoder: Path


class Pipeline:
    def __init__(self, model: str | ModelSpec, assets: Assets, runtime: RuntimeConfig,
                 codec: str | None = None, decode_chunk: int | None = None,
                 decode_overlap: int | None = None):
        self.model = MODELS[model] if isinstance(model, str) else model
        self.codec = CODECS[codec or self.model.codec]
        self.decode_chunk = decode_chunk or self.codec.decode_chunk
        self.decode_overlap = self.codec.decode_overlap if decode_overlap is None else decode_overlap
        self.assets = assets
        self.runtime = runtime
        self._graphs: Graphs | None = None
        found = dxgi.adapters() if runtime.provider == "dml" else []
        self.adapter = found[runtime.device_id] if 0 <= runtime.device_id < len(found) else None
        if self.adapter is not None and self.adapter.software:
            raise DirectMLUnavailable(f"adapter {runtime.device_id} ({self.adapter.name}) is a software "
                                      "renderer; pick a GPU from `python -m sa3_dml devices`")

    def graphs(self) -> Graphs:
        if self._graphs is None:
            self._graphs = Graphs(
                text_encoder=self.assets.text_encoder(),
                tokenizer=self.assets.tokenizer(self.model),
                dit=self.assets.dit(self.model),
                decoder=self.assets.decoder(self.codec),
            )
        return self._graphs

    def _vram(self) -> int:
        if self.adapter is None:
            return 0
        try:
            usage = dxgi.memory_usage(self.adapter.luid)
        except OSError:
            return 0
        return usage.local_usage if usage else 0

    def generate(self, request: Request,
                 on_step: Callable[[int, int], None] | None = None) -> Result:
        if not 0 < request.duration <= self.model.max_seconds:
            raise ValueError(f"duration must be in (0, {self.model.max_seconds:.1f}] s for {self.model.name}")
        if request.steps < 1:
            raise ValueError("steps must be >= 1")
        graphs = self.graphs()
        seed = request.seed if request.seed is not None else random.randrange(2 ** 31)
        noise_rng, sampler_rng, dither_rng = (
            np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(3))
        length = latent_length(request.duration, self.model.sample_size)
        frames = valid_length(request.duration, length)

        stages = {name: Stage() for name in ("text", "dit", "decoder")}

        start = time.perf_counter()
        encoder = TextEncoder(graphs.text_encoder, graphs.tokenizer, self.runtime)
        try:
            prompt = encoder.encode(request.prompt)
            negative = None
            if request.cfg_scale != 1.0 and request.negative_prompt is not None:
                negative = encoder.encode(request.negative_prompt)
            stages["text"].vram_bytes = self._vram()
        finally:
            stages["text"].placement = encoder.close()
        stages["text"].seconds = time.perf_counter() - start

        start = time.perf_counter()
        dit = DiT(graphs.dit, self.runtime, frames, prompt, request.duration, negative=negative,
                  cfg_scale=request.cfg_scale, apg_scale=request.apg_scale)
        try:
            noise = noise_rng.standard_normal((1, LATENT_DIM, frames), dtype=np.float32)
            latents = pingpong(dit, noise, schedule(request.steps), sampler_rng, on_step)
            stages["dit"].vram_bytes = self._vram()
        finally:
            stages["dit"].placement = dit.close()
        stages["dit"].seconds = time.perf_counter() - start
        if not np.isfinite(latents).all():
            raise FloatingPointError("the diffusion model produced non-finite latents")

        start = time.perf_counter()
        decoder = Decoder(graphs.decoder, self.runtime, length, self.decode_chunk, self.decode_overlap,
                          self.codec.token_noise)
        try:
            padded = np.pad(latents, ((0, 0), (0, 0), (0, length - frames)), mode="edge")
            audio = decoder(padded, dither_rng)
            stages["decoder"].vram_bytes = self._vram()
        finally:
            stages["decoder"].placement = decoder.close()
        stages["decoder"].seconds = time.perf_counter() - start

        audio = np.clip(audio[0, :, :int(request.duration * SAMPLE_RATE)], -1.0, 1.0)
        return Result(audio, SAMPLE_RATE, seed, frames, dit.forwards, stages)
