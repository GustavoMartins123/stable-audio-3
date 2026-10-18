"""SAME latent decoder with the reference overlapped chunking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .session import RuntimeConfig, Session
from .specs import AUDIO_CHANNELS, DOWNSAMPLING_RATIO
from .surgery import TOKEN_NOISE_INPUT

BOTTLENECK_DITHER = 1e-3


def chunk_starts(total: int, chunk: int, overlap: int) -> list[int]:
    hop = chunk - overlap
    starts = list(range(0, total - chunk + 1, hop))
    if starts[-1] != total - chunk:
        starts.append(total - chunk)
    return starts


class Decoder:
    """Decodes [1, 256, L] latents to [1, 2, L * 4096] audio in [-1, 1].

    Latents longer than `chunk` are decoded in windows of `chunk` frames overlapping by `overlap`,
    keeping only the centre of each window, so the session runs at one static shape. As in the
    reference decoder, every window draws from `rng` a bottleneck dither (1e-3 of the latent
    scale) and Gaussian noise of std `token_noise` on the learned upsampling tokens.
    """

    def __init__(self, graph: Path, runtime: RuntimeConfig, length: int, chunk: int, overlap: int,
                 token_noise: float):
        if chunk % 2 or overlap % 2 or not 0 <= overlap < chunk:
            raise ValueError("decode chunk and overlap must be even, with overlap < chunk")
        self.chunked = length >= chunk
        self.chunk, self.overlap = chunk, overlap
        self.token_noise = token_noise
        frames = chunk if self.chunked else length
        self.session = Session(graph, runtime, dims={"L": frames}, name="decoder")
        self.dtype = self.session.input_dtype("latent")
        shape = self.session.inputs[TOKEN_NOISE_INPUT].shape
        self.token_shape = tuple(frames if isinstance(d, str) or d is None else int(d) for d in shape)

    def _decode(self, latents: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
        if rng is not None:
            latents = latents + BOTTLENECK_DITHER * rng.standard_normal(latents.shape, dtype=np.float32)
            tokens = self.token_noise * rng.standard_normal(self.token_shape, dtype=np.float32)
        else:
            tokens = np.zeros(self.token_shape, np.float32)
        feeds = {"latent": latents.astype(self.dtype), TOKEN_NOISE_INPUT: tokens}
        return self.session.run(feeds)[0].astype(np.float32)

    def __call__(self, latents: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        total = latents.shape[-1]
        if not self.chunked:
            return self._decode(latents, rng)
        spl = DOWNSAMPLING_RATIO
        starts = chunk_starts(total, self.chunk, self.overlap)
        out = np.zeros((1, AUDIO_CHANNELS, total * spl), np.float32)
        half = (self.overlap // 2) * spl
        size = self.chunk * spl
        for i, start in enumerate(starts):
            audio = self._decode(latents[..., start:start + self.chunk], rng)
            last = i == len(starts) - 1
            offset = total * spl - size if last else start * spl
            left = 0 if i == 0 else half
            right = size if last else size - half
            out[..., offset + left:offset + right] = audio[..., left:right]
        return out

    def close(self):
        return self.session.close()
