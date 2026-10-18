"""WAV output."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

WAV_FORMATS = ("float32", "pcm16")


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int, fmt: str = "float32") -> Path:
    """Write [channels, samples] audio in [-1, 1] as IEEE float32 or 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, np.float32), -1.0, 1.0)
    channels, frames = audio.shape
    if fmt == "pcm16":
        data = np.rint(audio.T * 32767.0).astype("<i2").tobytes()
        tag, bits, extra = 1, 16, b""
    elif fmt == "float32":
        data = audio.T.astype("<f4").tobytes()
        tag, bits, extra = 3, 32, struct.pack("<H", 0)
    else:
        raise ValueError(f"unknown WAV format {fmt!r}")
    block = channels * bits // 8
    fmt_chunk = struct.pack("<HHIIHH", tag, channels, sample_rate, sample_rate * block, block, bits) + extra
    chunks = [b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk]
    if tag == 3:
        chunks.append(b"fact" + struct.pack("<II", 4, frames))
    chunks.append(b"data" + struct.pack("<I", len(data)) + data)
    body = b"WAVE" + b"".join(chunks)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path
