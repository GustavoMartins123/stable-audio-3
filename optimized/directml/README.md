# sa3_dml — Stable Audio 3 on DirectML

Text-to-audio inference for **Stable Audio 3** on Windows GPUs through ONNX Runtime and the
DirectML execution provider. It runs the official ONNX graphs from
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized),
adapted so that every network kernel executes on the GPU.

## Features

- Models: `medium` (DiT 1.4B + SAME-L), `small-sfx` and `small-music` (DiT 433M + SAME-S).
- Reference pipeline: T5Gemma text conditioning, rectified-flow pingpong sampler (8 steps),
  classifier-free guidance with negative prompt and APG, SAME decoding in overlapping 128-frame
  windows, WAV output (float32 or 16-bit PCM).
- Local checkpoints are reused: DiT weights are read from `model.safetensors` and bound into the
  official graph. With a local Medium checkpoint only the 4 MB graph is downloaded instead of
  2.9 GB of weights.
- One static shape per session, so DirectML compiles each graph into a single fused node and no
  kernel falls back to the CPU (`--verify-placement` reports it).

## Requirements

- Windows 10/11 with a DirectX 12 GPU. Tested on an AMD Radeon RX 7600 (8 GB).
- Python 3.11–3.14.
- Access to the gated Hugging Face repositories: accept the license on the
  [model pages](https://huggingface.co/collections/stabilityai/stable-audio-3), then log in once with
  `.venv\Scripts\hf auth login`.

## Installation

```powershell
cd optimized\directml
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Configuration

| Option | Environment | Purpose |
|---|---|---|
| `--checkpoint-dir` | `SA3_CHECKPOINT_DIR` | Local checkpoint: a model folder, a `model.safetensors`, or a folder with one subfolder per model (`medium`, `small-sfx`, `small-music`). The tokenizer is read from `t5gemma-b-b-ul2/` beside it. |
| `--no-checkpoint` | | Use the weights embedded in the official ONNX instead. |
| `--onnx-dir` | | Local copy of the `onnx/` folder of the optimized repository. |
| `--cache-dir` | `SA3_DML_CACHE` | DirectML-ready graphs (default: `models/` in this folder). |
| `--offline` | | Never access the network. |
| | `HF_HOME` | Hugging Face cache location. |

Without `--checkpoint-dir` the Hugging Face cache is searched for the model checkpoint. Files fetched
on first use:

| Artifact | Download |
|---|---|
| T5Gemma encoder | 620 MB |
| Medium DiT | 4 MB with a local checkpoint, 2.9 GB without |
| Small DiT | 921 MB |
| SAME-L decoder | 1.2 GB |
| SAME-S decoder | 219 MB |

They are converted once into the cache directory, by `prepare` or by the first `generate` (about
2.9 GB for Medium, 0.9 GB per small model, 1.2 GB SAME-L, 0.2 GB SAME-S).

## Usage

```powershell
.venv\Scripts\python -m sa3_dml devices
.venv\Scripts\python -m sa3_dml prepare --model medium --checkpoint-dir path\to\StableAudio3
.venv\Scripts\python -m sa3_dml generate --model medium -p "A beautiful piano arpeggio grows into a cinematic climax" --duration 30 -o piano.wav
.venv\Scripts\python -m sa3_dml generate --model small-sfx -p "Footsteps on gravel, slow walk" --duration 10 --seed 1
.venv\Scripts\python -m sa3_dml generate --model small-sfx -p "Thunderstorm with heavy rain" --cfg-scale 2 --negative-prompt "music, voices"
.venv\Scripts\python -m sa3_dml generate --model medium -p "Lo-fi hip hop beat, 90 BPM" --verify-placement
```

| `generate` flag | Default | |
|---|---|---|
| `-p`, `--prompt` | required | Text prompt; an empty string is unconditional. |
| `--duration` | 30 | Seconds; up to 380 (medium) or 120 (small). |
| `--steps` | 8 | Sampling steps. |
| `--cfg-scale`, `--negative-prompt`, `--apg-scale` | 1.0, none, 1.0 | Guidance; 1.0 disables CFG. The released models are tuned for 1.0 and saturate above about 2. |
| `--seed` | random | Seeds the latent noise, sampler noise and decoder noise. |
| `-o`, `--output`, `--wav-format` | `output/<prompt>_<seed>.wav`, `float32` | `pcm16` for 16-bit WAV. |
| `--decoder` | model default | `same-l` or `same-s`. |
| `--decode-chunk`, `--decode-overlap` | 128, 32 | Decoder window and overlap, in latent frames. |
| `--device` | 0 | DirectX adapter index from `devices`. |
| `--verify-placement` | off | Profiles every session and lists kernels per execution provider. |
| `--provider cpu` | | Runs the same graphs on the CPU, as a numerical reference. |

## Performance

AMD Radeon RX 7600, 8 steps, wall time including graph loading with the graphs in the OS file
cache (a cold start also reads them from disk):

| Model | Audio | Time | VRAM (DiT stage) |
|---|---|---|---|
| small-sfx | 10 s | 8 s | 0.9 GiB |
| small-sfx | 120 s | 11 s | 1.4 GiB |
| medium | 30 s | 20 s | 2.9 GiB |
| medium | 120 s | 36 s | 3.8 GiB |
| medium | 380 s | 71 s | 6.8 GiB |

## Graph adaptations

- SAME-L: the 12 `samel::diff_attn_swa` TensorRT plugin nodes are replaced by blocked ±17-frame
  sliding-window differential attention made of standard ops, with 4-D GEMMs only.
- DiT: a `cross_attn_scale` input scales the cross-attention conditioning, so the CFG
  unconditional branch is an all-zero sequence and a negative prompt zeroes its padding, as in the
  reference sampler.
- SAME decoders: the in-graph `RandomNormalLike` bottleneck dither is removed and a `token_noise`
  input restores the noise the reference decoder adds to its learned upsampling tokens; both are
  drawn on the host from the run seed. The int32 PCM output is replaced by the clamped waveform.
- The DiT samples the unmasked frames of the reference sample (duration + 6 s headroom), which is
  what the reference's padding-masked attention computes.

Work done on the CPU: tokenization, random number generation, the sampler update and guidance
arithmetic (element-wise NumPy on `[256, L]` arrays), decoder window stitching, WAV encoding and the
one-time graph preparation.

## Tests

```powershell
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tests
```

## Limitations

- Text-to-audio only; audio-to-audio, inpainting and LoRA are not implemented.
- Batch size 1; CFG runs two DiT forwards per step.
- Graphs are loaded on every run (about 10 s for the Medium DiT).
- The DiT runs the official fp16 graph: per-step velocity matches the fp32 reference at 35–48 dB SNR,
  and the 8-step sampler amplifies that into a slightly different render for the same seed.
- At 380 s the Medium DiT needs about 6.8 GiB of VRAM. When other applications hold GPU memory,
  Windows moves allocations to system memory and generation slows down.

## License

Code: MIT, see the repository [LICENSE](../../LICENSE). Model weights: [Stability AI Community
License](https://stability.ai/license); the T5Gemma encoder is subject to the Gemma Terms of Use.
No weights or ONNX graphs are distributed in this repository.
