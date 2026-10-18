"""Command-line interface: `python -m sa3_dml {generate,prepare,devices}`."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from . import dxgi
from .assets import AssetConfig, AssetError, Assets, default_cache_dir
from .audio import WAV_FORMATS, write_wav
from .pipeline import Pipeline, Request
from .session import DirectMLUnavailable, RuntimeConfig
from .specs import CODECS, MODELS

GIB = 2 ** 30


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=sorted(MODELS), default="medium",
                        help="model to run (default: medium)")
    parser.add_argument("--decoder", choices=sorted(CODECS), default=None,
                        help="SAME decoder (default: the model's own, same-l for medium, same-s for small)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="local checkpoint: a model folder, a model.safetensors file, or a folder "
                             "holding one subfolder per model (env SA3_CHECKPOINT_DIR)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="ignore local checkpoints and use the weights embedded in the official ONNX")
    parser.add_argument("--onnx-dir", type=Path, default=None,
                        help="local copy of the onnx/ folder of stabilityai/stable-audio-3-optimized")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where DirectML-ready graphs are stored (env SA3_DML_CACHE, default: ./models)")
    parser.add_argument("--offline", action="store_true", help="never access the network")
    parser.add_argument("--revision", default=None, help="Hugging Face revision of the ONNX repo")


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=int, default=0, help="DirectML adapter index (default: 0)")
    parser.add_argument("--provider", choices=("dml", "cpu"), default="dml",
                        help="execution provider; cpu only serves as a numerical reference")
    parser.add_argument("--verify-placement", action="store_true",
                        help="profile every session and report which execution provider ran each kernel")


def _assets(args) -> Assets:
    return Assets(AssetConfig(cache_dir=args.cache_dir or default_cache_dir(), onnx_dir=args.onnx_dir,
                              checkpoint=args.checkpoint_dir, offline=args.offline,
                              revision=args.revision, use_checkpoint=not args.no_checkpoint))


def _default_output(prompt: str, seed: int) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:48] or "audio"
    return Path("output") / f"{slug}_{seed}.wav"


def _progress(step: int, total: int) -> None:
    print(f"\r  sampling step {step}/{total}", end="" if step < total else "\n", flush=True)


def cmd_generate(args) -> int:
    runtime = RuntimeConfig(provider=args.provider, device_id=args.device, profile=args.verify_placement)
    pipeline = Pipeline(args.model, _assets(args), runtime, codec=args.decoder,
                        decode_chunk=args.decode_chunk, decode_overlap=args.decode_overlap)
    device = pipeline.adapter.name if pipeline.adapter else ("CPU" if args.provider == "cpu" else f"#{args.device}")
    print(f"model {args.model} | decoder {pipeline.codec.name} | provider {args.provider} ({device})")
    pipeline.graphs()
    request = Request(prompt=args.prompt, duration=args.duration, negative_prompt=args.negative_prompt,
                      steps=args.steps, cfg_scale=args.cfg_scale, apg_scale=args.apg_scale, seed=args.seed)
    started = time.perf_counter()
    result = pipeline.generate(request, on_step=_progress)
    total = time.perf_counter() - started
    output = args.output or _default_output(args.prompt, result.seed)
    write_wav(output, result.audio, result.sample_rate, args.wav_format)

    for name, stage in result.stages.items():
        line = f"  {name:8s} {stage.seconds:7.2f} s"
        if stage.vram_bytes:
            line += f"   VRAM in use {stage.vram_bytes / GIB:.2f} GiB"
        if args.verify_placement:
            line += f"   kernels: {stage.placement.describe()}"
        print(line)
    seconds = result.audio.shape[-1] / result.sample_rate
    print(f"  {result.latent_frames} latent frames, {result.forwards} DiT forwards, "
          f"{seconds:.1f} s of audio in {total:.1f} s, seed {result.seed}")
    if args.verify_placement:
        cpu = {n: s.placement.cpu_ops for n, s in result.stages.items() if s.placement.cpu_ops}
        print("  placement: " + ("all kernels ran on DirectML" if not cpu and args.provider == "dml"
                                 else f"CPU kernels in {', '.join(cpu) or 'all stages'}"))
    print(f"saved {Path(output).resolve()}")
    return 0


def cmd_prepare(args) -> int:
    assets = _assets(args)
    model = MODELS[args.model]
    codec = CODECS[args.decoder or model.codec]
    for label, fn in (("text encoder", assets.text_encoder), ("tokenizer", lambda: assets.tokenizer(model)),
                      ("dit", lambda: assets.dit(model)), ("decoder", lambda: assets.decoder(codec))):
        print(f"  {label:12s} {fn()}")
    return 0


def cmd_devices(args) -> int:
    found = dxgi.adapters()
    if not any(not a.software for a in found):
        print("no DirectX 12 hardware adapter found")
        return 1
    for adapter in found:
        kind = "software" if adapter.software else f"{adapter.dedicated_bytes / GIB:.1f} GiB dedicated"
        print(f"  --device {adapter.index}: {adapter.name} ({kind})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sa3_dml", description="Stable Audio 3 on DirectX 12 GPUs through DirectML")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="text-to-audio generation")
    gen.add_argument("-p", "--prompt", required=True, help="text prompt (empty string = unconditional)")
    gen.add_argument("--negative-prompt", default=None, help="negative prompt, used when --cfg-scale != 1")
    gen.add_argument("--duration", type=float, default=30.0, help="seconds of audio (default: 30)")
    gen.add_argument("--steps", type=int, default=8, help="sampling steps (default: 8)")
    gen.add_argument("--cfg-scale", type=float, default=1.0, help="classifier-free guidance (default: 1.0 = off)")
    gen.add_argument("--apg-scale", type=float, default=1.0,
                     help="adaptive projected guidance blend when CFG is on (1 = APG, 0 = plain CFG)")
    gen.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    gen.add_argument("-o", "--output", type=Path, default=None,
                     help="output WAV (default: output/<prompt>_<seed>.wav)")
    gen.add_argument("--wav-format", choices=WAV_FORMATS, default="float32", help="WAV sample format")
    gen.add_argument("--decode-chunk", type=int, default=None,
                     help="latent frames per decoder window (default: 128)")
    gen.add_argument("--decode-overlap", type=int, default=None,
                     help="latent frames shared by neighbouring decoder windows (default: 32)")
    _add_common(gen)
    _add_runtime(gen)
    gen.set_defaults(func=cmd_generate)

    prep = sub.add_parser("prepare", help="download and convert the graphs for a model")
    _add_common(prep)
    prep.set_defaults(func=cmd_prepare)

    dev = sub.add_parser("devices", help="list DirectX adapters and their --device index")
    dev.set_defaults(func=cmd_devices)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (AssetError, DirectMLUnavailable, ValueError, FloatingPointError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
