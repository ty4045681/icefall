#!/usr/bin/env python3
# Copyright 2026 Xiaomi Corporation
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming PyTorch KWS inference with sample-accurate WAV export.

There are two modes.

Single-WAV mode emits one JSON object per detected keyword to stdout.  It is
compatible with ``dma_kws.inference.locators.icefall_pt.IcefallPtKwsLocator``::

  python3 streaming_kws.py \
    --checkpoint /path/to/epoch-30.pt \
    --bpe-model /path/to/bpe.model \
    --wav /path/to/input.wav \
    --keywords "HEY EVA" \
    --causal 1 --chunk-size 16 --left-context-frames 64

Manifest mode reads the dma-kws CSV contract (``audio_path,keyword[,label]``),
writes every triggered clip as a 16-kHz mono PCM WAV, and creates a companion
CSV whose ``audio_path`` values are relative to the output manifest::

  python3 streaming_kws.py \
    --checkpoint /path/to/epoch-30.pt \
    --bpe-model /path/to/bpe.model \
    --manifest /path/to/manifest.csv \
    --output-dir /path/to/output \
    --causal 1 --chunk-size 16 --left-context-frames 64

The encoder is genuinely advanced chunk by chunk with persistent Conv2d and
Zipformer caches.  Keyword-search hypotheses are also persistent, so a keyword
may start in one chunk and finish in the next one.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Mapping, Optional, Sequence, Tuple


# Unlike run.sh, dma-kws launches this file directly and does not inject the
# icefall repository into PYTHONPATH.
ZIPFORMER_RECIPE_DIR = Path(__file__).resolve().parent
KWS_RECIPE_DIR = ZIPFORMER_RECIPE_DIR.parent
ICEFALL_ROOT = Path(__file__).resolve().parents[4]
if str(ICEFALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ICEFALL_ROOT))

LOG_EPS = math.log(1e-10)
SAMPLE_RATE = 16000
FBANK_FRAME_SHIFT_MS = 10.0
ENCODER_SUBSAMPLING_FACTOR = 4

# Conv2dSubsampling consumes 7 frames and its ConvNeXt block needs three
# right-context frames at the 50-Hz rate.  This matches streaming_decode.py.
ENCODER_EMBED_PAD = 7 + 2 * 3

DEFAULT_MODEL_ARGS = {
    "num_encoder_layers": "1,1,1,1,1,1",
    "downsampling_factor": "1,2,4,8,4,2",
    "feedforward_dim": "192,192,192,192,192,192",
    "num_heads": "4,4,4,8,4,4",
    "encoder_dim": "128,128,128,128,128,128",
    "query_head_dim": "32",
    "value_head_dim": "12",
    "pos_head_dim": "4",
    "pos_dim": 48,
    "encoder_unmasked_dim": "128,128,128,128,128,128",
    "cnn_module_kernel": "31,31,15,15,15,31",
    "decoder_dim": 320,
    "joiner_dim": 320,
    "causal": True,
    "chunk_size": "16",
    "left_context_frames": "64",
    "use_transducer": True,
    "use_ctc": False,
    "context_size": 2,
}

MODEL_ARG_KEYS = tuple(DEFAULT_MODEL_ARGS)
INT_MODEL_ARGS = {"pos_dim", "decoder_dim", "joiner_dim", "context_size"}
BOOL_MODEL_ARGS = {"causal", "use_transducer", "use_ctc"}

GENERATED_MANIFEST_FIELDS = (
    "source_audio_path",
    "source_audio_path_resolved",
    "source_manifest_row",
    "hit_index",
    "detected_keyword",
    "score",
    "raw_start_sec",
    "raw_end_sec",
    "start_sec",
    "end_sec",
    "duration_sec",
    "start_sample",
    "end_sample",
    "timestamp_frames",
    "tail_anchored",
    "decode_mode",
    "chunk_size",
    "left_context_frames",
)


def str2bool(value: Any) -> bool:
    """Parse the 0/1 and true/false forms used by icefall CLIs."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: 0, 1, true, false")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run stateful streaming Zipformer KWS, or batch-decode a dma-kws "
            "manifest and export detected WAV clips."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path, help="A single input audio file.")
    source.add_argument(
        "--manifest",
        type=Path,
        help="CSV with audio_path, keyword, and optional label/metadata columns.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="An icefall KWS .pt checkpoint. May instead come from --model-config.",
    )
    parser.add_argument(
        "--bpe-model",
        type=Path,
        help="SentencePiece model. May instead come from checkpoint/config metadata.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help=(
            "Optional JSON model overrides. Both a mapping and the one-entry "
            "eval_kws.py config-list format are accepted."
        ),
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("model", "model_avg"),
        default="model",
        help="State dict to load from a training checkpoint.",
    )
    parser.add_argument(
        "--keywords",
        action="append",
        default=[],
        help=(
            "Keyword text or pre-tokenized BPE to spot; repeat for multiple "
            "keywords. BPE forms are bpe_ids:123 456, [123,456], and "
            "bpe_pieces:\u2581HEY \u2581EVA. Required in --wav mode unless supplied "
            "by --model-config. Manifest rows use their own keyword column."
        ),
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=None,
        help="Must match the checkpoint architecture.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Streaming chunk size at the 50-Hz encoder-embed rate.",
    )
    parser.add_argument(
        "--left-context-frames",
        type=int,
        default=None,
        help="Cached left context at the 50-Hz encoder-embed rate.",
    )
    parser.add_argument("--beam", "--beam-size", dest="beam", type=int, default=4)
    parser.add_argument("--keywords-score", type=float, default=None)
    parser.add_argument("--keywords-threshold", type=float, default=None)
    parser.add_argument("--num-tailing-blanks", type=int, default=1)
    parser.add_argument("--blank-penalty", type=float, default=0.0)
    parser.add_argument(
        "--tail-padding-sec",
        type=float,
        default=0.30,
        help="Synthetic silence appended only to flush a causal stream.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Required in manifest mode; generated WAVs are written below it.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="Defaults to OUTPUT_DIR/manifest.csv.",
    )
    parser.add_argument(
        "--wav-subdir",
        default="wavs",
        help="WAV directory relative to --output-dir.",
    )
    parser.add_argument(
        "--pre-roll-sec",
        type=float,
        default=0.15,
        help="Audio retained before the first detected token in manifest mode.",
    )
    parser.add_argument(
        "--post-roll-sec",
        type=float,
        default=0.15,
        help="Audio retained after the last detected token in manifest mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing generated WAVs and the output manifest.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop at the first per-audio read/decode error instead of writing "
            "partial results. CSV schema errors always fail before inference."
        ),
    )
    return parser


@dataclass
class Detection:
    phrase: str
    timestamp_frames: List[int]
    score: float

    @property
    def start_frame(self) -> int:
        return min(self.timestamp_frames)

    @property
    def end_frame(self) -> int:
        return max(self.timestamp_frames)


@dataclass
class ManifestEntry:
    row_number: int
    row: Dict[str, str]
    audio_path: Path
    keyword: str
    label: Optional[int]


@dataclass
class ClipBounds:
    raw_start_sample: int
    raw_end_sample: int
    start_sample: int
    end_sample: int
    sample_rate: int
    tail_anchored: bool = False

    @property
    def raw_start_sec(self) -> float:
        return self.raw_start_sample / self.sample_rate

    @property
    def raw_end_sec(self) -> float:
        return self.raw_end_sample / self.sample_rate

    @property
    def start_sec(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_sec(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass
class _DecoderHypothesis:
    ys: List[int]
    log_prob: Any
    context_state: Any
    timestamp: List[int] = field(default_factory=list)
    ac_probs: List[float] = field(default_factory=list)
    num_tailing_blanks: int = 0

    @property
    def key(self) -> str:
        return "_".join(str(token) for token in self.ys)


class _HypothesisList:
    def __init__(self) -> None:
        self.data: Dict[str, _DecoderHypothesis] = {}

    def add(self, hyp: _DecoderHypothesis) -> None:
        import torch

        old = self.data.get(hyp.key)
        if old is None:
            self.data[hyp.key] = hyp
        else:
            old.log_prob = torch.logaddexp(old.log_prob, hyp.log_prob)

    def values(self) -> List[_DecoderHypothesis]:
        return list(self.data.values())

    def most_probable(self) -> _DecoderHypothesis:
        return max(
            self.data.values(),
            key=lambda hyp: float(hyp.log_prob) / max(1, len(hyp.ys)),
        )


def _single_deployment_value(value: Any, default: int, name: str) -> int:
    """Resolve a saved operating point without choosing from a training list."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value]
    else:
        values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(
            "checkpoint/config records multiple {} values {}; choose one "
            "explicitly with --{}".format(name, values, name.replace("_", "-"))
        )
    return int(values[0])


def _load_json_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if isinstance(config, list):
        if len(config) != 1 or not isinstance(config[0], dict):
            raise ValueError("--model-config list must contain exactly one model")
        config = config[0]
    if not isinstance(config, dict):
        raise ValueError("--model-config must contain a JSON object")
    return dict(config)


def _resolve_candidate_path(
    value: Any, *, bases: Sequence[Path], description: str
) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("{} is required".format(description))
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [base / raw for base in bases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError("{} not found; tried: {}".format(description, attempted))


def load_manifest(path: Path) -> Tuple[List[ManifestEntry], List[str]]:
    """Read the dma-kws CSV contract and resolve paths against the CSV."""
    path = path.expanduser().resolve()
    entries: List[ManifestEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no CSV header: {}".format(path))
        fieldnames = [str(name).strip() for name in reader.fieldnames]
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError("manifest has duplicate columns after trimming whitespace")
        reader.fieldnames = fieldnames
        missing = {"audio_path", "keyword"}.difference(fieldnames)
        if missing:
            raise ValueError(
                "manifest is missing required column(s): {}".format(
                    ", ".join(sorted(missing))
                )
            )

        for row_number, raw_row in enumerate(reader, start=2):
            row = {str(key): "" if value is None else str(value) for key, value in raw_row.items()}
            raw_audio_path = row["audio_path"].strip()
            keyword = row["keyword"].strip()
            if not raw_audio_path:
                raise ValueError("manifest row {} has empty audio_path".format(row_number))
            if not keyword:
                raise ValueError("manifest row {} has empty keyword".format(row_number))

            audio_path = Path(raw_audio_path.replace("\\", os.sep)).expanduser()
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            audio_path = audio_path.resolve()

            label = None
            raw_label = row.get("label", "").strip()
            if raw_label:
                try:
                    label = int(raw_label)
                except ValueError as error:
                    raise ValueError(
                        "manifest row {} has non-integer label {!r}".format(
                            row_number, raw_label
                        )
                    ) from error
                if label not in (0, 1):
                    raise ValueError(
                        "manifest row {} label must be 0 or 1".format(row_number)
                    )

            entries.append(
                ManifestEntry(
                    row_number=row_number,
                    row=row,
                    audio_path=audio_path,
                    keyword=keyword,
                    label=label,
                )
            )
    return entries, fieldnames


def compute_clip_bounds(
    detection: Detection,
    *,
    num_samples: int,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FBANK_FRAME_SHIFT_MS,
    subsampling_factor: int = ENCODER_SUBSAMPLING_FACTOR,
    pre_roll_sec: float = 0.15,
    post_roll_sec: float = 0.15,
) -> Optional[ClipBounds]:
    """Map encoder-frame timestamps to an exact half-open sample interval."""
    if not detection.timestamp_frames:
        return None
    if num_samples <= 0:
        return None
    frame_shift_samples = round(sample_rate * frame_shift_ms / 1000.0)
    output_frame_samples = frame_shift_samples * subsampling_factor
    pre_roll_samples = round(pre_roll_sec * sample_rate)
    post_roll_samples = round(post_roll_sec * sample_rate)

    emitted_start = max(0, detection.start_frame * output_frame_samples)
    emitted_end = max(
        emitted_start + output_frame_samples,
        (detection.end_frame + 1) * output_frame_samples,
    )
    tail_anchored = emitted_start >= num_samples
    if tail_anchored:
        # A causal RNN-T may emit the final keyword tokens only while consuming
        # synthetic flush silence. Anchor that token span to the real EOF rather
        # than dropping the hit or exporting artificial audio.
        emitted_span = emitted_end - emitted_start
        raw_end = num_samples
        raw_start = max(0, raw_end - emitted_span)
    else:
        raw_start = emitted_start
        raw_end = min(num_samples, emitted_end)
    start = max(0, raw_start - pre_roll_samples)
    end = min(num_samples, raw_end + post_roll_samples)
    if raw_start >= num_samples or raw_end <= raw_start or end <= start:
        return None
    return ClipBounds(
        raw_start_sample=raw_start,
        raw_end_sample=raw_end,
        start_sample=start,
        end_sample=end,
        sample_rate=sample_rate,
        tail_anchored=tail_anchored,
    )


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem[:48] or "audio"


def make_clip_filename(
    entry: ManifestEntry, hit_index: int, bounds: ClipBounds
) -> str:
    digest = hashlib.sha1(str(entry.audio_path).encode("utf-8")).hexdigest()[:8]
    return (
        "row_{:08d}_{}_{}_hit_{:03d}_{:010d}_{:010d}.wav".format(
            entry.row_number - 2,
            _safe_stem(entry.audio_path),
            digest,
            hit_index,
            bounds.start_sample,
            bounds.end_sample,
        )
    )


def _format_seconds(value: float) -> str:
    return "{:.6f}".format(value)


def detection_json(
    detection: Detection,
    *,
    duration_samples: int,
    sample_rate: int,
    decode_mode: str,
) -> Dict[str, Any]:
    """Create the seconds-based stdout contract consumed by dma-kws."""
    bounds = compute_clip_bounds(
        detection,
        num_samples=duration_samples,
        sample_rate=sample_rate,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
    )
    if bounds is None:
        return {}
    # Do not emit a field named `timestamps`: dma-kws treats that field as
    # seconds.  `timestamp_frames` is deliberately explicit and start/end are
    # already converted to seconds.
    return {
        "keyword": detection.phrase,
        "start_time": round(bounds.raw_start_sec, 6),
        "end_time": round(bounds.raw_end_sec, 6),
        "score": round(detection.score, 8),
        "timestamp_frames": detection.timestamp_frames,
        "tail_anchored": bounds.tail_anchored,
        "decode_mode": decode_mode,
    }


def write_output_manifest(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    input_fields: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    """Write output atomically while retaining arbitrary input metadata."""
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            "output manifest already exists (pass --overwrite): {}".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["audio_path", "keyword"]
    if "label" in input_fields or any("label" in row for row in rows):
        fields.append("label")
    for name in input_fields:
        if name not in fields and name not in GENERATED_MANIFEST_FIELDS:
            fields.append(name)
    for name in GENERATED_MANIFEST_FIELDS:
        if name not in fields:
            fields.append(name)
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}-".format(path.name), suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_int_tuple(value: Any, name: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as error:
        raise ValueError(
            "{} must be an integer or comma-separated integers".format(name)
        ) from error
    if not parsed:
        raise ValueError("{} must not be empty".format(name))
    return parsed


def validate_streaming_configuration(params: Mapping[str, Any]) -> str:
    """Validate the deployed point and return streaming/offline mode."""
    causal = str2bool(params["causal"])
    chunk_size = int(params["chunk_size"])
    left_context = int(params["left_context_frames"])
    if not causal or chunk_size == -1:
        if chunk_size == -1 and left_context != -1:
            raise ValueError("chunk_size=-1 requires left_context_frames=-1")
        return "offline"
    if chunk_size <= 0:
        raise ValueError("streaming chunk_size must be positive or -1")
    if left_context <= 0:
        raise ValueError("streaming left_context_frames must be positive")

    downsampling = _parse_int_tuple(params["downsampling_factor"], "downsampling_factor")
    if any(chunk_size % factor for factor in downsampling):
        raise ValueError(
            "chunk_size={} must be divisible by every downsampling factor {}".format(
                chunk_size, downsampling
            )
        )
    kernels = _parse_int_tuple(params["cnn_module_kernel"], "cnn_module_kernel")
    if len(kernels) != len(downsampling):
        raise ValueError("cnn_module_kernel and downsampling_factor lengths differ")
    required_left_context = max(
        (kernel // 2) * factor for kernel, factor in zip(kernels, downsampling)
    )
    if left_context < required_left_context:
        raise ValueError(
            "left_context_frames={} is smaller than the convolution requirement {}".format(
                left_context, required_left_context
            )
        )
    return "streaming"


def _merge_model_args(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    values = dict(DEFAULT_MODEL_ARGS)
    for name in MODEL_ARG_KEYS:
        if name in checkpoint and checkpoint[name] is not None:
            values[name] = checkpoint[name]

    model_config = config.get("model_args", config)
    if not isinstance(model_config, Mapping):
        raise ValueError("model_args in --model-config must be a JSON object")
    for name in MODEL_ARG_KEYS:
        if name in model_config and model_config[name] is not None:
            values[name] = model_config[name]

    if args.causal is not None:
        values["causal"] = args.causal
    causal = str2bool(values["causal"])
    values["causal"] = causal

    if not causal:
        if args.chunk_size not in (None, -1) or args.left_context_frames not in (
            None,
            -1,
        ):
            raise ValueError(
                "non-causal decoding does not accept a streaming chunk/left context"
            )
        values["chunk_size"] = "-1"
        values["left_context_frames"] = "-1"
        for name in INT_MODEL_ARGS:
            values[name] = int(values[name])
        for name in BOOL_MODEL_ARGS:
            values[name] = str2bool(values[name])
        if not values["use_transducer"]:
            raise ValueError(
                "keyword search requires a checkpoint with use_transducer=true"
            )
        return values

    if args.chunk_size is not None:
        chunk_size = args.chunk_size
    else:
        chunk_size = _single_deployment_value(
            values.get("chunk_size"), 16, "chunk_size"
        )

    if args.left_context_frames is not None:
        left_context = args.left_context_frames
    else:
        left_context = _single_deployment_value(
            values.get("left_context_frames"), 64, "left_context_frames"
        )

    values["chunk_size"] = str(chunk_size)
    values["left_context_frames"] = str(left_context)
    for name in INT_MODEL_ARGS:
        values[name] = int(values[name])
    for name in BOOL_MODEL_ARGS:
        values[name] = str2bool(values[name])
    if not values["use_transducer"]:
        raise ValueError("keyword search requires a checkpoint with use_transducer=true")
    return values


@dataclass
class LoadedRuntime:
    model: Any
    sp: Any
    params: Any
    checkpoint_path: Path
    decode_mode: str
    keywords_score: float
    keywords_threshold: float
    config: Dict[str, Any]


def _torch_load(path: Path, device: Any) -> Any:
    import torch

    try:
        return torch.load(
            str(path), map_location=device, weights_only=False
        )
    except TypeError:
        # PyTorch releases before weights_only was added.
        return torch.load(str(path), map_location=device)


def _state_dict_from_checkpoint(
    checkpoint: Any, checkpoint_key: str
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a state dict or an icefall checkpoint")

    if "model" in checkpoint or "model_avg" in checkpoint:
        if checkpoint_key not in checkpoint or checkpoint[checkpoint_key] is None:
            raise ValueError(
                "checkpoint has no non-empty {!r} state dict".format(checkpoint_key)
            )
        state_dict = checkpoint[checkpoint_key]
        metadata = checkpoint
    else:
        # Exported weight-only checkpoints may be a raw state dict.
        state_dict = checkpoint
        metadata = {}
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("selected checkpoint state dict is empty or invalid")
    return state_dict, metadata


def _remove_ddp_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    keys = list(state_dict)
    if keys and all(str(key).startswith("module.") for key in keys):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _sentencepiece_unk_id(sp: Any) -> int:
    unk_value = getattr(sp, "unk_id", None)
    if callable(unk_value):
        return int(unk_value())
    if unk_value is not None:
        return int(unk_value)
    return int(sp.piece_to_id("<unk>"))


def _exact_sentencepiece_piece_id(sp: Any, piece: str) -> Optional[int]:
    """Return an ID only when both SentencePiece mappings agree exactly."""
    piece_to_id = getattr(sp, "piece_to_id", None)
    id_to_piece = getattr(sp, "id_to_piece", None)
    if not callable(piece_to_id) or not callable(id_to_piece):
        return None
    token_id = int(piece_to_id(piece))
    if token_id < 0:
        return None
    try:
        mapped_piece = str(id_to_piece(token_id))
    except (IndexError, RuntimeError):
        return None
    return token_id if mapped_piece == piece else None


def _require_sentencepiece_unk_id(sp: Any) -> int:
    """Return SentencePiece's semantic unknown ID, including custom unk_piece."""
    token_id = _sentencepiece_unk_id(sp)
    vocab_size = int(sp.get_piece_size())
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError("SentencePiece model has no valid semantic unknown ID")

    id_to_piece = getattr(sp, "id_to_piece", None)
    piece_to_id = getattr(sp, "piece_to_id", None)
    if not callable(id_to_piece) or not callable(piece_to_id):
        raise ValueError("SentencePiece processor cannot validate its unknown ID")
    try:
        unk_piece = str(id_to_piece(token_id))
    except (IndexError, RuntimeError) as error:
        raise ValueError("SentencePiece semantic unknown ID is invalid") from error
    if int(piece_to_id(unk_piece)) != token_id:
        raise ValueError("SentencePiece semantic unknown mappings are inconsistent")

    is_unknown = getattr(sp, "is_unknown", None)
    if callable(is_unknown) and not bool(is_unknown(token_id)):
        raise ValueError("SentencePiece unk_id is not marked as unknown")
    return token_id


def _require_sentencepiece_piece(sp: Any, piece: str) -> int:
    token_id = _exact_sentencepiece_piece_id(sp, piece)
    if token_id is None:
        raise ValueError(
            "SentencePiece model has no exact {!r} token".format(piece)
        )
    return token_id


def load_runtime(args: argparse.Namespace) -> LoadedRuntime:
    """Load checkpoint metadata first, then construct the matching model."""
    import sentencepiece as spm
    import torch

    from train import get_model, get_params

    config = _load_json_config(args.model_config)
    config_base = (
        args.model_config.expanduser().resolve().parent
        if args.model_config is not None
        else Path.cwd()
    )
    checkpoint_value = args.checkpoint or config.get("pt_path")
    checkpoint_path = _resolve_candidate_path(
        checkpoint_value,
        bases=(Path.cwd(), config_base),
        description="--checkpoint",
    )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    state_dict, metadata = _state_dict_from_checkpoint(
        checkpoint, args.checkpoint_key
    )
    state_dict = _remove_ddp_prefix(state_dict)

    model_args = _merge_model_args(metadata, config, args)
    decode_mode = validate_streaming_configuration(model_args)
    params = get_params()
    params.update(model_args)

    bpe_value = args.bpe_model or config.get("bpe_model") or metadata.get("bpe_model")
    bpe_path = _resolve_candidate_path(
        bpe_value,
        bases=(
            Path.cwd(),
            config_base,
            checkpoint_path.parent,
            KWS_RECIPE_DIR,
        ),
        description="--bpe-model",
    )
    sp = spm.SentencePieceProcessor()
    loaded = sp.load(str(bpe_path))
    if loaded is False:
        raise ValueError("failed to load SentencePiece model: {}".format(bpe_path))
    params.unk_id = _require_sentencepiece_unk_id(sp)
    params.blank_id = _require_sentencepiece_piece(sp, "<blk>")
    params.vocab_size = sp.get_piece_size()
    if params.blank_id == params.unk_id:
        raise ValueError("SentencePiece <blk> and <unk> IDs must differ")

    model = get_model(params)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "checkpoint/model architecture mismatch; use the original BPE and "
            "pass --model-config for a weight-only checkpoint: {}".format(error)
        ) from error
    model.to(device)
    model.eval()
    # keywords_search() looks for this optional attribute; AsrModel itself does
    # not copy unk_id from params.
    model.unk_id = params.unk_id

    keywords_score = (
        args.keywords_score
        if args.keywords_score is not None
        else float(config.get("score", 1.5))
    )
    keywords_threshold = (
        args.keywords_threshold
        if args.keywords_threshold is not None
        else float(config.get("threshold", 0.35))
    )
    if args.beam <= 0:
        raise ValueError("--beam must be positive")
    if args.num_tailing_blanks < 0:
        raise ValueError("--num-tailing-blanks must be non-negative")
    if not 0.0 <= keywords_threshold <= 1.0:
        raise ValueError("--keywords-threshold must be between 0 and 1")
    if args.tail_padding_sec < 0:
        raise ValueError("--tail-padding-sec must be non-negative")

    logging.info(
        "Loaded %s (%s, chunk=%s, left-context=%s, parameters=%s)",
        checkpoint_path,
        decode_mode,
        params.chunk_size,
        params.left_context_frames,
        sum(parameter.numel() for parameter in model.parameters()),
    )
    return LoadedRuntime(
        model=model,
        sp=sp,
        params=params,
        checkpoint_path=checkpoint_path,
        decode_mode=decode_mode,
        keywords_score=keywords_score,
        keywords_threshold=keywords_threshold,
        config=config,
    )


def _sentencepiece_vocab_size(sp: Any) -> Optional[int]:
    for name in ("get_piece_size", "vocab_size"):
        value = getattr(sp, name, None)
        if callable(value):
            return int(value())
        if value is not None:
            return int(value)
    try:
        return int(len(sp))
    except (TypeError, AttributeError):
        return None


def _parse_json_bpe_array(payload: str, *, source: str) -> Tuple[str, List[Any]]:
    try:
        values = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(
            "invalid JSON BPE keyword {!r}: {}".format(source, error.msg)
        ) from error
    if not isinstance(values, list):
        raise ValueError("BPE keyword must be a JSON array: {!r}".format(source))
    if not values:
        raise ValueError("BPE keyword must not be empty: {!r}".format(source))
    if all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        return "ids", values
    if all(isinstance(item, str) for item in values):
        return "pieces", values
    raise ValueError(
        "BPE JSON array must contain only integer IDs or only string pieces: "
        "{!r}".format(source)
    )


def _parse_keyword_spec(value: str) -> Tuple[str, Any]:
    """Return ``(text|ids|pieces, payload)`` for one keyword cell."""
    source = str(value).strip()
    if not source:
        raise ValueError("keyword must not be empty")

    tagged = re.match(
        r"^(text|bpe_ids|bpe_pieces)\s*:(.*)$", source, re.I | re.S
    )
    if tagged is not None:
        kind = tagged.group(1).lower()
        payload = tagged.group(2).strip()
        if not payload:
            raise ValueError("{} keyword must not be empty: {!r}".format(kind, source))
        if kind == "text":
            return "text", payload
        if payload.startswith("["):
            parsed_kind, values = _parse_json_bpe_array(payload, source=source)
            expected = "ids" if kind == "bpe_ids" else "pieces"
            if parsed_kind != expected:
                raise ValueError(
                    "{} requires a JSON array of {}: {!r}".format(
                        kind,
                        "integer IDs" if expected == "ids" else "string pieces",
                        source,
                    )
                )
            return expected, values
        if kind == "bpe_ids":
            if re.fullmatch(r"[+-]?\d+(?:[\s,]+[+-]?\d+)*", payload) is None:
                raise ValueError(
                    "bpe_ids must be integers separated by spaces or commas: "
                    "{!r}".format(source)
                )
            return "ids", [int(item) for item in re.split(r"[\s,]+", payload)]
        return "pieces", payload.split()

    if re.match(r"^bpe[^:]*:", source, re.I) is not None:
        raise ValueError(
            "unknown BPE keyword prefix; use bpe_ids: or bpe_pieces: in {!r}".format(
                source
            )
        )
    if source.startswith("["):
        return _parse_json_bpe_array(source, source=source)
    return "text", source


def _validate_keyword_ids(
    sp: Any,
    tokens: Sequence[int],
    *,
    source: str,
    require_vocab_size: bool,
) -> List[int]:
    token_ids = [int(token) for token in tokens]
    if not token_ids:
        raise ValueError("keyword tokenizes to an empty sequence: {!r}".format(source))

    vocab_size = _sentencepiece_vocab_size(sp)
    if require_vocab_size and vocab_size is None:
        raise ValueError(
            "SentencePiece processor cannot report its vocabulary size; "
            "cannot validate BPE IDs"
        )
    if vocab_size is not None:
        invalid = [token for token in token_ids if token < 0 or token >= vocab_size]
        if invalid:
            raise ValueError(
                "keyword contains BPE IDs outside [0, {}): {} in {!r}".format(
                    vocab_size, invalid, source
                )
            )

    unk_id = _sentencepiece_unk_id(sp)
    if unk_id >= 0 and unk_id in token_ids:
        raise ValueError(
            "keyword contains <unk> or tokens outside the SentencePiece "
            "vocabulary: {!r}".format(source)
        )

    blank_id = _exact_sentencepiece_piece_id(sp, "<blk>")
    if blank_id is not None and blank_id in token_ids:
        raise ValueError("keyword must not contain <blk>: {!r}".format(source))

    for piece in ("<sos/eos>", "<unk>"):
        special_id = _exact_sentencepiece_piece_id(sp, piece)
        if special_id is not None and special_id in token_ids:
            raise ValueError(
                "keyword must not contain reserved token {}: {!r}".format(
                    piece, source
                )
            )

    for name in ("bos_id", "eos_id", "pad_id"):
        value = getattr(sp, name, None)
        special_id = (
            int(value() if callable(value) else value) if value is not None else -1
        )
        if special_id >= 0 and special_id in token_ids:
            raise ValueError(
                "keyword must not contain SentencePiece {}: {!r}".format(name, source)
            )
    for name in ("is_control", "is_unused"):
        predicate = getattr(sp, name, None)
        if callable(predicate) and any(bool(predicate(token)) for token in token_ids):
            raise ValueError(
                "keyword must not contain SentencePiece control/unused IDs: "
                "{!r}".format(source)
            )
    return token_ids


def _pieces_to_keyword_ids(
    sp: Any, pieces: Sequence[str], *, source: str
) -> List[int]:
    if not pieces or any(not piece for piece in pieces):
        raise ValueError(
            "BPE piece list must not contain empty pieces: {!r}".format(source)
        )
    piece_to_id = getattr(sp, "piece_to_id", None)
    if not callable(piece_to_id):
        raise ValueError("SentencePiece processor cannot map BPE pieces to IDs")

    unk_id = _sentencepiece_unk_id(sp)
    id_to_piece = getattr(sp, "id_to_piece", None)
    token_ids: List[int] = []
    for piece in pieces:
        token_id = int(piece_to_id(piece))
        known_piece = None
        if callable(id_to_piece) and token_id >= 0:
            try:
                known_piece = str(id_to_piece(token_id))
            except (IndexError, RuntimeError):
                known_piece = None
        if token_id < 0 or token_id == unk_id or (
            known_piece is not None and known_piece != piece
        ):
            raise ValueError(
                "BPE piece is not in the SentencePiece vocabulary: {!r} in {!r}".format(
                    piece, source
                )
            )
        token_ids.append(token_id)
    return _validate_keyword_ids(
        sp, token_ids, source=source, require_vocab_size=True
    )


def _decode_keyword_ids(sp: Any, tokens: Sequence[int]) -> str:
    for name in ("decode", "decode_ids"):
        decoder = getattr(sp, name, None)
        if callable(decoder):
            try:
                decoded = str(decoder(list(tokens))).strip()
            except (TypeError, RuntimeError):
                continue
            if decoded:
                return decoded
    id_to_piece = getattr(sp, "id_to_piece", None)
    if callable(id_to_piece):
        return " ".join(str(id_to_piece(token)) for token in tokens)
    return "BPE_IDS[{}]".format(",".join(str(token) for token in tokens))


def encode_keywords(
    sp: Any, keywords: Sequence[str]
) -> Tuple[List[str], List[List[int]]]:
    """Encode text or validate pre-tokenized BPE keyword specifications.

    Plain values (and ``text:...``) are normalized to upper case and encoded
    with SentencePiece. Direct forms are ``bpe_ids:1 2``, ``[1,2]``,
    ``bpe_pieces:\u2581HEY \u2581EVA``, or a JSON string-piece array.
    """
    phrases: List[str] = []
    token_ids: List[List[int]] = []
    seen_tokens = set()
    for value in keywords:
        source = str(value).strip()
        if not source:
            continue
        kind, payload = _parse_keyword_spec(source)
        if kind == "text":
            phrase = str(payload).strip().upper()
            tokens = _validate_keyword_ids(
                sp,
                list(sp.encode(phrase)),
                source=source,
                require_vocab_size=False,
            )
        elif kind == "ids":
            tokens = _validate_keyword_ids(
                sp, payload, source=source, require_vocab_size=True
            )
            phrase = _decode_keyword_ids(sp, tokens)
        else:
            tokens = _pieces_to_keyword_ids(sp, payload, source=source)
            phrase = _decode_keyword_ids(sp, tokens)

        token_key = tuple(tokens)
        if token_key in seen_tokens:
            continue
        phrases.append(phrase)
        token_ids.append(tokens)
        seen_tokens.add(token_key)
    if not phrases:
        raise ValueError("at least one non-empty keyword is required")
    return phrases, token_ids


def build_keywords_graph(
    sp: Any,
    keywords: Sequence[str],
    *,
    score: float,
    threshold: float,
) -> Any:
    from icefall import ContextGraph

    phrases, token_ids = encode_keywords(sp, keywords)

    graph = ContextGraph(context_score=score, ac_threshold=threshold)
    graph.build(
        token_ids=token_ids,
        phrases=phrases,
        scores=[score] * len(phrases),
        ac_thresholds=[threshold] * len(phrases),
    )
    return graph


class StatefulKeywordDecoder:
    """The existing keywords_search loop with state retained across chunks."""

    def __init__(
        self,
        *,
        model: Any,
        keywords_graph: Any,
        beam: int,
        num_tailing_blanks: int,
        blank_penalty: float,
    ) -> None:
        import torch

        self.model = model
        self.keywords_graph = keywords_graph
        self.beam = beam
        self.num_tailing_blanks = num_tailing_blanks
        self.blank_penalty = blank_penalty
        self.blank_id = model.decoder.blank_id
        self.unk_id = getattr(model, "unk_id", self.blank_id)
        self.context_size = model.decoder.context_size
        self.device = next(model.parameters()).device
        self.torch = torch
        self.frame_offset = 0
        self.hypotheses = _HypothesisList()
        self._reset_hypotheses()

    def _reset_hypotheses(self) -> None:
        self.hypotheses = _HypothesisList()
        self.hypotheses.add(
            _DecoderHypothesis(
                ys=[-1] * (self.context_size - 1) + [self.blank_id],
                log_prob=self.torch.tensor(
                    0.0, dtype=self.torch.float32, device=self.device
                ),
                context_state=self.keywords_graph.root,
            )
        )

    def _matched_detection(
        self, hyp: _DecoderHypothesis
    ) -> Optional[Detection]:
        matched, matched_state = self.keywords_graph.is_matched(hyp.context_state)
        if not matched or matched_state is None:
            return None
        level = int(matched_state.level)
        if level <= 0 or len(hyp.ac_probs) < level or len(hyp.timestamp) < level:
            return None
        score = sum(hyp.ac_probs[-level:]) / level
        if score < float(matched_state.ac_threshold):
            return None
        return Detection(
            phrase=str(matched_state.phrase),
            timestamp_frames=hyp.timestamp[-level:],
            score=score,
        )

    def advance(self, encoder_out: Any) -> List[Detection]:
        """Consume ``(1, T, C)`` encoder frames and preserve all search state."""
        torch = self.torch
        if encoder_out.ndim != 3 or encoder_out.size(0) != 1:
            raise ValueError(
                "StatefulKeywordDecoder expects encoder_out shape (1, T, C)"
            )
        projected = self.model.joiner.encoder_proj(encoder_out)
        detections: List[Detection] = []

        for local_t in range(projected.size(1)):
            absolute_t = self.frame_offset + local_t
            active = self.hypotheses.values()
            if not active:
                raise RuntimeError("keyword beam unexpectedly became empty")

            decoder_input = torch.tensor(
                [hyp.ys[-self.context_size :] for hyp in active],
                device=self.device,
                dtype=torch.int64,
            )
            decoder_out = self.model.decoder(decoder_input, need_pad=False).unsqueeze(1)
            decoder_out = self.model.joiner.decoder_proj(decoder_out)
            current_encoder = projected[:, local_t : local_t + 1, :].unsqueeze(2)
            current_encoder = current_encoder.expand(len(active), -1, -1, -1)
            logits = self.model.joiner(
                current_encoder, decoder_out, project_input=False
            ).squeeze(1).squeeze(1)
            if self.blank_penalty:
                logits[:, self.blank_id] -= self.blank_penalty

            probs = logits.softmax(dim=-1)
            log_probs = logits.log_softmax(dim=-1)
            previous = torch.stack([hyp.log_prob for hyp in active]).reshape(-1, 1)
            combined = log_probs + previous
            vocab_size = combined.size(1)
            top_values, top_indexes = combined.reshape(-1).topk(
                min(self.beam, combined.numel())
            )

            next_hypotheses = _HypothesisList()
            for value, flat_index in zip(top_values, top_indexes):
                flat = int(flat_index.item())
                hyp_index = flat // vocab_size
                token = flat % vocab_size
                hyp = active[hyp_index]
                new_ys = hyp.ys[:]
                new_timestamps = hyp.timestamp[:]
                new_ac_probs = hyp.ac_probs[:]
                new_context_state = hyp.context_state
                context_score = 0.0
                tailing_blanks = hyp.num_tailing_blanks + 1

                if token not in (self.blank_id, self.unk_id):
                    new_ys.append(token)
                    new_timestamps.append(absolute_t)
                    new_ac_probs.append(float(probs[hyp_index, token].item()))
                    (
                        context_score,
                        new_context_state,
                        _,
                    ) = self.keywords_graph.forward_one_step(
                        hyp.context_state, token
                    )
                    tailing_blanks = 0
                    if new_context_state.token == -1:
                        new_ys[-self.context_size :] = [
                            -1
                        ] * (self.context_size - 1) + [self.blank_id]

                next_hypotheses.add(
                    _DecoderHypothesis(
                        ys=new_ys,
                        log_prob=value + context_score,
                        context_state=new_context_state,
                        timestamp=new_timestamps,
                        ac_probs=new_ac_probs,
                        num_tailing_blanks=tailing_blanks,
                    )
                )

            self.hypotheses = next_hypotheses
            top = self.hypotheses.most_probable()
            detection = self._matched_detection(top)
            # Preserve the strict `>` behavior in the existing keywords_search.
            if (
                detection is not None
                and top.num_tailing_blanks > self.num_tailing_blanks
            ):
                detections.append(detection)
                self._reset_hypotheses()

        self.frame_offset += projected.size(1)
        return detections

    def finalize(self) -> List[Detection]:
        """Match a keyword at EOS even if there are insufficient tail blanks."""
        if not self.hypotheses.data:
            return []
        detection = self._matched_detection(self.hypotheses.most_probable())
        if detection is None:
            return []
        self._reset_hypotheses()
        return [detection]


def get_init_states(model: Any, device: Any) -> List[Any]:
    import torch

    states = model.encoder.get_init_states(batch_size=1, device=device)
    states.append(model.encoder_embed.get_init_states(batch_size=1, device=device))
    states.append(torch.zeros(1, dtype=torch.int32, device=device))
    return states


def streaming_forward(
    *,
    features: Any,
    feature_lens: Any,
    model: Any,
    states: List[Any],
    chunk_size: int,
    left_context_len: int,
) -> Tuple[Any, Any, List[Any]]:
    """Advance encoder-embed and Zipformer caches by one chunk."""
    import torch

    from icefall.utils import make_pad_mask

    cached_embed_left_pad = states[-2]
    x, x_lens, new_cached_embed_left_pad = model.encoder_embed.streaming_forward(
        x=features,
        x_lens=feature_lens,
        cached_left_pad=cached_embed_left_pad,
    )
    if x.size(1) != chunk_size:
        raise RuntimeError(
            "encoder_embed produced {} frames; expected chunk_size={}".format(
                x.size(1), chunk_size
            )
        )

    src_key_padding_mask = make_pad_mask(x_lens)
    processed_mask = torch.arange(left_context_len, device=x.device).expand(
        x.size(0), left_context_len
    )
    processed_lens = states[-1]
    processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
    new_processed_lens = processed_lens + x_lens
    src_key_padding_mask = torch.cat(
        [processed_mask, src_key_padding_mask], dim=1
    )

    encoder_out, encoder_out_lens, new_encoder_states = model.encoder.streaming_forward(
        x=x.permute(1, 0, 2),
        x_lens=x_lens,
        states=states[:-2],
        src_key_padding_mask=src_key_padding_mask,
    )
    encoder_out = encoder_out.permute(1, 0, 2)
    return (
        encoder_out,
        encoder_out_lens,
        new_encoder_states + [new_cached_embed_left_pad, new_processed_lens],
    )


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> Any:
    """Load, downmix, and deterministically resample to model-input audio."""
    import soundfile as sf
    import torch

    data, source_sample_rate = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    if data.shape[0] == 0:
        raise ValueError("audio is empty: {}".format(path))
    waveform = torch.from_numpy(data).mean(dim=1).contiguous()
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("audio contains NaN or Inf: {}".format(path))
    if source_sample_rate != sample_rate:
        try:
            import torchaudio.functional as audio_functional
        except (ImportError, OSError) as error:
            raise ImportError(
                "torchaudio is required to resample {} Hz audio to {} Hz: {}".format(
                    source_sample_rate, sample_rate, path
                )
            ) from error
        waveform = audio_functional.resample(
            waveform, source_sample_rate, sample_rate
        )
    return waveform.contiguous()


def compute_fbank(waveform: Any, device: Any) -> Any:
    """Compute the normalized 80-bin feature profile used by this recipe."""
    from kaldifeat import Fbank, FbankOptions

    options = FbankOptions()
    options.device = device
    options.frame_opts.dither = 0
    options.frame_opts.snip_edges = False
    options.frame_opts.samp_freq = SAMPLE_RATE
    options.mel_opts.num_bins = 80
    options.mel_opts.high_freq = -400
    features = Fbank(options)(waveform.to(device))
    if features.ndim != 2 or features.size(0) == 0:
        raise ValueError("audio produced no usable fbank frames")
    return features


def run_keyword_inference(
    *,
    runtime: LoadedRuntime,
    features: Any,
    keywords_graph: Any,
    args: argparse.Namespace,
) -> List[Detection]:
    import torch

    decoder = StatefulKeywordDecoder(
        model=runtime.model,
        keywords_graph=keywords_graph,
        beam=args.beam,
        num_tailing_blanks=args.num_tailing_blanks,
        blank_penalty=args.blank_penalty,
    )
    device = next(runtime.model.parameters()).device
    tail_frames = round(args.tail_padding_sec * 1000.0 / FBANK_FRAME_SHIFT_MS)
    detections: List[Detection] = []

    if runtime.decode_mode == "offline":
        if features.size(0) < ENCODER_EMBED_PAD:
            return []
        padded = features
        if runtime.params.causal and tail_frames:
            padded = torch.nn.functional.pad(
                padded, (0, 0, 0, tail_frames), value=LOG_EPS
            )
        feature_lens = torch.tensor(
            [padded.size(0)], dtype=torch.int64, device=device
        )
        encoder_out, encoder_out_lens = runtime.model.forward_encoder(
            padded.unsqueeze(0), feature_lens
        )
        valid = int(encoder_out_lens[0].item())
        if valid:
            detections.extend(decoder.advance(encoder_out[:, :valid, :]))
        detections.extend(decoder.finalize())
        return detections

    chunk_size = int(runtime.params.chunk_size)
    left_context = int(runtime.params.left_context_frames)
    feature_step = chunk_size * 2
    required_segment = feature_step + ENCODER_EMBED_PAD
    padded = torch.nn.functional.pad(
        features,
        (0, 0, 0, ENCODER_EMBED_PAD + tail_frames),
        value=LOG_EPS,
    )
    states = get_init_states(runtime.model, device)
    position = 0
    while position < padded.size(0):
        segment = padded[position : position + required_segment]
        segment_length = segment.size(0)
        position += feature_step
        if segment_length < required_segment:
            extra = required_segment - segment_length
            segment = torch.nn.functional.pad(
                segment, (0, 0, 0, extra), value=LOG_EPS
            )
            # streaming_decode.py treats the final synthetic pad as valid so
            # that the encoder always returns a complete final chunk.
            segment_length += extra

        feature_lens = torch.tensor(
            [segment_length], dtype=torch.int64, device=device
        )
        encoder_out, encoder_out_lens, states = streaming_forward(
            features=segment.unsqueeze(0),
            feature_lens=feature_lens,
            model=runtime.model,
            states=states,
            chunk_size=chunk_size,
            left_context_len=left_context,
        )
        valid = int(encoder_out_lens[0].item())
        if valid:
            detections.extend(decoder.advance(encoder_out[:, :valid, :]))

    detections.extend(decoder.finalize())
    return detections


def _write_wav_atomic(path: Path, waveform: Any, overwrite: bool) -> None:
    import soundfile as sf

    if path.exists() and not overwrite:
        raise FileExistsError(
            "output WAV already exists (pass --overwrite): {}".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".{}-".format(path.stem),
        suffix=".tmp.wav",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(
            str(temporary),
            waveform.detach().cpu().numpy(),
            SAMPLE_RATE,
            subtype="PCM_16",
            format="WAV",
        )
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_output_row(
    *,
    entry: ManifestEntry,
    detection: Detection,
    hit_index: int,
    bounds: ClipBounds,
    clip_path: Path,
    output_manifest: Path,
    runtime: LoadedRuntime,
) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(entry.row)
    for name in GENERATED_MANIFEST_FIELDS:
        if name in row and row[name] != "":
            row["source_manifest_{}".format(name)] = row[name]

    row["source_audio_path"] = entry.row["audio_path"]
    row["source_audio_path_resolved"] = str(entry.audio_path)
    row["audio_path"] = Path(
        os.path.relpath(str(clip_path), str(output_manifest.parent))
    ).as_posix()
    row["keyword"] = entry.keyword
    if entry.label is not None:
        row["label"] = str(entry.label)
    row.update(
        {
            "source_manifest_row": entry.row_number,
            "hit_index": hit_index,
            "detected_keyword": detection.phrase,
            "score": "{:.8f}".format(detection.score),
            "raw_start_sec": _format_seconds(bounds.raw_start_sec),
            "raw_end_sec": _format_seconds(bounds.raw_end_sec),
            "start_sec": _format_seconds(bounds.start_sec),
            "end_sec": _format_seconds(bounds.end_sec),
            "duration_sec": _format_seconds(bounds.end_sec - bounds.start_sec),
            "start_sample": bounds.start_sample,
            "end_sample": bounds.end_sample,
            "timestamp_frames": json.dumps(detection.timestamp_frames),
            "tail_anchored": int(bounds.tail_anchored),
            "decode_mode": runtime.decode_mode,
            "chunk_size": runtime.params.chunk_size,
            "left_context_frames": runtime.params.left_context_frames,
        }
    )
    return row


def _keywords_for_single_wav(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> List[str]:
    keywords = list(args.keywords)
    if not keywords and config.get("keyword"):
        keywords = [str(config["keyword"])]
    if not keywords:
        raise ValueError("--wav mode requires --keywords (or model-config keyword)")
    return keywords


def run_single_wav(args: argparse.Namespace, runtime: LoadedRuntime) -> int:
    import torch

    wav_path = args.wav.expanduser().resolve()
    if not wav_path.is_file():
        raise FileNotFoundError("input audio not found: {}".format(wav_path))
    keywords = _keywords_for_single_wav(args, runtime.config)
    graph = build_keywords_graph(
        runtime.sp,
        keywords,
        score=runtime.keywords_score,
        threshold=runtime.keywords_threshold,
    )
    waveform = load_audio(wav_path)
    with torch.inference_mode():
        features = compute_fbank(
            waveform, next(runtime.model.parameters()).device
        )
        detections = run_keyword_inference(
            runtime=runtime,
            features=features,
            keywords_graph=graph,
            args=args,
        )
    for detection in detections:
        payload = detection_json(
            detection,
            duration_samples=waveform.numel(),
            sample_rate=SAMPLE_RATE,
            decode_mode=runtime.decode_mode,
        )
        if payload:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _validate_output_layout(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    if args.output_dir is None:
        raise ValueError("--manifest mode requires --output-dir")
    if args.pre_roll_sec < 0 or args.post_roll_sec < 0:
        raise ValueError("--pre-roll-sec and --post-roll-sec must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    output_manifest = (
        args.output_manifest.expanduser().resolve()
        if args.output_manifest is not None
        else output_dir / "manifest.csv"
    )
    wav_root = (output_dir / args.wav_subdir).resolve()
    try:
        common = Path(os.path.commonpath([str(output_dir), str(wav_root)]))
    except ValueError as error:
        raise ValueError("--wav-subdir must stay under --output-dir") from error
    if common != output_dir:
        raise ValueError("--wav-subdir must stay under --output-dir")
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(
            "output manifest already exists (pass --overwrite): {}".format(
                output_manifest
            )
        )
    return output_dir, wav_root, output_manifest


def _ensure_output_is_not_protected(
    path: Path, resolved_protected_paths: AbstractSet[Path], description: str
) -> None:
    resolved = path.expanduser().resolve()
    if resolved in resolved_protected_paths:
        raise ValueError(
            "{} must not overwrite an input/protected file: {}".format(
                description, resolved
            )
        )


def run_manifest(args: argparse.Namespace, runtime: LoadedRuntime) -> int:
    import torch

    _, wav_root, output_manifest = _validate_output_layout(args)
    input_manifest = args.manifest.expanduser().resolve()
    if output_manifest == input_manifest:
        raise ValueError("output manifest must not overwrite the input manifest")
    entries, input_fields = load_manifest(args.manifest)
    source_audio_paths = {entry.audio_path for entry in entries}
    _ensure_output_is_not_protected(
        output_manifest,
        {input_manifest}.union(source_audio_paths),
        "output manifest",
    )
    protected_clip_paths = {input_manifest, output_manifest}.union(
        source_audio_paths
    )
    logging.info("Loaded %s manifest rows from %s", len(entries), args.manifest)

    graph_cache: Dict[str, Any] = {}
    output_rows: List[Dict[str, Any]] = []
    errors: List[Tuple[int, str, str]] = []
    miss_count = 0

    for index, entry in enumerate(entries, start=1):
        try:
            if not entry.audio_path.is_file():
                raise FileNotFoundError(
                    "audio file not found: {}".format(entry.audio_path)
                )
            graph = graph_cache.get(entry.keyword)
            if graph is None:
                graph = build_keywords_graph(
                    runtime.sp,
                    [entry.keyword],
                    score=runtime.keywords_score,
                    threshold=runtime.keywords_threshold,
                )
                graph_cache[entry.keyword] = graph

            waveform = load_audio(entry.audio_path)
            with torch.inference_mode():
                features = compute_fbank(
                    waveform, next(runtime.model.parameters()).device
                )
                detections = run_keyword_inference(
                    runtime=runtime,
                    features=features,
                    keywords_graph=graph,
                    args=args,
                )
            detections.sort(key=lambda item: (item.start_frame, item.end_frame))
            written = 0
            for hit_index, detection in enumerate(detections):
                bounds = compute_clip_bounds(
                    detection,
                    num_samples=waveform.numel(),
                    sample_rate=SAMPLE_RATE,
                    pre_roll_sec=args.pre_roll_sec,
                    post_roll_sec=args.post_roll_sec,
                )
                if bounds is None:
                    logging.warning(
                        "Ignoring out-of-range hit in manifest row %s: %s",
                        entry.row_number,
                        detection.timestamp_frames,
                    )
                    continue
                filename = make_clip_filename(entry, hit_index, bounds)
                clip_path = wav_root / filename
                _ensure_output_is_not_protected(
                    clip_path, protected_clip_paths, "output WAV"
                )
                _write_wav_atomic(
                    clip_path,
                    waveform[bounds.start_sample : bounds.end_sample],
                    args.overwrite,
                )
                output_rows.append(
                    _manifest_output_row(
                        entry=entry,
                        detection=detection,
                        hit_index=hit_index,
                        bounds=bounds,
                        clip_path=clip_path,
                        output_manifest=output_manifest,
                        runtime=runtime,
                    )
                )
                written += 1
            if written == 0:
                miss_count += 1
        except Exception as error:
            if args.fail_fast:
                raise
            logging.exception("Failed manifest row %s", entry.row_number)
            errors.append((entry.row_number, str(entry.audio_path), str(error)))

        if index % 100 == 0 or index == len(entries):
            logging.info(
                "Processed %s/%s rows; clips=%s, misses=%s, errors=%s",
                index,
                len(entries),
                len(output_rows),
                miss_count,
                len(errors),
            )

    write_output_manifest(
        output_manifest,
        output_rows,
        input_fields,
        overwrite=args.overwrite,
    )
    logging.info(
        "Wrote %s clips and %s to %s (misses=%s, errors=%s)",
        len(output_rows),
        "manifest row" if len(output_rows) == 1 else "manifest rows",
        output_manifest,
        miss_count,
        len(errors),
    )
    if errors:
        preview = "; ".join(
            "row {}: {}".format(row_number, message)
            for row_number, _, message in errors[:5]
        )
        logging.error("%s row(s) failed: %s", len(errors), preview)
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_parser().parse_args(argv)
    runtime = load_runtime(args)
    if args.wav is not None:
        return run_single_wav(args, runtime)
    return run_manifest(args, runtime)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    try:
        sys.exit(main())
    except Exception as error:
        logging.error("%s", error)
        sys.exit(1)
