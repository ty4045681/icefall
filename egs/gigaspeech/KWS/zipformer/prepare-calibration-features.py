#!/usr/bin/env python3
#
# Copyright 2026 The icefall project contributors

"""Prepare per-utterance NPZ features for static ONNX quantization.

The input can be either an audio directory or a Lhotse CutSet manifest. Stored
cut features are reused; cuts without features are computed from their
recordings. Each output NPZ contains only a float32 array named ``features``
with shape ``(num_frames, num_mel_bins)``.

Examples:

  python ./zipformer/prepare-calibration-features.py \
    --audio-dir /path/to/audio \
    --output-dir feature_npz_dir

  python ./zipformer/prepare-calibration-features.py \
    --cuts data/fbank/gigaspeech_cuts_DEV.jsonl.gz \
    --output-dir feature_npz_dir
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


AUDIO_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

FeatureRecord = Tuple[str, np.ndarray, str]


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def positive_int(value: str) -> int:
    ans = int(value)
    if ans <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return ans


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare NPZ features for static ONNX quantization.",
        formatter_class=HelpFormatter,
        epilog="""examples:
  %(prog)s --audio-dir /path/to/audio --output-dir feature_npz_dir
  %(prog)s --cuts data/fbank/gigaspeech_cuts_DEV.jsonl.gz \\
    --output-dir feature_npz_dir --max-utterances 300
""",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--audio-dir",
        type=Path,
        help="Recursively read audio files from this directory.",
    )
    source.add_argument(
        "--cuts",
        type=Path,
        help="Read a Lhotse CutSet manifest (.jsonl or .jsonl.gz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="An empty or new directory for output NPZ files.",
    )
    parser.add_argument(
        "--max-utterances",
        type=positive_int,
        default=100,
        help="Maximum number of utterances to export.",
    )
    parser.add_argument(
        "--sample-rate",
        type=positive_int,
        default=16000,
        help="Target sample rate when features need to be computed.",
    )
    parser.add_argument(
        "--num-mel-bins",
        type=positive_int,
        default=80,
        help="Expected feature dimension.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Kaldifeat device used for feature computation, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--log-interval",
        type=positive_int,
        default=10,
        help="Log progress after this many utterances.",
    )
    return parser


def discover_audio_files(audio_dir: Path) -> List[Path]:
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory does not exist: {audio_dir}")
    return sorted(
        filename
        for filename in audio_dir.rglob("*")
        if filename.is_file() and filename.suffix.lower() in AUDIO_EXTENSIONS
    )


def normalize_features(
    features: np.ndarray,
    source: str,
    num_mel_bins: int,
) -> np.ndarray:
    features = np.asarray(features)
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"Features from {source} are not floating point")
    if features.ndim != 2 or features.shape[1] != num_mel_bins:
        raise ValueError(
            f"Features from {source} have shape {features.shape}; "
            f"expected [T, {num_mel_bins}]"
        )
    if features.shape[0] == 0:
        raise ValueError(f"Features from {source} are empty")
    features = np.ascontiguousarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Features from {source} contain NaN/Inf")
    return features


def safe_source_id(source: str, max_length: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("._-")
    return (value or "utterance")[:max_length]


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory must be empty to avoid stale samples: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)


def write_feature_archive(
    output_dir: Path,
    index: int,
    source_id: str,
    features: np.ndarray,
    num_mel_bins: int,
) -> Dict[str, object]:
    features = normalize_features(features, source_id, num_mel_bins)
    filename = f"{index:06d}-{safe_source_id(source_id)}.npz"
    np.savez_compressed(output_dir / filename, features=features)
    return {
        "filename": filename,
        "source": source_id,
        "num_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }


def create_kaldifeat_extractor(
    sample_rate: int,
    num_mel_bins: int,
    device: str,
):
    try:
        import torch
        from lhotse.features.kaldifeat import (
            KaldifeatFbank,
            KaldifeatFbankConfig,
            KaldifeatFrameOptions,
            KaldifeatMelOptions,
        )
    except ImportError as error:
        raise RuntimeError(
            "Feature computation requires torch, lhotse, and kaldifeat"
        ) from error

    return KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sample_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device=torch.device(device),
        )
    )


def iter_audio_features(
    audio_dir: Path,
    max_utterances: int,
    sample_rate: int,
    num_mel_bins: int,
    device: str,
) -> Iterator[FeatureRecord]:
    try:
        from lhotse import MonoCut, Recording
    except ImportError as error:
        raise RuntimeError("Audio-directory mode requires lhotse") from error

    audio_files = discover_audio_files(audio_dir)
    if not audio_files:
        raise ValueError(f"No supported audio files found in {audio_dir}")

    extractor = create_kaldifeat_extractor(sample_rate, num_mel_bins, device)
    for index, audio_path in enumerate(audio_files[:max_utterances]):
        relative_path = audio_path.relative_to(audio_dir)
        source_id = relative_path.as_posix()
        recording_id = f"calibration-{index:06d}-{safe_source_id(source_id)}"
        try:
            recording = Recording.from_file(audio_path, recording_id)
            if recording.sampling_rate != sample_rate:
                recording = recording.resample(sample_rate)
            if len(recording.channel_ids) > 1:
                logging.warning(
                    "Use the first channel of multi-channel audio %s", audio_path
                )
            cut = MonoCut(
                id=recording_id,
                start=0.0,
                duration=recording.duration,
                channel=recording.channel_ids[0],
                recording=recording,
            )
            features = cut.compute_features(extractor=extractor)
        except Exception as error:
            raise RuntimeError(
                f"Failed to extract features from {audio_path}"
            ) from error
        yield source_id, features, "computed"


def iter_cut_features(
    cuts: Iterable[object],
    max_utterances: int,
    sample_rate: int,
    extractor_factory: Callable[[], object],
) -> Iterator[FeatureRecord]:
    extractor: Optional[object] = None
    for index, cut in enumerate(cuts):
        if index >= max_utterances:
            break
        cut_id = str(getattr(cut, "id", f"cut-{index:06d}"))
        try:
            if cut.has_features:
                features = cut.load_features()
                origin = "stored"
            else:
                if not cut.has_recording:
                    raise ValueError("cut has neither stored features nor a recording")
                if extractor is None:
                    extractor = extractor_factory()
                source_cut = cut
                if cut.sampling_rate != sample_rate:
                    source_cut = cut.resample(sample_rate)
                features = source_cut.compute_features(extractor=extractor)
                origin = "computed"
        except Exception as error:
            raise RuntimeError(f"Failed to obtain features for cut {cut_id}") from error
        yield cut_id, features, origin


def iter_manifest_features(
    cuts_filename: Path,
    max_utterances: int,
    sample_rate: int,
    num_mel_bins: int,
    device: str,
) -> Iterator[FeatureRecord]:
    if not cuts_filename.is_file():
        raise FileNotFoundError(f"CutSet manifest does not exist: {cuts_filename}")
    try:
        from lhotse import CutSet, load_manifest_lazy
    except ImportError as error:
        raise RuntimeError("CutSet mode requires lhotse") from error

    cuts = load_manifest_lazy(cuts_filename)
    if cuts is None:
        raise ValueError(f"CutSet manifest is empty: {cuts_filename}")
    if not isinstance(cuts, CutSet):
        raise TypeError(f"Manifest is not a CutSet: {cuts_filename}")

    def extractor_factory():
        return create_kaldifeat_extractor(sample_rate, num_mel_bins, device)

    yield from iter_cut_features(
        cuts=cuts,
        max_utterances=max_utterances,
        sample_rate=sample_rate,
        extractor_factory=extractor_factory,
    )


def generate_feature_directory(
    records: Iterable[FeatureRecord],
    output_dir: Path,
    num_mel_bins: int,
    log_interval: int,
    source_mode: str,
    source_path: Path,
    sample_rate: int,
) -> Dict[str, object]:
    prepare_output_dir(output_dir)
    file_records = []
    origin_counts: Dict[str, int] = {}
    for index, (source_id, features, origin) in enumerate(records):
        record = write_feature_archive(
            output_dir=output_dir,
            index=index,
            source_id=source_id,
            features=features,
            num_mel_bins=num_mel_bins,
        )
        record["origin"] = origin
        file_records.append(record)
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        if (index + 1) % log_interval == 0:
            logging.info("Prepared %s utterances", index + 1)

    if not file_records:
        raise ValueError("No calibration features were generated")

    manifest = {
        "schema_version": 1,
        "source_mode": source_mode,
        "source_path": str(source_path.resolve()),
        "sample_rate": sample_rate,
        "feature_dim": num_mel_bins,
        "num_utterances": len(file_records),
        "origin_counts": origin_counts,
        "files": file_records,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args()

    if args.audio_dir is not None:
        source_mode = "audio_dir"
        source_path = args.audio_dir
        records = iter_audio_features(
            audio_dir=args.audio_dir,
            max_utterances=args.max_utterances,
            sample_rate=args.sample_rate,
            num_mel_bins=args.num_mel_bins,
            device=args.device,
        )
    else:
        source_mode = "cuts"
        source_path = args.cuts
        records = iter_manifest_features(
            cuts_filename=args.cuts,
            max_utterances=args.max_utterances,
            sample_rate=args.sample_rate,
            num_mel_bins=args.num_mel_bins,
            device=args.device,
        )

    manifest = generate_feature_directory(
        records=records,
        output_dir=args.output_dir,
        num_mel_bins=args.num_mel_bins,
        log_interval=args.log_interval,
        source_mode=source_mode,
        source_path=source_path,
        sample_rate=args.sample_rate,
    )
    logging.info(
        "Generated %s calibration utterances in %s (%s)",
        manifest["num_utterances"],
        args.output_dir,
        manifest["origin_counts"],
    )


if __name__ == "__main__":
    main()
