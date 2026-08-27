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
"""Batch negative-set evaluation for :mod:`streaming_kws`.

The inference and reporting layers are deliberately separate:

* ``prepare`` recursively creates a negative-only dma-kws manifest.
* ``prepare-features`` incrementally caches one CPU Fbank per unique audio.
* ``sweep`` validates exposure, invokes ``streaming_kws.py`` exactly once for
  all requested thresholds, then builds the report.
* ``report`` rebuilds metrics and charts from an existing run without model
  inference.

One input manifest row is one audio-keyword trial.  A trial can contribute
multiple false-alarm events, but is counted as triggered at most once.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import wave
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kws_eval_artifacts import (  # noqa: E402
    ARTIFACT_FILENAMES,
    ArtifactError,
    build_result_record as build_evaluation_result,
    write_evaluation_artifacts,
)
from kws_feature_cache import (  # noqa: E402
    FbankFeatureCache,
    prepare_cached_fbank,
)
from kws_progress import (  # noqa: E402
    PROGRESS_EVENT_ENV,
    PROGRESS_EVENT_PREFIX,
    ConsoleProgress,
    parse_progress_event,
)

DEFAULT_DECODE_SCRIPT = SCRIPT_DIR / "streaming_kws.py"
RUN_SCHEMA_VERSION = 1
UNCATEGORIZED = "uncategorized"
ALL_CATEGORY = "ALL"

EXPOSURE_FIELDS = (
    "source_manifest_row",
    "audio_path",
    "audio_path_resolved",
    "keyword",
    "label",
    "category",
    "duration_sec",
)
SOURCE_METRIC_FIELDS = (
    "threshold",
    "source_manifest_row",
    "audio_path",
    "audio_path_resolved",
    "keyword",
    "category",
    "duration_sec",
    "false_alarm_events",
    "triggered",
    "max_score",
)
THRESHOLD_METRIC_FIELDS = (
    "threshold",
    "keyword",
    "category",
    "source_trials",
    "unique_audio_files",
    "exposure_sec",
    "exposure_hours",
    "false_alarm_events",
    "triggered_source_trials",
    "triggered_audio_files",
    "fa_per_hour",
    "source_trial_trigger_rate",
)
OWNED_DECODER_ARGS = {
    "--manifest",
    "--output-dir",
    "--output-manifest",
    "--wav",
    "--keywords-threshold",
    "--keywords-thresholds",
    "--fail-fast",
    "--overwrite",
    "--progress",
    "--no-progress",
}


class EvaluationError(RuntimeError):
    """A concise, user-facing evaluation error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_number(value: float, digits: int = 10) -> str:
    if not math.isfinite(value):
        raise EvaluationError("cannot serialize a non-finite number")
    return format(float(value), ".{}g".format(digits))


def _format_threshold(value: Decimal) -> str:
    # streaming_kws writes thresholds with format(float(value), ".12g").
    # Canonicalizing here gives report joins the exact same stable key.
    result = format(float(value), ".12g")
    if not math.isfinite(float(result)):
        raise EvaluationError("threshold must be finite: {}".format(value))
    return result


def _atomic_write_text(path: Path, text: str, overwrite: bool = True) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            "output already exists (pass --overwrite): {}".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}-".format(path.name), suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any, overwrite: bool = True) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    overwrite: bool = True,
) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            "output already exists (pass --overwrite): {}".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}-".format(path.name), suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("CSV not found: {}".format(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EvaluationError("CSV has no header: {}".format(path))
        fieldnames = [str(name).strip() for name in reader.fieldnames]
        if not all(fieldnames) or len(fieldnames) != len(set(fieldnames)):
            raise EvaluationError(
                "CSV header contains empty or duplicate names after trimming: {}".format(
                    path
                )
            )
        reader.fieldnames = fieldnames
        rows = []
        for raw_row in reader:
            rows.append(
                {
                    str(key): "" if value is None else str(value)
                    for key, value in raw_row.items()
                }
            )
    return fieldnames, rows


def _parse_decimal(text: str, description: str) -> Decimal:
    try:
        value = Decimal(str(text).strip())
    except InvalidOperation as error:
        raise EvaluationError("invalid {} {!r}".format(description, text)) from error
    if not value.is_finite():
        raise EvaluationError("{} must be finite: {}".format(description, text))
    return value


def parse_thresholds(values: Sequence[str]) -> List[str]:
    """Parse explicit, comma-separated, and inclusive ``start:stop:step`` forms."""
    raw_tokens: List[str] = []
    for value in values:
        raw_tokens.extend(token for token in re.split(r"[\s,]+", value) if token)
    if not raw_tokens:
        raise EvaluationError("--thresholds must contain at least one value")

    expanded: List[Decimal] = []
    for token in raw_tokens:
        if ":" not in token:
            expanded.append(_parse_decimal(token, "threshold"))
            continue
        parts = token.split(":")
        if len(parts) != 3 or not all(parts):
            raise EvaluationError(
                "threshold range must be start:stop:step, got {!r}".format(token)
            )
        start, stop, step = (
            _parse_decimal(parts[0], "range start"),
            _parse_decimal(parts[1], "range stop"),
            _parse_decimal(parts[2], "range step"),
        )
        if step == 0:
            raise EvaluationError("threshold range step must not be zero")
        if (stop - start) * step < 0:
            raise EvaluationError(
                "threshold range step points away from stop: {!r}".format(token)
            )
        current = start
        # At most 100001 values prevents an accidental effectively-infinite run.
        for _ in range(100001):
            if (step > 0 and current > stop) or (step < 0 and current < stop):
                break
            expanded.append(current)
            current += step
        else:
            raise EvaluationError("threshold range is too large: {!r}".format(token))

    canonical: Dict[str, Decimal] = {}
    for value in expanded:
        if value < 0 or value > 1:
            raise EvaluationError("threshold must be between 0 and 1: {}".format(value))
        key = _format_threshold(value)
        canonical[key] = Decimal(key)
    return [key for key, _ in sorted(canonical.items(), key=lambda item: item[1])]


def _parse_extensions(values: Sequence[str]) -> List[str]:
    extensions = []
    for value in values:
        for token in re.split(r"[\s,]+", value.strip()):
            if not token:
                continue
            extension = token.lower()
            if not extension.startswith("."):
                extension = "." + extension
            extensions.append(extension)
    if not extensions:
        raise EvaluationError("--extensions must contain at least one extension")
    return sorted(set(extensions))


def _probe_wav_duration(path: Path) -> float:
    if path.suffix.lower() not in {".wav", ".wave"}:
        raise EvaluationError(
            "stdlib duration probing supports WAV only: {}".format(path)
        )
    try:
        with wave.open(str(path), "rb") as reader:
            frame_rate = reader.getframerate()
            frame_count = reader.getnframes()
    except (wave.Error, EOFError) as error:
        raise EvaluationError("cannot read WAV metadata {}: {}".format(path, error))
    if frame_rate <= 0 or frame_count <= 0:
        raise EvaluationError("WAV has no positive-duration audio: {}".format(path))
    return frame_count / float(frame_rate)


def _resolve_audio_path(raw_path: str, manifest_path: Path, row_number: int) -> Path:
    text = raw_path.strip()
    if not text:
        raise EvaluationError("manifest row {} has empty audio_path".format(row_number))
    path = Path(text.replace("\\", os.sep)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "manifest row {} audio not found: {}".format(row_number, path)
        )
    return path


def prepare_manifest(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError("input directory not found: {}".format(input_dir))
    extensions = set(_parse_extensions(args.extensions))
    audio_paths = sorted(
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    # Resolved-path de-duplication handles symlinked datasets deterministically.
    audio_paths = list(dict.fromkeys(audio_paths))
    if not audio_paths:
        raise EvaluationError(
            "no matching audio under {} (extensions: {})".format(
                input_dir, ", ".join(sorted(extensions))
            )
        )
    if output_manifest in set(audio_paths):
        raise EvaluationError(
            "output manifest must not overwrite an input audio file: {}".format(
                output_manifest
            )
        )

    keywords = []
    seen_keywords = set()
    for raw_keyword in args.keyword:
        keyword = raw_keyword.strip()
        if not keyword:
            raise EvaluationError("--keyword must not be empty")
        if keyword not in seen_keywords:
            keywords.append(keyword)
            seen_keywords.add(keyword)

    rows: List[Dict[str, Any]] = []
    for audio_path in audio_paths:
        try:
            relative = audio_path.relative_to(input_dir)
        except ValueError as error:
            raise EvaluationError(
                "resolved audio path escapes --input-dir through a symlink: "
                "{}".format(audio_path)
            ) from error
        directory_parts = relative.parts[:-1]
        if 0 <= args.category_depth < len(directory_parts):
            category = directory_parts[args.category_depth]
        elif args.category_depth < 0 and directory_parts:
            category = directory_parts[-1]
        else:
            category = UNCATEGORIZED
        if category == ALL_CATEGORY:
            raise EvaluationError(
                "category name {!r} is reserved for aggregates: {}".format(
                    ALL_CATEGORY, audio_path
                )
            )
        duration = _probe_wav_duration(audio_path)
        if args.absolute_paths:
            manifest_audio_path = str(audio_path)
        else:
            manifest_audio_path = Path(
                os.path.relpath(str(audio_path), str(output_manifest.parent))
            ).as_posix()
        for keyword in keywords:
            rows.append(
                {
                    "audio_path": manifest_audio_path,
                    "keyword": keyword,
                    "label": "0",
                    "category": category,
                    "source_duration_sec": _format_number(duration, 12),
                }
            )
    _atomic_write_csv(
        output_manifest,
        (
            "audio_path",
            "keyword",
            "label",
            "category",
            "source_duration_sec",
        ),
        rows,
        overwrite=args.overwrite,
    )
    print(
        "Prepared {} negative trials from {} audio files: {}".format(
            len(rows), len(audio_paths), output_manifest
        )
    )
    return 0


def prepare_features(args: argparse.Namespace) -> int:
    """Incrementally cache one CPU Fbank matrix per unique manifest audio."""
    import streaming_kws

    if args.num_workers <= 0:
        raise EvaluationError("--num-workers must be positive")
    entries, _ = streaming_kws.load_manifest(args.manifest)
    audio_paths = list(dict.fromkeys(entry.audio_path for entry in entries))
    FbankFeatureCache(args.feature_cache_dir)
    display = ConsoleProgress(
        total=len(audio_paths),
        description="KWS feature cache",
        enabled=getattr(args, "progress", None),
        unit="files",
    )
    counts = {"cached": 0, "computed": 0, "errors": 0}
    errors = []

    def record(path: Path, result: Optional[Mapping[str, Any]], error: Any) -> None:
        if error is not None:
            counts["errors"] += 1
            errors.append((path, str(error)))
            logging.error("Failed to cache %s: %s", path, error)
        elif result is not None:
            status = str(result.get("status", "computed"))
            counts[status if status in counts else "computed"] += 1
        completed = sum(counts.values())
        display.update(
            completed=completed,
            status="cached={} computed={} errors={}".format(
                counts["cached"], counts["computed"], counts["errors"]
            ),
        )

    display.start()
    try:
        if args.num_workers == 1:
            for audio_path in audio_paths:
                try:
                    result = prepare_cached_fbank(
                        str(audio_path),
                        str(args.feature_cache_dir),
                        args.force,
                    )
                except Exception as error:
                    record(audio_path, None, error)
                else:
                    record(audio_path, result, None)
        else:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_workers,
                mp_context=context,
            ) as executor:
                pending = {
                    executor.submit(
                        prepare_cached_fbank,
                        str(audio_path),
                        str(args.feature_cache_dir),
                        args.force,
                    ): audio_path
                    for audio_path in audio_paths
                }
                for future in concurrent.futures.as_completed(pending):
                    audio_path = pending[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        record(audio_path, None, error)
                    else:
                        record(audio_path, result, None)
    finally:
        display.stop()

    print(
        "Feature cache: {} unique files, {} reused, {} computed, {} errors: {}".format(
            len(audio_paths),
            counts["cached"],
            counts["computed"],
            counts["errors"],
            args.feature_cache_dir.expanduser().resolve(),
        )
    )
    if errors:
        preview = "; ".join("{}: {}".format(path, error) for path, error in errors[:5])
        logging.error("Feature cache failures: %s", preview)
        return 2
    return 0


def _build_exposure(
    manifest_path: Path,
    *,
    category_field: str,
    duration_field: str,
    assume_negative: bool,
) -> List[Dict[str, Any]]:
    fieldnames, rows = _read_csv(manifest_path)
    missing = {"audio_path", "keyword"}.difference(fieldnames)
    if missing:
        raise EvaluationError(
            "manifest is missing required column(s): {}".format(
                ", ".join(sorted(missing))
            )
        )
    if category_field and category_field not in fieldnames:
        raise EvaluationError(
            "manifest has no category field {!r}; pass --category-field '' "
            "to use {!r}".format(category_field, UNCATEGORIZED)
        )
    if "label" not in fieldnames and not assume_negative:
        raise EvaluationError(
            "negative evaluation requires label=0; use --assume-negative only "
            "for a verified label-free manifest"
        )

    exposure = []
    for offset, row in enumerate(rows, start=2):
        keyword = row.get("keyword", "").strip()
        if not keyword:
            raise EvaluationError("manifest row {} has empty keyword".format(offset))
        raw_label = row.get("label", "").strip()
        if raw_label:
            try:
                label = int(raw_label)
            except ValueError as error:
                raise EvaluationError(
                    "manifest row {} has invalid label={!r}".format(offset, raw_label)
                ) from error
            if label != 0:
                raise EvaluationError(
                    "manifest row {} is not negative (label={!r})".format(
                        offset, raw_label
                    )
                )
        elif not assume_negative:
            raise EvaluationError(
                "manifest row {} has no label; expected label=0".format(offset)
            )

        audio_path = _resolve_audio_path(
            row.get("audio_path", ""), manifest_path, offset
        )
        category = (
            row.get(category_field, "").strip() if category_field else ""
        ) or UNCATEGORIZED
        if category == ALL_CATEGORY:
            raise EvaluationError(
                "manifest row {} category {!r} is reserved for aggregates".format(
                    offset, ALL_CATEGORY
                )
            )
        raw_duration = row.get(duration_field, "").strip()
        if raw_duration:
            try:
                duration = float(raw_duration)
            except ValueError as error:
                raise EvaluationError(
                    "manifest row {} has invalid {}={!r}".format(
                        offset, duration_field, raw_duration
                    )
                ) from error
            if not math.isfinite(duration) or duration <= 0:
                raise EvaluationError(
                    "manifest row {} {} must be positive and finite".format(
                        offset, duration_field
                    )
                )
        else:
            duration = _probe_wav_duration(audio_path)
        exposure.append(
            {
                "source_manifest_row": offset,
                "audio_path": row["audio_path"],
                "audio_path_resolved": str(audio_path),
                "keyword": keyword,
                "label": "0",
                "category": category,
                "duration_sec": _format_number(duration, 12),
            }
        )
    if not exposure:
        raise EvaluationError("manifest contains no trials: {}".format(manifest_path))
    return exposure


def _validate_decoder_args(values: Sequence[str]) -> List[str]:
    decoder_args = list(values)
    if decoder_args and decoder_args[0] == "--":
        decoder_args = decoder_args[1:]
    for value in decoder_args:
        flag = value.split("=", 1)[0]
        owned_matches = (
            sorted(option for option in OWNED_DECODER_ARGS if option.startswith(flag))
            if flag.startswith("--")
            else []
        )
        if owned_matches:
            raise EvaluationError(
                "decoder argument {} conflicts with sweep-owned option(s): {}".format(
                    flag, ", ".join(owned_matches)
                )
            )
    return decoder_args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _path_metadata_fingerprint(
    paths: Iterable[Path], count_field: str
) -> Dict[str, Any]:
    references = sorted(
        {Path(os.path.abspath(str(path.expanduser()))) for path in paths},
        key=str,
    )
    digest = hashlib.sha256()
    resolved_paths = []
    for reference in references:
        path = reference.resolve()
        try:
            stat = path.stat()
        except OSError as error:
            raise EvaluationError(
                "cannot fingerprint input {}: {}".format(path, error)
            ) from error
        resolved_paths.append(path)
        digest.update(str(reference).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(reference\\0resolved_path\\0size\\0mtime_ns\\n)",
        count_field: len(references),
        "sha256": digest.hexdigest(),
        "references": [str(path) for path in references],
        "paths": [str(path) for path in sorted(set(resolved_paths), key=str)],
    }


def _audio_fingerprint(exposure: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return _path_metadata_fingerprint(
        (Path(str(row["audio_path_resolved"])) for row in exposure),
        "unique_audio_files",
    )


def _decoder_option_value(decoder_args: Sequence[str], option: str) -> Optional[str]:
    result: Optional[str] = None
    for index, value in enumerate(decoder_args):
        if value == option and index + 1 < len(decoder_args):
            result = decoder_args[index + 1]
        elif value.startswith(option + "="):
            result = value.split("=", 1)[1]
    return result if result and not result.startswith("--") else None


def _first_existing_path(value: Any, bases: Sequence[Path]) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [base / raw for base in bases]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _official_decoder_asset_paths(decoder_args: Sequence[str]) -> List[Path]:
    """Resolve explicit/config assets with the official streaming runner rules."""
    cwd = Path.cwd()
    config_path = _first_existing_path(
        _decoder_option_value(decoder_args, "--model-config"), (cwd,)
    )
    config: Mapping[str, Any] = {}
    if config_path is not None:
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and len(loaded) == 1:
                loaded = loaded[0]
            if isinstance(loaded, dict):
                config = loaded
        except (OSError, json.JSONDecodeError):
            # The child runner will provide the authoritative config error.
            pass
    config_base = config_path.parent if config_path is not None else cwd
    checkpoint_value = _decoder_option_value(decoder_args, "--checkpoint")
    if checkpoint_value is None:
        checkpoint_value = config.get("pt_path")
    checkpoint_path = _first_existing_path(checkpoint_value, (cwd, config_base))
    bpe_value = _decoder_option_value(decoder_args, "--bpe-model")
    if bpe_value is None:
        bpe_value = config.get("bpe_model")
    bpe_bases = [cwd, config_base]
    if checkpoint_path is not None:
        bpe_bases.append(checkpoint_path.parent)
    bpe_bases.append(SCRIPT_DIR.parent)
    bpe_path = _first_existing_path(bpe_value, bpe_bases)
    return [
        path for path in (config_path, checkpoint_path, bpe_path) if path is not None
    ]


def _decoder_input_fingerprint(
    decoder_args: Sequence[str], known_paths: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    candidates = []
    if known_paths is not None:
        candidates = [Path(value) for value in known_paths]
    else:
        for index, value in enumerate(decoder_args):
            candidate: Optional[str] = None
            if value.startswith("--") and "=" in value:
                candidate = value.split("=", 1)[1]
            elif (
                index > 0
                and decoder_args[index - 1].startswith("--")
                and not value.startswith("-")
            ):
                candidate = value
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            try:
                if path.is_file():
                    candidates.append(path)
            except OSError:
                # A broken candidate cannot be a currently usable model input.
                continue
        candidates.extend(_official_decoder_asset_paths(decoder_args))
    result = _path_metadata_fingerprint(candidates, "input_files")
    required_assets = ("--checkpoint", "--bpe-model")
    resolved_references = {str(Path(value).resolve()) for value in result["references"]}
    tracked_assets = set()
    for index, value in enumerate(decoder_args):
        for option in required_assets:
            raw_path: Optional[str] = None
            if value == option and index + 1 < len(decoder_args):
                raw_path = decoder_args[index + 1]
            elif value.startswith(option + "=") and value.split("=", 1)[1]:
                raw_path = value.split("=", 1)[1]
            if raw_path:
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = Path.cwd() / path
                if str(path.resolve()) in resolved_references:
                    tracked_assets.add(option)
    result["resume_safe"] = all(option in tracked_assets for option in required_assets)
    result["resume_requires_explicit"] = list(required_assets)
    return result


def _relative_to_run(path: Path, run_dir: Path) -> str:
    return Path(os.path.relpath(str(path), str(run_dir))).as_posix()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_managed_output_paths(
    *,
    output_dir: Path,
    managed_files: Sequence[Path],
    managed_roots: Sequence[Path],
    protected_inputs: Sequence[Path],
) -> None:
    resolved_files = [path.resolve() for path in managed_files]
    resolved_roots = [path.resolve() for path in managed_roots]
    for path in resolved_files + resolved_roots:
        if not _is_within(path, output_dir):
            raise EvaluationError(
                "managed output escapes --output-dir through a symlink: {}".format(path)
            )
    for protected in protected_inputs:
        resolved = protected.expanduser().resolve()
        if resolved in resolved_files or any(
            _is_within(resolved, root) for root in resolved_roots
        ):
            raise EvaluationError(
                "managed evaluation output must not overwrite input: {}".format(
                    resolved
                )
            )


def _run_decoder(
    command: Sequence[str],
    log_path: Path,
    on_progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(log_path.parent), prefix=".runner-", suffix=".tmp.log"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", errors="replace") as log:
            environment = dict(os.environ)
            environment.pop(PROGRESS_EVENT_ENV, None)
            if on_progress is None:
                completed = subprocess.run(
                    list(command),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    check=False,
                    env=environment,
                )
                return_code = int(completed.returncode)
            else:
                environment[PROGRESS_EVENT_ENV] = "1"
                process = subprocess.Popen(
                    list(command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                )
                assert process.stdout is not None
                progress_callback = on_progress
                try:
                    with process.stdout:
                        for line in process.stdout:
                            event = parse_progress_event(line)
                            if event is None:
                                log.write(line)
                            elif progress_callback is not None:
                                try:
                                    progress_callback(event)
                                except Exception as error:
                                    log.write(
                                        "progress display disabled after error: "
                                        "{}\n".format(error)
                                    )
                                    progress_callback = None
                    return_code = int(process.wait())
                except BaseException:
                    if process.poll() is None:
                        try:
                            process.terminate()
                        except OSError:
                            pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                        except OSError:
                            pass
                        process.wait()
                    raise
            log.flush()
            os.fsync(log.fileno())
        os.replace(str(temporary), str(log_path))
        return return_code
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_resume(
    *,
    run: Mapping[str, Any],
    run_dir: Path,
    manifest_path: Path,
    manifest_sha256: str,
    decode_script: Path,
    decode_script_sha256: str,
    audio_fingerprint: Mapping[str, Any],
    decoder_input_fingerprint: Mapping[str, Any],
    thresholds: Sequence[str],
    decoder_args: Sequence[str],
    category_field: str,
    duration_field: str,
    assume_negative: bool,
) -> Path:
    if not decoder_input_fingerprint.get("resume_safe"):
        raise EvaluationError(
            "--resume requires explicit, existing --checkpoint and --bpe-model "
            "paths resolvable from the current working directory so all runtime "
            "model assets can be fingerprinted"
        )
    expected = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "decode_script": str(decode_script),
        "decode_script_sha256": decode_script_sha256,
        "audio_fingerprint": dict(audio_fingerprint),
        "decoder_input_fingerprint": dict(decoder_input_fingerprint),
        "thresholds": list(thresholds),
        "decoder_args": list(decoder_args),
        "category_field": category_field,
        "duration_field": duration_field,
        "assume_negative": bool(assume_negative),
    }
    mismatches = []
    for name, value in expected.items():
        if run.get(name) != value:
            mismatches.append(name)
    if mismatches:
        raise EvaluationError(
            "--resume configuration differs in: {}".format(
                ", ".join(sorted(mismatches))
            )
        )
    if run.get("decoder_return_code") != 0 or run.get("status") not in {
        "inference_complete",
        "complete",
    }:
        raise EvaluationError(
            "--resume requires a previously successful inference, got status={!r}".format(
                run.get("status")
            )
        )
    exposure_path = _run_relative_path(
        run_dir, run.get("exposure_csv"), "exposure_csv"
    )
    expected_exposure_sha256 = run.get("exposure_csv_sha256")
    if not isinstance(expected_exposure_sha256, str) or not expected_exposure_sha256:
        raise EvaluationError("--resume run.json has no usable exposure_csv_sha256")
    if _sha256(exposure_path) != expected_exposure_sha256:
        raise EvaluationError(
            "--resume exposure CSV content differs from the completed run"
        )
    event_manifest = _run_relative_path(
        run_dir, run.get("event_manifest"), "event_manifest"
    )
    if not event_manifest.is_file():
        raise EvaluationError(
            "--resume event manifest is missing: {}".format(event_manifest)
        )
    expected_event_sha256 = run.get("event_manifest_sha256")
    if not isinstance(expected_event_sha256, str) or not expected_event_sha256:
        raise EvaluationError("--resume run.json has no usable event_manifest_sha256")
    if _sha256(event_manifest) != expected_event_sha256:
        raise EvaluationError(
            "--resume event manifest content differs from the completed run"
        )
    _exact_results_path_from_run(run_dir, run)
    return event_manifest


def run_sweep(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    decode_script = args.decode_script.expanduser().resolve()
    if not decode_script.is_file():
        raise FileNotFoundError("decode script not found: {}".format(decode_script))
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.overwrite
        and not args.resume
    ):
        raise FileExistsError(
            "output directory is not empty (pass --resume or --overwrite): {}".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.thresholds)
    decoder_args = _validate_decoder_args(args.decoder_args)
    manifest_sha256 = _sha256(manifest_path)
    decode_script_sha256 = _sha256(decode_script)
    exposure = _build_exposure(
        manifest_path,
        category_field=args.category_field,
        duration_field=args.duration_field,
        assume_negative=args.assume_negative,
    )
    audio_fingerprint = _audio_fingerprint(exposure)
    decoder_input_fingerprint = _decoder_input_fingerprint(decoder_args)
    exposure_path = output_dir / "exposure.csv"
    inference_dir = output_dir / "inference"
    event_manifest = inference_dir / "manifest.csv"
    inference_results_path = inference_dir / "results.jsonl"
    log_path = output_dir / "runner.log"
    run_path = output_dir / "run.json"
    report_dir = output_dir / "report"
    evaluation_artifact_paths = tuple(
        output_dir / name for name in ARTIFACT_FILENAMES
    )
    inference_artifact_paths = tuple(
        inference_dir / name for name in ARTIFACT_FILENAMES
    )
    protected_inputs = [
        manifest_path,
        decode_script,
        *(Path(str(row["audio_path_resolved"])) for row in exposure),
        *(Path(value) for value in decoder_input_fingerprint.get("paths", [])),
    ]
    _validate_managed_output_paths(
        output_dir=output_dir,
        managed_files=(
            exposure_path,
            log_path,
            run_path,
            *evaluation_artifact_paths,
        ),
        managed_roots=(inference_dir, report_dir),
        protected_inputs=protected_inputs,
    )
    if event_manifest == manifest_path:
        raise EvaluationError(
            "inference output manifest must not overwrite the input manifest: {}".format(
                manifest_path
            )
        )

    if args.resume:
        run_record = _load_run(output_dir)
        previous_decoder_inputs = run_record.get("decoder_input_fingerprint", {})
        if not isinstance(previous_decoder_inputs, dict) or not isinstance(
            previous_decoder_inputs.get("references"), list
        ):
            raise EvaluationError(
                "--resume run.json has no usable decoder_input_fingerprint"
            )
        decoder_input_fingerprint = _decoder_input_fingerprint(
            decoder_args, known_paths=previous_decoder_inputs["references"]
        )
        _validate_resume(
            run=run_record,
            run_dir=output_dir,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            decode_script=decode_script,
            decode_script_sha256=decode_script_sha256,
            audio_fingerprint=audio_fingerprint,
            decoder_input_fingerprint=decoder_input_fingerprint,
            thresholds=thresholds,
            decoder_args=decoder_args,
            category_field=args.category_field,
            duration_field=args.duration_field,
            assume_negative=args.assume_negative,
        )
        artifacts = build_report(output_dir, overwrite=True)
        run_record.update(
            {
                "status": "complete",
                "resumed_at": _utc_now(),
                "finished_at": _utc_now(),
                "artifacts": artifacts,
            }
        )
        _atomic_write_json(run_path, run_record, overwrite=True)
        print(
            "Reused existing inference for {} trials: {}".format(
                len(exposure), output_dir / "report" / "report.html"
            )
        )
        return 0

    if args.overwrite:
        for previous in (
            event_manifest,
            report_dir / "report.html",
            *evaluation_artifact_paths,
            *inference_artifact_paths,
        ):
            if previous.exists() or previous.is_symlink():
                previous.unlink()
    _atomic_write_csv(
        exposure_path, EXPOSURE_FIELDS, exposure, overwrite=args.overwrite
    )
    exposure_sha256 = _sha256(exposure_path)

    command = [
        str(args.python),
        str(decode_script),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(inference_dir),
        "--output-manifest",
        str(event_manifest),
        "--keywords-thresholds",
        ",".join(thresholds),
        "--fail-fast",
    ]
    if args.overwrite:
        command.append("--overwrite")
    command.extend(decoder_args)
    run_record: Dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "decode_script": str(decode_script),
        "decode_script_sha256": decode_script_sha256,
        "audio_fingerprint": audio_fingerprint,
        "decoder_input_fingerprint": decoder_input_fingerprint,
        "decoder_args": decoder_args,
        "thresholds": thresholds,
        "category_field": args.category_field,
        "duration_field": args.duration_field,
        "assume_negative": bool(args.assume_negative),
        "command": command,
        "exposure_csv": _relative_to_run(exposure_path, output_dir),
        "exposure_csv_sha256": exposure_sha256,
        "event_manifest": _relative_to_run(event_manifest, output_dir),
        "runner_log": _relative_to_run(log_path, output_dir),
        "report_dir": "report",
    }
    _atomic_write_json(run_path, run_record, overwrite=True)
    display = ConsoleProgress(
        total=len(exposure),
        description="KWS negative evaluation",
        enabled=getattr(args, "progress", None),
        unit="trials",
    )

    def update_progress(event: Mapping[str, Any]) -> None:
        if int(event.get("total", -1)) != len(exposure):
            return
        display.update(
            completed=int(event["completed"]),
            status="clips={} misses={} errors={}".format(
                event.get("clips", 0),
                event.get("misses", 0),
                event.get("errors", 0),
            ),
        )

    display.start()
    try:
        return_code = _run_decoder(
            command,
            log_path,
            on_progress=update_progress if display.enabled else None,
        )
        if return_code == 0 and event_manifest.is_file():
            display.complete(
                status="{} thresholds; inference complete".format(len(thresholds))
            )
    except OSError as error:
        run_record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "decoder_error": str(error),
            }
        )
        _atomic_write_json(run_path, run_record, overwrite=True)
        raise EvaluationError(
            "could not start streaming inference: {}".format(error)
        ) from error
    except BaseException as error:
        run_record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "decoder_error": "{}: {}".format(
                    type(error).__name__, error or "interrupted"
                ),
            }
        )
        try:
            _atomic_write_json(run_path, run_record, overwrite=True)
        except OSError:
            pass
        raise
    finally:
        display.stop()
    if return_code != 0:
        run_record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "decoder_return_code": return_code,
            }
        )
        _atomic_write_json(run_path, run_record, overwrite=True)
        raise EvaluationError(
            "streaming inference failed with exit code {}; see {}".format(
                return_code, log_path
            )
        )
    if not event_manifest.is_file():
        run_record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "decoder_error": "event manifest was not created",
            }
        )
        _atomic_write_json(run_path, run_record, overwrite=True)
        raise EvaluationError(
            "streaming inference succeeded but did not create {}".format(event_manifest)
        )

    if inference_results_path.is_symlink():
        run_record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "decoder_error": "inference results JSONL is a symlink",
            }
        )
        _atomic_write_json(run_path, run_record, overwrite=True)
        raise EvaluationError(
            "inference results JSONL must not be a symlink: {}".format(
                inference_results_path
            )
        )
    exact_results_available = inference_results_path.is_file()
    if not exact_results_available and decode_script == DEFAULT_DECODE_SCRIPT.resolve():
        run_record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "decoder_error": "default decoder did not create exact results JSONL",
            }
        )
        _atomic_write_json(run_path, run_record, overwrite=True)
        raise EvaluationError(
            "streaming inference succeeded but did not create {}".format(
                inference_results_path
            )
        )
    if exact_results_available:
        exposure_by_row = {
            int(row["source_manifest_row"]): row for row in exposure
        }
        try:
            _load_exact_results(
                inference_results_path,
                thresholds=thresholds,
                exposure_by_row=exposure_by_row,
                allow_missing_label=bool(args.assume_negative),
            )
        except EvaluationError as error:
            run_record.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "decoder_error": str(error),
                }
            )
            _atomic_write_json(run_path, run_record, overwrite=True)
            raise
        run_record.update(
            {
                "inference_results_jsonl": _relative_to_run(
                    inference_results_path, output_dir
                ),
                "inference_results_jsonl_sha256": _sha256(
                    inference_results_path
                ),
                "observation_semantics": "exact_results_jsonl_v1",
            }
        )
    else:
        run_record["observation_semantics"] = "legacy_event_manifest_v1"

    run_record.update(
        {
            "status": "inference_complete",
            "decoder_return_code": 0,
            "event_manifest_sha256": _sha256(event_manifest),
            "inference_finished_at": _utc_now(),
        }
    )
    _atomic_write_json(run_path, run_record, overwrite=True)
    artifacts = build_report(output_dir, overwrite=True)
    run_record.update(
        {
            "status": "complete",
            "finished_at": _utc_now(),
            "artifacts": artifacts,
        }
    )
    _atomic_write_json(run_path, run_record, overwrite=True)
    print(
        "Evaluated {} trials at {} thresholds: {}".format(
            len(exposure), len(thresholds), output_dir / "report" / "report.html"
        )
    )
    return 0


def _load_run(run_dir: Path) -> Dict[str, Any]:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError("run metadata not found: {}".format(run_path))
    try:
        value = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError("cannot read {}: {}".format(run_path, error))
    if not isinstance(value, dict) or value.get("schema_version") != RUN_SCHEMA_VERSION:
        raise EvaluationError("unsupported run.json schema: {}".format(run_path))
    return value


def _run_relative_path(run_dir: Path, value: Any, description: str) -> Path:
    if not value or not str(value).strip():
        raise EvaluationError("run.json is missing {}".format(description))
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _exact_results_path_from_run(
    run_dir: Path, run: Mapping[str, Any]
) -> Optional[Path]:
    value = run.get("inference_results_jsonl")
    expected_sha256 = run.get("inference_results_jsonl_sha256")
    semantics = run.get("observation_semantics")
    if semantics not in (
        None,
        "exact_results_jsonl_v1",
        "legacy_event_manifest_v1",
    ):
        raise EvaluationError(
            "run.json has unsupported observation_semantics={!r}".format(semantics)
        )
    if value is None and expected_sha256 is None:
        if semantics == "exact_results_jsonl_v1":
            raise EvaluationError(
                "run.json declares exact observations but has no results fingerprint"
            )
        return None
    if semantics == "legacy_event_manifest_v1":
        raise EvaluationError(
            "run.json declares legacy observations but also contains exact results"
        )
    if not value or not isinstance(expected_sha256, str) or not expected_sha256:
        raise EvaluationError(
            "run.json has an incomplete exact results path/fingerprint pair"
        )
    raw_path = Path(str(value)).expanduser()
    if not raw_path.is_absolute():
        raw_path = run_dir / raw_path
    if raw_path.is_symlink():
        raise EvaluationError(
            "exact results JSONL must not be a symlink: {}".format(raw_path)
        )
    path = raw_path.resolve()
    expected_path = (run_dir / "inference" / "results.jsonl").resolve()
    if path != expected_path:
        raise EvaluationError(
            "exact results JSONL must be {}: {}".format(expected_path, path)
        )
    if not path.is_file():
        raise EvaluationError("exact results JSONL is missing: {}".format(path))
    if _sha256(path) != expected_sha256:
        raise EvaluationError(
            "exact results JSONL content differs from the completed run: {}".format(
                path
            )
        )
    return path


def _load_exposure(path: Path) -> List[Dict[str, Any]]:
    fields, rows = _read_csv(path)
    missing = set(EXPOSURE_FIELDS).difference(fields)
    if missing:
        raise EvaluationError(
            "exposure CSV is missing column(s): {}".format(", ".join(sorted(missing)))
        )
    result = []
    seen_rows = set()
    for csv_row, row in enumerate(rows, start=2):
        try:
            source_row = int(row["source_manifest_row"])
            duration = float(row["duration_sec"])
        except ValueError as error:
            raise EvaluationError(
                "invalid exposure value at CSV row {}".format(csv_row)
            ) from error
        if source_row in seen_rows:
            raise EvaluationError(
                "duplicate source_manifest_row in exposure: {}".format(source_row)
            )
        if not math.isfinite(duration) or duration <= 0:
            raise EvaluationError(
                "exposure row {} has invalid duration".format(source_row)
            )
        keyword = row["keyword"].strip()
        category = row["category"].strip() or UNCATEGORIZED
        if row["label"].strip() != "0":
            raise EvaluationError(
                "exposure row {} is not negative (label={!r})".format(
                    source_row, row["label"]
                )
            )
        if not keyword:
            raise EvaluationError(
                "exposure row {} has empty keyword".format(source_row)
            )
        if category == ALL_CATEGORY:
            raise EvaluationError(
                "exposure row {} category {!r} is reserved for aggregates".format(
                    source_row, ALL_CATEGORY
                )
            )
        seen_rows.add(source_row)
        result.append(
            {
                **row,
                "source_manifest_row": source_row,
                "duration_sec": duration,
                "keyword": keyword,
                "category": category,
            }
        )
    if not result:
        raise EvaluationError("exposure CSV contains no trials: {}".format(path))
    return result


def _load_hits(
    path: Path,
    *,
    thresholds: Sequence[str],
    exposure_by_row: Mapping[int, Mapping[str, Any]],
) -> Dict[Tuple[str, int], List[float]]:
    fields, rows = _read_csv(path)
    required = {"source_manifest_row", "keywords_threshold", "score"}
    missing = required.difference(fields)
    if missing:
        raise EvaluationError(
            "event manifest is missing column(s): {}".format(", ".join(sorted(missing)))
        )
    requested = set(thresholds)
    hits: DefaultDict[Tuple[str, int], List[float]] = defaultdict(list)
    for csv_row, row in enumerate(rows, start=2):
        try:
            source_row = int(row["source_manifest_row"])
            threshold = _format_threshold(
                _parse_decimal(row["keywords_threshold"], "event threshold")
            )
            score = float(row["score"])
        except (ValueError, EvaluationError) as error:
            raise EvaluationError(
                "invalid event value at CSV row {}: {}".format(csv_row, error)
            ) from error
        if source_row not in exposure_by_row:
            raise EvaluationError(
                "event CSV row {} references unknown source_manifest_row {}".format(
                    csv_row, source_row
                )
            )
        event_keyword = row.get("keyword", "").strip()
        if event_keyword and event_keyword != exposure_by_row[source_row]["keyword"]:
            raise EvaluationError(
                "event CSV row {} keyword {!r} does not match source row {} keyword {!r}".format(
                    csv_row,
                    event_keyword,
                    source_row,
                    exposure_by_row[source_row]["keyword"],
                )
            )
        if threshold not in requested:
            raise EvaluationError(
                "event CSV row {} has unrequested threshold {}".format(
                    csv_row, threshold
                )
            )
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise EvaluationError(
                "event CSV row {} score must be finite and between 0 and 1".format(
                    csv_row
                )
            )
        hits[(threshold, source_row)].append(score)
    return dict(hits)


def _strict_utf8_lines(path: Path) -> Iterable[Tuple[int, str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                yield line_number, line
    except UnicodeDecodeError as error:
        raise EvaluationError(
            "exact results JSONL is not valid UTF-8: {}".format(path)
        ) from error


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-standard JSON constant {!r}".format(value))


def _load_exact_results(
    path: Path,
    *,
    thresholds: Sequence[str],
    exposure_by_row: Mapping[int, Mapping[str, Any]],
    allow_missing_label: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], List[float]]]:
    """Validate and canonicalize streaming_kws exact-threshold observations."""
    if not path.is_file():
        raise EvaluationError("exact results JSONL is missing: {}".format(path))
    requested = set(thresholds)
    expected = {
        (threshold, source_row)
        for threshold in thresholds
        for source_row in exposure_by_row
    }
    seen = set()
    records: List[Dict[str, Any]] = []
    hits: Dict[Tuple[str, int], List[float]] = {}
    for line_number, line in _strict_utf8_lines(path):
        try:
            if not line.strip():
                continue
            try:
                raw = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise EvaluationError(
                    "invalid exact results JSON at line {}: {}".format(
                        line_number, error
                    )
                ) from error
            if not isinstance(raw, dict):
                raise EvaluationError(
                    "exact results line {} is not a JSON object".format(
                        line_number
                    )
                )
            raw_source_row = raw.get("source_manifest_row")
            if isinstance(raw_source_row, bool) or not isinstance(
                raw_source_row, int
            ):
                raise EvaluationError(
                    "exact results line {} has invalid source_manifest_row".format(
                        line_number
                    )
                )
            try:
                source_row = raw_source_row
                threshold = _format_threshold(
                    _parse_decimal(raw["threshold"], "result threshold")
                )
            except (KeyError, TypeError, ValueError, EvaluationError) as error:
                raise EvaluationError(
                    "invalid exact result identity at line {}: {}".format(
                        line_number, error
                    )
                ) from error
            identity = (threshold, source_row)
            if threshold not in requested:
                raise EvaluationError(
                    "exact results line {} has unrequested threshold {}".format(
                        line_number, threshold
                    )
                )
            if source_row not in exposure_by_row:
                raise EvaluationError(
                    "exact results line {} references unknown source row {}".format(
                        line_number, source_row
                    )
                )
            if identity in seen:
                raise EvaluationError(
                    "duplicate exact result for threshold {} and source row {}".format(
                        threshold, source_row
                    )
                )
            seen.add(identity)
            exposure = exposure_by_row[source_row]
            if str(raw.get("keyword", "")).strip() != str(exposure["keyword"]):
                raise EvaluationError(
                    "exact results line {} keyword does not match source row {}".format(
                        line_number, source_row
                    )
                )
            if str(raw.get("audio_path", "")) != str(exposure["audio_path"]):
                raise EvaluationError(
                    "exact results line {} audio_path does not match source "
                    "row {}".format(line_number, source_row)
                )
            resolved_audio = raw.get("audio_path_resolved")
            expected_audio = Path(
                str(exposure["audio_path_resolved"])
            ).expanduser().resolve()
            if (
                resolved_audio is None
                or Path(str(resolved_audio)).expanduser().resolve()
                != expected_audio
            ):
                raise EvaluationError(
                    "exact results line {} resolved audio path does not match "
                    "source row {}".format(line_number, source_row)
                )
            label = raw.get("label")
            label_was_missing = label is None
            if label_was_missing and allow_missing_label:
                label = 0
            elif isinstance(label, bool) or not isinstance(label, int):
                raise EvaluationError(
                    "exact results line {} has no usable negative label".format(
                        line_number
                    )
                )
            if label != 0:
                raise EvaluationError(
                    "exact results line {} is not negative".format(line_number)
                )
            skipped = raw.get("skipped")
            if not isinstance(skipped, bool) or skipped or raw.get("error") not in (
                None,
                "",
            ):
                raise EvaluationError(
                    "exact results line {} is skipped or failed".format(line_number)
                )
            detections = raw.get("detections")
            if not isinstance(detections, list) or any(
                not isinstance(event, dict) for event in detections
            ):
                raise EvaluationError(
                    "exact results line {} has invalid detections".format(line_number)
                )
            canonical_detections = []
            for event_index, event in enumerate(detections):
                canonical_event = dict(event)
                clip_value = canonical_event.get("clip_audio_path")
                if clip_value not in (None, ""):
                    clip_path = Path(str(clip_value))
                    if not clip_path.is_absolute():
                        clip_path = path.parent / clip_path
                    resolved_clip = clip_path.resolve()
                    try:
                        resolved_clip.relative_to(path.parent.resolve())
                    except ValueError as error:
                        raise EvaluationError(
                            "exact results line {} event {} clip path escapes "
                            "the inference directory".format(
                                line_number, event_index
                            )
                        ) from error
                    recorded_resolved = canonical_event.get(
                        "clip_audio_path_resolved"
                    )
                    if recorded_resolved not in (None, "") and Path(
                        str(recorded_resolved)
                    ).expanduser().resolve() != resolved_clip:
                        raise EvaluationError(
                            "exact results line {} event {} clip paths disagree".format(
                                line_number, event_index
                            )
                        )
                    canonical_event["clip_audio_path"] = Path(
                        os.path.relpath(str(resolved_clip), str(path.parent.parent))
                    ).as_posix()
                    canonical_event["clip_audio_path_resolved"] = str(
                        resolved_clip
                    )
                canonical_detections.append(canonical_event)
            detections = canonical_detections
            detection_count = raw.get("detection_count")
            if isinstance(detection_count, bool) or not isinstance(
                detection_count, int
            ):
                raise EvaluationError(
                    "exact results line {} has invalid detection_count".format(
                        line_number
                    )
                )
            if detection_count != len(detections):
                raise EvaluationError(
                    "exact results line {} detection_count disagrees with "
                    "events".format(line_number)
                )
            false_alarm_events = raw.get("false_alarm_events")
            if (
                false_alarm_events is None
                and label_was_missing
                and allow_missing_label
            ):
                false_alarm_events = detection_count
            if isinstance(false_alarm_events, bool) or not isinstance(
                false_alarm_events, int
            ) or false_alarm_events != detection_count:
                raise EvaluationError(
                    "exact results line {} false_alarm_events disagrees with "
                    "events".format(line_number)
                )
            detected = raw.get("detected")
            if not isinstance(detected, bool) or detected != bool(detections):
                raise EvaluationError(
                    "exact results line {} detected disagrees with events".format(
                        line_number
                    )
                )
            scores = []
            for event_index, event in enumerate(detections):
                raw_score = event.get("score")
                if isinstance(raw_score, bool) or not isinstance(
                    raw_score, (int, float)
                ):
                    raise EvaluationError(
                        "exact results line {} event {} has invalid score".format(
                            line_number, event_index
                        )
                    )
                score = float(raw_score)
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise EvaluationError(
                        "exact results line {} event {} score is out of range".format(
                            line_number, event_index
                        )
                    )
                scores.append(score)
            raw_qbyt_score = raw.get("qbyt_score")
            if isinstance(raw_qbyt_score, bool) or not isinstance(
                raw_qbyt_score, (int, float)
            ):
                raise EvaluationError(
                    "exact results line {} has invalid qbyt_score".format(line_number)
                )
            qbyt_score = float(raw_qbyt_score)
            expected_score = max(scores) if scores else 0.0
            if not math.isfinite(qbyt_score) or not math.isclose(
                qbyt_score, expected_score, rel_tol=0.0, abs_tol=1e-12
            ):
                raise EvaluationError(
                    "exact results line {} qbyt_score disagrees with events".format(
                        line_number
                    )
                )
            raw_manifest_meta = raw.get("manifest_meta", {})
            if not isinstance(raw_manifest_meta, dict):
                raise EvaluationError(
                    "exact results line {} has invalid manifest_meta".format(
                        line_number
                    )
                )
            manifest_meta = dict(raw_manifest_meta)
            manifest_meta["category"] = str(exposure["category"])
            records.append(
                build_evaluation_result(
                    source_manifest_row=source_row,
                    audio_path=str(exposure["audio_path"]),
                    audio_path_resolved=str(exposure["audio_path_resolved"]),
                    keyword=str(exposure["keyword"]),
                    label=0,
                    threshold=threshold,
                    events=detections,
                    duration_sec=float(exposure["duration_sec"]),
                    manifest_meta=manifest_meta,
                )
            )
            hits[identity] = scores
        except ArtifactError as error:
            raise EvaluationError(
                "invalid exact result at line {}: {}".format(line_number, error)
            ) from error
    if seen != expected:
        missing = sorted(
            expected.difference(seen),
            key=lambda item: (item[1], float(item[0])),
        )
        preview = ", ".join(
            "threshold {} / row {}".format(threshold, source_row)
            for threshold, source_row in missing[:5]
        )
        raise EvaluationError(
            "exact results are missing {} threshold/trial observation(s): {}".format(
                len(missing), preview
            )
        )
    return records, hits


def _source_metrics(
    thresholds: Sequence[str],
    exposure: Sequence[Mapping[str, Any]],
    hits: Mapping[Tuple[str, int], Sequence[float]],
) -> List[Dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        for item in exposure:
            source_row = int(item["source_manifest_row"])
            scores = list(hits.get((threshold, source_row), ()))
            rows.append(
                {
                    "threshold": threshold,
                    "source_manifest_row": source_row,
                    "audio_path": item["audio_path"],
                    "audio_path_resolved": item["audio_path_resolved"],
                    "keyword": item["keyword"],
                    "category": item["category"],
                    "duration_sec": _format_number(float(item["duration_sec"]), 12),
                    "false_alarm_events": len(scores),
                    "triggered": int(bool(scores)),
                    "max_score": "" if not scores else _format_number(max(scores), 12),
                }
            )
    return rows


def _standard_evaluation_records(
    source_rows: Sequence[Mapping[str, Any]],
    hits: Mapping[Tuple[str, int], Sequence[float]],
) -> List[Dict[str, Any]]:
    """Adapt exact threshold/trial observations to the shared JSONL schema."""
    records = []
    for row in source_rows:
        threshold = str(row["threshold"])
        source_row = int(row["source_manifest_row"])
        scores = list(hits.get((threshold, source_row), ()))
        records.append(
            build_evaluation_result(
                source_manifest_row=source_row,
                audio_path=str(row["audio_path"]),
                audio_path_resolved=str(row["audio_path_resolved"]),
                keyword=str(row["keyword"]),
                label=0,
                threshold=threshold,
                events=[
                    {
                        "score": float(score),
                        "event_index": index,
                        "clip_exported": True,
                    }
                    for index, score in enumerate(scores)
                ],
                duration_sec=float(row["duration_sec"]),
                manifest_meta={"category": str(row["category"])},
            )
        )
    return records


def _threshold_metrics(
    thresholds: Sequence[str], source_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_threshold: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_threshold[str(row["threshold"])].append(row)

    result = []
    for threshold in thresholds:
        threshold_rows = by_threshold[threshold]
        keywords = sorted({str(row["keyword"]) for row in threshold_rows})
        for keyword in keywords:
            keyword_rows = [row for row in threshold_rows if row["keyword"] == keyword]
            categories = sorted({str(row["category"]) for row in keyword_rows})
            for category in categories + [ALL_CATEGORY]:
                group = (
                    keyword_rows
                    if category == ALL_CATEGORY
                    else [row for row in keyword_rows if row["category"] == category]
                )
                exposure_sec = sum(float(row["duration_sec"]) for row in group)
                exposure_hours = exposure_sec / 3600.0
                event_count = sum(int(row["false_alarm_events"]) for row in group)
                triggered = [row for row in group if int(row["triggered"])]
                unique_files = {str(row["audio_path_resolved"]) for row in group}
                triggered_files = {str(row["audio_path_resolved"]) for row in triggered}
                trial_count = len(group)
                result.append(
                    {
                        "threshold": threshold,
                        "keyword": keyword,
                        "category": category,
                        "source_trials": trial_count,
                        "unique_audio_files": len(unique_files),
                        "exposure_sec": _format_number(exposure_sec, 12),
                        "exposure_hours": _format_number(exposure_hours, 12),
                        "false_alarm_events": event_count,
                        "triggered_source_trials": len(triggered),
                        "triggered_audio_files": len(triggered_files),
                        "fa_per_hour": _format_number(
                            event_count / exposure_hours if exposure_hours else 0.0, 12
                        ),
                        "source_trial_trigger_rate": _format_number(
                            len(triggered) / trial_count if trial_count else 0.0, 12
                        ),
                    }
                )
    return result


def _nonmonotonic_warnings(
    summary_rows: Sequence[Mapping[str, Any]], metric: str
) -> List[str]:
    grouped: DefaultDict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["keyword"]), str(row["category"]))].append(row)
    warnings = []
    for (keyword, category), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: Decimal(str(row["threshold"])))
        for previous, current in zip(ordered, ordered[1:]):
            if float(current[metric]) > float(previous[metric]) + 1e-12:
                warnings.append(
                    "{} increased for keyword={!r}, category={!r}: {} at {} -> {} at {}".format(
                        metric,
                        keyword,
                        category,
                        previous[metric],
                        previous["threshold"],
                        current[metric],
                        current["threshold"],
                    )
                )
    return warnings


def _slug(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not stem:
        stem = "keyword"
    return "{}-{}".format(
        stem[:48], hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    )


def _svg_line_chart(
    *,
    title: str,
    y_label: str,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> str:
    categories = sorted(
        {str(row["category"]) for row in rows},
        key=lambda value: (value != ALL_CATEGORY, value),
    )
    legend_columns = 4
    legend_rows = max(1, (len(categories) + legend_columns - 1) // legend_columns)
    width = 960
    height = 520 + max(0, legend_rows - 2) * 22
    left, right, top, bottom = 92, 34, 70 + legend_rows * 22, 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    thresholds = sorted({float(row["threshold"]) for row in rows})
    values = [float(row[metric]) for row in rows]
    y_max = max(values) if values else 0.0
    if y_max <= 0:
        y_max = 1.0
    else:
        y_max *= 1.08
    x_min, x_max = min(thresholds), max(thresholds)
    if x_min == x_max:
        x_min = max(0.0, x_min - 0.05)
        x_max = min(1.0, x_max + 0.05)
        if x_min == x_max:
            x_max = x_min + 1.0

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return top + plot_height - value / y_max * plot_height

    colors = (
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#4b5563",
        "#be185d",
    )
    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" role="img">'.format(
            width, height, width, height
        ),
        "<title>{}</title>".format(html.escape(title)),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="32" font-family="sans-serif" font-size="20" font-weight="600">{}</text>'.format(
            left, html.escape(title)
        ),
    ]
    for index in range(6):
        value = y_max * index / 5.0
        y = y_pos(value)
        pieces.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#e5e7eb"/>'.format(
                left, y, left + plot_width, y
            )
        )
        pieces.append(
            '<text x="{}" y="{:.2f}" text-anchor="end" dominant-baseline="middle" font-family="sans-serif" font-size="12" fill="#4b5563">{}</text>'.format(
                left - 10, y, html.escape(_format_number(value, 5))
            )
        )
    tick_thresholds = [thresholds[0]]
    minimum_tick_gap = plot_width / 11.0
    for threshold in thresholds[1:-1]:
        if (
            x_pos(threshold) - x_pos(tick_thresholds[-1]) >= minimum_tick_gap
            and x_pos(thresholds[-1]) - x_pos(threshold) >= minimum_tick_gap
        ):
            tick_thresholds.append(threshold)
    if thresholds[-1] != tick_thresholds[-1]:
        tick_thresholds.append(thresholds[-1])
    for threshold in tick_thresholds:
        x = x_pos(threshold)
        pieces.append(
            '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" stroke="#f3f4f6"/>'.format(
                x, top, x, top + plot_height
            )
        )
        pieces.append(
            '<text x="{:.2f}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#4b5563">{}</text>'.format(
                x, top + plot_height + 24, html.escape(format(threshold, ".6g"))
            )
        )
    pieces.extend(
        [
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#111827"/>'.format(
                left, top + plot_height, left + plot_width, top + plot_height
            ),
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#111827"/>'.format(
                left, top, left, top + plot_height
            ),
            '<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="14">Threshold</text>'.format(
                left + plot_width / 2, height - 22
            ),
            '<text x="22" y="{}" text-anchor="middle" transform="rotate(-90 22 {})" font-family="sans-serif" font-size="14">{}</text>'.format(
                top + plot_height / 2,
                top + plot_height / 2,
                html.escape(y_label),
            ),
        ]
    )
    dash_patterns = ("", "9 4", "3 3", "12 3 3 3", "6 3 2 3")
    for category_index, category in enumerate(categories):
        color = colors[category_index % len(colors)]
        dash = dash_patterns[category_index % len(dash_patterns)]
        dash_attribute = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        marker_radius = max(2.8, 5.5 - category_index * 0.6)
        series = sorted(
            (row for row in rows if row["category"] == category),
            key=lambda row: float(row["threshold"]),
        )
        points = [
            (x_pos(float(row["threshold"])), y_pos(float(row[metric])))
            for row in series
        ]
        pieces.append(
            '<polyline fill="none" stroke="{}" stroke-width="2.5"{} '
            'stroke-linecap="round" points="{}"/>'.format(
                color,
                dash_attribute,
                " ".join("{:.2f},{:.2f}".format(x, y) for x, y in points),
            )
        )
        for x, y in points:
            pieces.append(
                '<circle cx="{:.2f}" cy="{:.2f}" r="{:.2f}" fill="#fff" '
                'stroke="{}" stroke-width="1.4"/>'.format(x, y, marker_radius, color)
            )
        legend_x = left + 10 + (category_index % legend_columns) * 205
        legend_y = 62 + (category_index // legend_columns) * 22
        pieces.append(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" '
            'stroke-width="3"{} stroke-linecap="round"/>'.format(
                legend_x,
                legend_y,
                legend_x + 20,
                legend_y,
                color,
                dash_attribute,
            )
        )
        pieces.append(
            '<text x="{}" y="{}" dominant-baseline="middle" font-family="sans-serif" font-size="12">{}</text>'.format(
                legend_x + 27, legend_y, html.escape(category)
            )
        )
    pieces.append("</svg>\n")
    return "\n".join(pieces)


def _html_report(
    *,
    run: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    charts: Sequence[Mapping[str, str]],
    warnings: Sequence[str],
) -> str:
    table_head = "".join(
        "<th>{}</th>".format(html.escape(name)) for name in THRESHOLD_METRIC_FIELDS
    )
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            "<tr>{}</tr>".format(
                "".join(
                    "<td>{}</td>".format(html.escape(str(row.get(name, ""))))
                    for name in THRESHOLD_METRIC_FIELDS
                )
            )
        )
    warning_html = ""
    if warnings:
        warning_html = "<section><h2>Non-monotonic observations</h2><p>These are retained as measured; no curve correction was applied.</p><ul>{}</ul></section>".format(
            "".join("<li>{}</li>".format(html.escape(item)) for item in warnings)
        )
    chart_html = "".join(
        '<section><h2>{}</h2><img src="{}" alt="{}" loading="lazy"></section>'.format(
            html.escape(chart["title"]),
            html.escape(chart["path"]),
            html.escape(chart["title"]),
        )
        for chart in charts
    )
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KWS negative-set evaluation</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;color:#111827;background:#f8fafc}}main{{max-width:1200px;margin:auto;padding:28px}}section{{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:18px;margin:18px 0;overflow:auto}}h1,h2{{margin-top:0}}code{{background:#f3f4f6;padding:2px 5px;border-radius:4px}}img{{width:100%;min-width:700px;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}th{{position:sticky;top:0;background:#f9fafb}}small{{color:#4b5563}}
</style>
</head>
<body><main>
<h1>KWS negative-set evaluation</h1>
<p><small>Generated {generated}. One manifest row is one audio-keyword trial. FA/h = false-alarm events / summed trial exposure hours.</small></p>
<section><h2>Run</h2><p>Manifest: <code>{manifest}</code></p><p>Thresholds: <code>{thresholds}</code></p></section>
{warnings}
{charts}
<section><h2>Threshold metrics</h2><table><thead><tr>{table_head}</tr></thead><tbody>{table_rows}</tbody></table></section>
</main></body></html>
""".format(
        generated=html.escape(_utc_now()),
        manifest=html.escape(str(run.get("manifest", ""))),
        thresholds=html.escape(
            ", ".join(str(item) for item in run.get("thresholds", []))
        ),
        warnings=warning_html,
        charts=chart_html,
        table_head=table_head,
        table_rows="".join(table_rows),
    )


def build_report(run_dir: Path, overwrite: bool) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    run = _load_run(run_dir)
    if run.get("status") not in {"inference_complete", "complete"}:
        raise EvaluationError(
            "report requires completed inference, got run status={!r}".format(
                run.get("status")
            )
        )
    thresholds = parse_thresholds([str(value) for value in run.get("thresholds", [])])
    exposure_path = _run_relative_path(run_dir, run.get("exposure_csv"), "exposure_csv")
    event_path = _run_relative_path(
        run_dir, run.get("event_manifest"), "event_manifest"
    )
    expected_exposure_sha256 = run.get("exposure_csv_sha256")
    if not isinstance(expected_exposure_sha256, str) or not expected_exposure_sha256:
        raise EvaluationError("run.json has no usable exposure_csv_sha256")
    if _sha256(exposure_path) != expected_exposure_sha256:
        raise EvaluationError(
            "exposure CSV content differs from the completed run: {}".format(
                exposure_path
            )
        )
    expected_event_sha256 = run.get("event_manifest_sha256")
    if not isinstance(expected_event_sha256, str) or not expected_event_sha256:
        raise EvaluationError("run.json has no usable event_manifest_sha256")
    if _sha256(event_path) != expected_event_sha256:
        raise EvaluationError(
            "event manifest content differs from the completed run: {}".format(
                event_path
            )
        )
    exposure = _load_exposure(exposure_path)
    exposure_by_row = {int(row["source_manifest_row"]): row for row in exposure}
    exported_hits = _load_hits(
        event_path, thresholds=thresholds, exposure_by_row=exposure_by_row
    )
    exact_results_path = _exact_results_path_from_run(run_dir, run)
    if exact_results_path is None:
        hits = exported_hits
        standard_records: List[Dict[str, Any]] = []
        observation_semantics = "legacy_event_manifest_v1"
    else:
        standard_records, hits = _load_exact_results(
            exact_results_path,
            thresholds=thresholds,
            exposure_by_row=exposure_by_row,
            allow_missing_label=bool(run.get("assume_negative", False)),
        )
        observation_semantics = "exact_results_jsonl_v1"
    source_rows = _source_metrics(thresholds, exposure, hits)
    if exact_results_path is None:
        standard_records = _standard_evaluation_records(source_rows, hits)
    summary_rows = _threshold_metrics(thresholds, source_rows)
    warnings = _nonmonotonic_warnings(summary_rows, "fa_per_hour")
    warnings.extend(_nonmonotonic_warnings(summary_rows, "source_trial_trigger_rate"))

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    source_path = report_dir / "source_metrics.csv"
    threshold_path = report_dir / "threshold_metrics.csv"
    summary_path = report_dir / "summary.json"
    html_path = report_dir / "report.html"
    _atomic_write_csv(
        source_path, SOURCE_METRIC_FIELDS, source_rows, overwrite=overwrite
    )
    _atomic_write_csv(
        threshold_path, THRESHOLD_METRIC_FIELDS, summary_rows, overwrite=overwrite
    )

    standard_artifacts = write_evaluation_artifacts(
        output_dir=run_dir,
        records=standard_records,
        thresholds=thresholds,
        manifest_path=Path(str(run.get("manifest", ""))),
        overwrite=overwrite,
        mode="musan",
        summary_metadata={
            name: run[name]
            for name in (
                "decode_script",
                "decode_script_sha256",
                "decoder_args",
                "decoder_input_fingerprint",
                "manifest_sha256",
                "inference_results_jsonl",
                "inference_results_jsonl_sha256",
            )
            if run.get(name) is not None
        },
    )

    charts = []
    keywords = sorted({str(row["keyword"]) for row in summary_rows})
    for keyword in keywords:
        keyword_rows = [row for row in summary_rows if row["keyword"] == keyword]
        slug = _slug(keyword)
        for metric, prefix, y_label in (
            ("fa_per_hour", "fa-per-hour", "False alarms / hour"),
            (
                "source_trial_trigger_rate",
                "source-trial-trigger-rate",
                "Triggered trial rate",
            ),
        ):
            filename = "{}-{}.svg".format(prefix, slug)
            title = "{} — {}".format(keyword, y_label)
            _atomic_write_text(
                report_dir / filename,
                _svg_line_chart(
                    title=title,
                    y_label=y_label,
                    rows=keyword_rows,
                    metric=metric,
                ),
                overwrite=overwrite,
            )
            charts.append({"title": title, "path": filename})

    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "semantics": {
            "trial": "one input manifest row (one audio-keyword pair)",
            "false_alarm_events": "all detected events; multiple events per trial count",
            "triggered_source_trials": "trials with one or more events; each trial counts once",
            "fa_per_hour": "false_alarm_events / summed source-trial exposure hours",
            "observation_source": observation_semantics,
        },
        "thresholds": thresholds,
        "source_trials": len(exposure),
        "event_count": sum(len(scores) for scores in hits.values()),
        "exported_clip_event_count": sum(
            len(scores) for scores in exported_hits.values()
        ),
        "warnings": warnings,
        "metrics": summary_rows,
        "charts": charts,
        "standard_artifacts": {
            name: _relative_to_run(Path(path), run_dir)
            for name, path in standard_artifacts.items()
        },
    }
    _atomic_write_json(summary_path, summary, overwrite=overwrite)
    _atomic_write_text(
        html_path,
        _html_report(
            run=run,
            summary_rows=summary_rows,
            charts=charts,
            warnings=warnings,
        ),
        overwrite=overwrite,
    )
    artifacts = {
        "source_metrics_csv": _relative_to_run(source_path, run_dir),
        "threshold_metrics_csv": _relative_to_run(threshold_path, run_dir),
        "summary_json": _relative_to_run(summary_path, run_dir),
        "report_html": _relative_to_run(html_path, run_dir),
        "charts": [
            _relative_to_run(report_dir / chart["path"], run_dir) for chart in charts
        ],
    }
    artifacts.update(
        {
            "results_jsonl": _relative_to_run(
                Path(standard_artifacts["results.jsonl"]), run_dir
            ),
            "standard_summary_json": _relative_to_run(
                Path(standard_artifacts["summary.json"]), run_dir
            ),
            "threshold_scan_summary_json": _relative_to_run(
                Path(standard_artifacts["threshold_scan_summary.json"]), run_dir
            ),
            "threshold_scan_csv": _relative_to_run(
                Path(standard_artifacts["threshold_scan.csv"]), run_dir
            ),
            "threshold_scan_png": _relative_to_run(
                Path(standard_artifacts["threshold_scan.png"]), run_dir
            ),
        }
    )
    return artifacts


def run_report(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    artifacts = build_report(run_dir, overwrite=args.overwrite)
    run_path = run_dir / "run.json"
    run = _load_run(run_dir)
    run["artifacts"] = artifacts
    run["report_rebuilt_at"] = _utc_now()
    if run.get("status") == "inference_complete":
        run["status"] = "complete"
        run["finished_at"] = _utc_now()
    _atomic_write_json(run_path, run, overwrite=True)
    print("Rebuilt report: {}".format(run_dir / artifacts["report_html"]))
    return 0


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Prepare and evaluate negative KWS datasets with exact threshold scanning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        allow_abbrev=False,
        help="Recursively create a label=0 manifest (one row per audio-keyword pair).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare.add_argument("--input-dir", type=Path, required=True)
    prepare.add_argument("--output-manifest", type=Path, required=True)
    prepare.add_argument(
        "--keyword",
        action="append",
        required=True,
        help="Keyword text or direct BPE form; repeat to create multiple trials per audio.",
    )
    prepare.add_argument(
        "--extensions",
        nargs="+",
        default=["wav"],
        help="Comma/space-separated file extensions. Duration probing is WAV-only.",
    )
    prepare.add_argument(
        "--category-depth",
        type=int,
        default=0,
        help="Directory component below input root; negative selects the direct parent.",
    )
    prepare.add_argument("--absolute-paths", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=prepare_manifest)

    feature_prepare = subparsers.add_parser(
        "prepare-features",
        allow_abbrev=False,
        help="Incrementally cache one CPU Fbank matrix per unique audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    feature_prepare.add_argument("--manifest", type=Path, required=True)
    feature_prepare.add_argument(
        "--feature-cache-dir",
        type=Path,
        required=True,
    )
    feature_prepare.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="CPU worker processes used for missing or stale entries.",
    )
    feature_prepare.add_argument(
        "--force",
        action="store_true",
        help="Recompute valid entries instead of reusing them.",
    )
    feature_progress = feature_prepare.add_mutually_exclusive_group()
    feature_progress.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show progress even when stderr is not a TTY.",
    )
    feature_progress.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the feature preparation progress display.",
    )
    feature_prepare.set_defaults(progress=None, handler=prepare_features)

    sweep = subparsers.add_parser(
        "sweep",
        allow_abbrev=False,
        help="Run one shared-encoder inference pass and build metrics/reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sweep.add_argument("--manifest", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)
    sweep.add_argument(
        "--thresholds",
        nargs="+",
        required=True,
        help="Values, comma lists, or inclusive ranges such as 0.2:0.8:0.05.",
    )
    sweep.add_argument("--decode-script", type=Path, default=DEFAULT_DECODE_SCRIPT)
    sweep.add_argument("--python", default=sys.executable)
    sweep.add_argument(
        "--category-field",
        default="category",
        help="Existing category column; pass an empty string to disable grouping.",
    )
    sweep.add_argument("--duration-field", default="source_duration_sec")
    sweep.add_argument(
        "--assume-negative",
        action="store_true",
        help="Allow missing/blank labels only after independently verifying all rows are negative.",
    )
    existing = sweep.add_mutually_exclusive_group()
    existing.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible completed inference and rebuild the report.",
    )
    existing.add_argument("--overwrite", action="store_true")
    progress = sweep.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show inference progress even when stderr is not a TTY.",
    )
    progress.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the inference progress display.",
    )
    sweep.set_defaults(progress=None)
    sweep.add_argument(
        "decoder_args",
        nargs=argparse.REMAINDER,
        help="Arguments for streaming_kws.py; place them after a standalone --.",
    )
    sweep.set_defaults(handler=run_sweep)

    report = subparsers.add_parser(
        "report",
        allow_abbrev=False,
        help="Rebuild CSV/JSON/HTML/SVG artifacts without model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--overwrite", action="store_true")
    report.set_defaults(handler=run_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        EvaluationError,
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
    ) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    except PermissionError as error:
        print("error: permission denied: {}".format(error), file=sys.stderr)
        return 2
    except (OSError, csv.Error) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
