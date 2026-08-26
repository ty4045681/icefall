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
"""Incremental, process-safe CPU Fbank cache for manifest KWS inference."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

SAMPLE_RATE = 16000
FBANK_FRAME_SHIFT_MS = 10.0
FBANK_NUM_BINS = 80
FEATURE_CACHE_SCHEMA_VERSION = 1
FRONTEND_ALGORITHM_VERSION = 1


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def frontend_spec() -> Dict[str, Any]:
    """Return every setting that can change the cached feature values."""
    return {
        "algorithm_version": FRONTEND_ALGORITHM_VERSION,
        "compute_device": "cpu",
        "sample_rate": SAMPLE_RATE,
        "downmix": "channel_mean",
        "resampler": "torchaudio.functional.resample",
        "fbank": {
            "implementation": "kaldifeat.Fbank",
            "num_bins": FBANK_NUM_BINS,
            "frame_shift_ms": FBANK_FRAME_SHIFT_MS,
            "dither": 0,
            "snip_edges": False,
            "high_freq": -400,
        },
        "versions": {
            "kaldifeat": _package_version("kaldifeat"),
            "soundfile": _package_version("soundfile"),
            "torch": _package_version("torch"),
            "torchaudio": _package_version("torchaudio"),
        },
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def frontend_fingerprint(spec: Optional[Mapping[str, Any]] = None) -> str:
    values = frontend_spec() if spec is None else dict(spec)
    return hashlib.sha256(_canonical_json(values).encode("utf-8")).hexdigest()


def source_fingerprint(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "resolved_path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _entry_key(source: Mapping[str, Any], frontend_sha256: str) -> str:
    value = {
        "source": dict(source),
        "frontend_sha256": frontend_sha256,
    }
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}-".format(path.name), suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> Any:
    """Load, downmix, and deterministically resample model-input audio."""
    import soundfile as sf
    import torch

    data, source_sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
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
        waveform = audio_functional.resample(waveform, source_sample_rate, sample_rate)
    return waveform.contiguous()


def compute_fbank(waveform: Any, device: Any) -> Any:
    """Compute the exact 80-bin frontend profile used by streaming KWS."""
    from kaldifeat import Fbank, FbankOptions

    options = FbankOptions()
    options.device = device
    options.frame_opts.dither = 0
    options.frame_opts.snip_edges = False
    options.frame_opts.samp_freq = SAMPLE_RATE
    options.frame_opts.frame_shift_ms = FBANK_FRAME_SHIFT_MS
    options.mel_opts.num_bins = FBANK_NUM_BINS
    options.mel_opts.high_freq = -400
    features = Fbank(options)(waveform.to(device))
    if features.ndim != 2 or features.size(0) == 0:
        raise ValueError("audio produced no usable fbank frames")
    return features


@dataclass(frozen=True)
class CachedFbank:
    features: Any
    num_samples: int
    key: str
    metadata: Mapping[str, Any]


class FbankFeatureCache:
    """A cache whose JSON sidecar is the atomic completion marker."""

    def __init__(
        self,
        root: Path,
        spec: Optional[Mapping[str, Any]] = None,
        writable: bool = True,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.spec = frontend_spec() if spec is None else dict(spec)
        self.frontend_sha256 = frontend_fingerprint(self.spec)
        self.writable = writable
        if writable:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_cache_manifest()

    def _write_cache_manifest(self) -> None:
        value = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "frontend_sha256": self.frontend_sha256,
            "frontend_spec": self.spec,
        }
        path = self.root / "cache.json"
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing != value:
            _atomic_write_json(path, value)

    def _paths(self, key: str) -> Tuple[Path, Path]:
        directory = self.root / "objects" / key[:2]
        return directory / "{}.npy".format(key), directory / "{}.json".format(key)

    def _current_identity(self, audio_path: Path) -> Tuple[Dict[str, Any], str]:
        source = source_fingerprint(audio_path)
        return source, _entry_key(source, self.frontend_sha256)

    def lookup_metadata(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        try:
            source, key = self._current_identity(audio_path)
        except OSError:
            return None
        data_path, metadata_path = self._paths(key)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data_stat = data_path.stat()
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict):
            return None
        expected = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "key": key,
            "source": source,
            "frontend_sha256": self.frontend_sha256,
            "dtype": "float32",
            "sample_rate": SAMPLE_RATE,
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            return None
        shape = metadata.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in shape
            )
            or shape[0] <= 0
            or shape[1] != FBANK_NUM_BINS
        ):
            return None
        num_samples = metadata.get("num_samples")
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples <= 0
        ):
            return None
        data_size = metadata.get("data_size")
        if (
            isinstance(data_size, bool)
            or not isinstance(data_size, int)
            or data_size != int(data_stat.st_size)
        ):
            return None
        expected_relative = data_path.relative_to(self.root).as_posix()
        if metadata.get("data_path") != expected_relative:
            return None
        try:
            import numpy as np

            array = np.load(str(data_path), mmap_mode="r", allow_pickle=False)
            valid = list(array.shape) == shape and str(array.dtype) == "float32"
            del array
        except Exception:
            return None
        return metadata if valid else None

    def load(self, audio_path: Path) -> Optional[CachedFbank]:
        metadata = self.lookup_metadata(audio_path)
        if metadata is None:
            return None
        try:
            import numpy as np
            import torch

            data_path = self.root / str(metadata["data_path"])
            array = np.load(str(data_path), allow_pickle=False)
            array = np.array(array, dtype=np.float32, order="C", copy=True)
            features = torch.from_numpy(array)
        except Exception:
            return None
        return CachedFbank(
            features=features,
            num_samples=int(metadata["num_samples"]),
            key=str(metadata["key"]),
            metadata=metadata,
        )

    def store(
        self,
        audio_path: Path,
        features: Any,
        num_samples: int,
        *,
        source_identity: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        import numpy as np

        if not self.writable:
            raise RuntimeError("cannot store into a read-only feature cache")
        if hasattr(features, "detach"):
            array = features.detach().cpu().numpy()
        else:
            array = np.asarray(features)
        array = np.asarray(array, dtype=np.float32, order="C")
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != FBANK_NUM_BINS:
            raise ValueError(
                "cached fbank must have shape (frames, {}), got {}".format(
                    FBANK_NUM_BINS, array.shape
                )
            )
        if not np.isfinite(array).all():
            raise ValueError("cached fbank contains NaN or Inf")
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, Integral)
            or int(num_samples) <= 0
        ):
            raise ValueError("num_samples must be a positive integer")

        if source_identity is None:
            source, key = self._current_identity(audio_path)
        else:
            source = dict(source_identity)
            key = _entry_key(source, self.frontend_sha256)
        data_path, metadata_path = self._paths(key)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(data_path.parent),
            prefix=".{}-".format(data_path.name),
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(data_path))
        finally:
            if temporary.exists():
                temporary.unlink()

        metadata = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "key": key,
            "source": source,
            "frontend_sha256": self.frontend_sha256,
            "data_path": data_path.relative_to(self.root).as_posix(),
            "data_size": int(data_path.stat().st_size),
            "dtype": "float32",
            "shape": [int(array.shape[0]), int(array.shape[1])],
            "num_samples": int(num_samples),
            "sample_rate": SAMPLE_RATE,
        }
        _atomic_write_json(metadata_path, metadata)
        return metadata


def prepare_cached_fbank(
    audio_path: str,
    cache_dir: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Process-pool entry point; large feature tensors never cross IPC."""
    import torch

    torch.set_num_threads(1)
    path = Path(audio_path).expanduser().resolve()
    cache = FbankFeatureCache(Path(cache_dir))
    if not force:
        metadata = cache.lookup_metadata(path)
        if metadata is not None:
            return {"status": "cached", "audio_path": str(path), **metadata}
    source_before = source_fingerprint(path)
    waveform = load_audio(path)
    features = compute_fbank(waveform, torch.device("cpu"))
    source_after = source_fingerprint(path)
    if source_after != source_before:
        raise RuntimeError(
            "audio changed while computing its feature cache: {}".format(path)
        )
    metadata = cache.store(
        path,
        features,
        int(waveform.numel()),
        source_identity=source_before,
    )
    return {"status": "computed", "audio_path": str(path), **metadata}
