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
"""Shared, model-independent artifacts for manifest KWS evaluation.

Unlike a conventional classifier, the streaming keyword decoder resets its
search state after a hit.  A high-threshold run therefore cannot be recovered
by filtering the scores emitted by a low-threshold run.  This module consumes
one *observed* result per ``(manifest trial, decoder threshold)`` and aggregates
those exact observations; it never re-thresholds a score.

The public surface intentionally contains no torch, k2, lhotse, numpy, or
matplotlib types.  Both command-line entry points can depend on it without
depending on each other or adding inference-time framework imports.
"""

from __future__ import annotations

import binascii
import csv
import json
import math
import os
import re
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
RESULTS_FILENAME = "results.jsonl"
SUMMARY_FILENAME = "summary.json"
THRESHOLD_SCAN_SUMMARY_FILENAME = "threshold_scan_summary.json"
THRESHOLD_SCAN_CSV_FILENAME = "threshold_scan.csv"
THRESHOLD_SCAN_PNG_FILENAME = "threshold_scan.png"
ARTIFACT_FILENAMES = (
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    THRESHOLD_SCAN_SUMMARY_FILENAME,
    THRESHOLD_SCAN_CSV_FILENAME,
    THRESHOLD_SCAN_PNG_FILENAME,
)

CLIP_SCAN_FIELDS = (
    "threshold",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "youden_j",
)
MUSAN_SCAN_FIELDS = (
    "threshold",
    "fp",
    "tn",
    "fpr",
    "false_alarm_events",
    "negative_exposure_hours",
    "fa_per_hour",
    "fa_per_1000_hours",
)
CLIP_SCAN_EXTENSION_FIELDS = (
    "num_labeled",
    "num_usable",
    "detection_rate",
)


class ArtifactError(ValueError):
    """Raised when normalized evaluation observations are inconsistent."""


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise ArtifactError(
            "{} must be a finite number, got {!r}".format(description, value)
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            "{} must be a finite number, got {!r}".format(description, value)
        ) from error
    if not math.isfinite(number):
        raise ArtifactError(
            "{} must be a finite number, got {!r}".format(description, value)
        )
    return number


def format_threshold(value: Any) -> str:
    """Return the stable threshold key used by both KWS entry points."""
    number = _finite_float(value, "threshold")
    if not 0.0 <= number <= 1.0:
        raise ArtifactError("threshold must be between 0 and 1: {}".format(value))
    if number == 0.0:
        number = 0.0
    return format(number, ".12g")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _json_ready_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ArtifactError("metadata must be JSON serializable: {}".format(error))
    return result


def build_result_record(
    *,
    source_manifest_row: int,
    audio_path: str,
    keyword: str,
    threshold: Any,
    events: Sequence[Mapping[str, Any]] = (),
    label: Optional[int] = None,
    audio_path_resolved: Optional[str] = None,
    duration_sec: Optional[float] = None,
    skipped: bool = False,
    error: Optional[str] = None,
    manifest_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one strict result for an actually decoded threshold/trial pair.

    ``qbyt_score`` is retained for compatibility with the referenced DMA-KWS
    result schema.  Here it means the maximum emitted keyword acoustic score
    at this exact decoder threshold, or zero when no event was emitted.
    ``detected`` is authoritative; consumers must not re-threshold the score.
    """
    if isinstance(source_manifest_row, bool) or not isinstance(
        source_manifest_row, int
    ):
        raise ArtifactError(
            "source_manifest_row must be an integer: {!r}".format(
                source_manifest_row
            )
        )
    row_number = source_manifest_row
    if row_number <= 0:
        raise ArtifactError("source_manifest_row must be positive")
    audio_path = str(audio_path)
    keyword = str(keyword).strip()
    if not audio_path:
        raise ArtifactError("audio_path must not be empty")
    if not keyword:
        raise ArtifactError("keyword must not be empty")
    threshold_number = float(format_threshold(threshold))
    if label is not None:
        if isinstance(label, bool) or not isinstance(label, int):
            raise ArtifactError("label must be 0, 1, or omitted")
        if label not in (0, 1):
            raise ArtifactError("label must be 0, 1, or omitted")
    if not isinstance(skipped, bool):
        raise ArtifactError("skipped must be a boolean")
    if duration_sec is not None:
        duration_sec = _finite_float(duration_sec, "duration_sec")
        if duration_sec <= 0.0:
            raise ArtifactError("duration_sec must be positive")

    normalized_events: List[Dict[str, Any]] = []
    scores: List[float] = []
    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            raise ArtifactError("event {} must be an object".format(index))
        event = dict(raw_event)
        if "score" not in event:
            raise ArtifactError("event {} has no score".format(index))
        score = _finite_float(event["score"], "event {} score".format(index))
        if not 0.0 <= score <= 1.0:
            raise ArtifactError(
                "event {} score must be between 0 and 1".format(index)
            )
        event["score"] = score
        try:
            json.dumps(event, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                "event {} must be JSON serializable: {}".format(index, exc)
            ) from exc
        normalized_events.append(event)
        scores.append(score)

    record: Dict[str, Any] = {
        "audio_path": audio_path,
        "keyword": keyword,
        "qbyt_score": max(scores) if scores else 0.0,
        "detected": bool(normalized_events),
        "threshold": threshold_number,
        "skipped": bool(skipped),
        "source_manifest_row": row_number,
        "detection_count": len(normalized_events),
        "score_semantics": "max_emitted_keyword_acoustic_score",
        "detections": normalized_events,
    }
    if label is not None:
        record["label"] = label
        if label == 0:
            record["false_alarm_events"] = len(normalized_events)
    if audio_path_resolved is not None:
        record["audio_path_resolved"] = str(audio_path_resolved)
    if duration_sec is not None:
        record["duration_sec"] = duration_sec
    if error is not None:
        record["error"] = str(error)
    metadata = _json_ready_mapping(manifest_meta)
    if metadata:
        record["manifest_meta"] = metadata
    return record


def _atomic_write_bytes(path: Path, data: bytes, overwrite: bool) -> None:
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
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, text: str, overwrite: bool) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def _atomic_write_json(path: Path, value: Any, overwrite: bool) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        overwrite=overwrite,
    )


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    overwrite: bool,
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


def _threshold_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labeled = [
        row
        for row in records
        if not bool(row.get("skipped", False)) and row.get("label") in (0, 1)
    ]
    tp = sum(bool(row["detected"]) and int(row["label"]) == 1 for row in labeled)
    tn = sum(not bool(row["detected"]) and int(row["label"]) == 0 for row in labeled)
    fp = sum(bool(row["detected"]) and int(row["label"]) == 0 for row in labeled)
    fn = sum(not bool(row["detected"]) and int(row["label"]) == 1 for row in labeled)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)
    event_count = sum(
        int(row.get("detection_count", 0))
        for row in records
        if not bool(row.get("skipped", False)) and row.get("label") == 0
    )
    negative_duration_sec = sum(
        float(row["duration_sec"])
        for row in records
        if not bool(row.get("skipped", False))
        and row.get("label") == 0
        and row.get("duration_sec") is not None
    )
    exposure_hours = negative_duration_sec / 3600.0
    usable = [row for row in records if not bool(row.get("skipped", False))]
    return {
        "threshold": float(records[0]["threshold"]) if records else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": _safe_divide(tp + tn, len(labeled)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2.0 * precision * recall, precision + recall),
        "fpr": fpr,
        "fnr": fnr,
        "youden_j": recall - fpr,
        "false_alarm_events": event_count,
        "negative_exposure_hours": exposure_hours,
        "fa_per_hour": _safe_divide(event_count, exposure_hours),
        "fa_per_1000_hours": _safe_divide(event_count * 1000.0, exposure_hours),
        "detection_rate": _safe_divide(
            sum(bool(row.get("detected", False)) for row in usable), len(usable)
        ),
        "num_labeled": len(labeled),
        "num_usable": len(usable),
    }


def _category_slug(value: str, used: Dict[str, str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "other"
    candidate = base
    suffix = 2
    while candidate in used and used[candidate] != value:
        candidate = "{}_{}".format(base, suffix)
        suffix += 1
    used[candidate] = value
    return candidate


def _category_name(record: Mapping[str, Any]) -> Optional[str]:
    direct = record.get("category")
    if direct is not None and str(direct).strip():
        return str(direct).strip()
    metadata = record.get("manifest_meta")
    if isinstance(metadata, Mapping):
        value = metadata.get("category", metadata.get("subset"))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _add_subset_metrics(
    metric_row: Dict[str, Any],
    records: Sequence[Mapping[str, Any]],
    slugs: Mapping[str, str],
) -> None:
    for slug, category in sorted(slugs.items()):
        subset = [
            row
            for row in records
            if not bool(row.get("skipped", False))
            and row.get("label") == 0
            and _category_name(row) == category
        ]
        fp = sum(bool(row.get("detected", False)) for row in subset)
        tn = len(subset) - fp
        event_count = sum(int(row.get("detection_count", 0)) for row in subset)
        duration_sec = sum(
            float(row["duration_sec"])
            for row in subset
            if row.get("duration_sec") is not None
        )
        hours = duration_sec / 3600.0
        metric_row["subset_{}_fp".format(slug)] = fp
        metric_row["subset_{}_tn".format(slug)] = tn
        metric_row["subset_{}_fpr".format(slug)] = _safe_divide(fp, fp + tn)
        metric_row["subset_{}_false_alarm_events".format(slug)] = event_count
        metric_row["subset_{}_exposure_hours".format(slug)] = hours
        if hours:
            metric_row["subset_{}_fa_per_hour".format(slug)] = _safe_divide(
                event_count, hours
            )


def _best_row(
    rows: Sequence[Mapping[str, Any]], primary: str, secondary: str
) -> Dict[str, Any]:
    return dict(
        max(
            rows,
            key=lambda row: (
                float(row[primary]),
                float(row[secondary]),
                float(row["threshold"]),
            ),
        )
    )


def _sampled_auc(rows: Sequence[Mapping[str, Any]]) -> Tuple[float, List[float]]:
    """Integrate only the sampled exact-decoder operating points.

    There is no coherent per-trial score ranking across independent decoder
    thresholds, so this is deliberately not exposed as a conventional ROC
    AUC. Duplicate FPR values retain the best observed recall; endpoints are
    never invented.
    """
    best_by_fpr: Dict[float, float] = {}
    for row in rows:
        fpr = float(row["fpr"])
        recall = float(row["recall"])
        best_by_fpr[fpr] = max(best_by_fpr.get(fpr, 0.0), recall)
    points = sorted(best_by_fpr.items())
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    span = [points[0][0], points[-1][0]] if points else [0.0, 0.0]
    return min(1.0, max(0.0, area)), span


def _nonmonotonic_warnings(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    warnings = []
    ordered = sorted(rows, key=lambda row: float(row["threshold"]))
    for previous, current in zip(ordered, ordered[1:]):
        for metric in ("fpr", "fa_per_hour", "recall", "detection_rate"):
            if float(current[metric]) > float(previous[metric]) + 1e-12:
                warnings.append(
                    "{} increased as threshold rose: {} at {} -> {} at {}".format(
                        metric,
                        previous[metric],
                        previous["threshold"],
                        current[metric],
                        current["threshold"],
                    )
                )
    return warnings


# Five-column bitmap font used only when matplotlib is unavailable.  Each byte
# is one vertical column, with bit zero at the top of the glyph.
_FONT_5X7 = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00),
    "-": (0x08, 0x08, 0x08, 0x08, 0x08),
    ".": (0x00, 0x60, 0x60, 0x00, 0x00),
    "/": (0x20, 0x10, 0x08, 0x04, 0x02),
    "0": (0x3E, 0x51, 0x49, 0x45, 0x3E),
    "1": (0x00, 0x42, 0x7F, 0x40, 0x00),
    "2": (0x42, 0x61, 0x51, 0x49, 0x46),
    "3": (0x21, 0x41, 0x45, 0x4B, 0x31),
    "4": (0x18, 0x14, 0x12, 0x7F, 0x10),
    "5": (0x27, 0x45, 0x45, 0x45, 0x39),
    "6": (0x3C, 0x4A, 0x49, 0x49, 0x30),
    "7": (0x01, 0x71, 0x09, 0x05, 0x03),
    "8": (0x36, 0x49, 0x49, 0x49, 0x36),
    "9": (0x06, 0x49, 0x49, 0x29, 0x1E),
    "A": (0x7E, 0x11, 0x11, 0x11, 0x7E),
    "B": (0x7F, 0x49, 0x49, 0x49, 0x36),
    "C": (0x3E, 0x41, 0x41, 0x41, 0x22),
    "D": (0x7F, 0x41, 0x41, 0x22, 0x1C),
    "E": (0x7F, 0x49, 0x49, 0x49, 0x41),
    "F": (0x7F, 0x09, 0x09, 0x09, 0x01),
    "G": (0x3E, 0x41, 0x49, 0x49, 0x7A),
    "H": (0x7F, 0x08, 0x08, 0x08, 0x7F),
    "I": (0x00, 0x41, 0x7F, 0x41, 0x00),
    "J": (0x20, 0x40, 0x41, 0x3F, 0x01),
    "K": (0x7F, 0x08, 0x14, 0x22, 0x41),
    "L": (0x7F, 0x40, 0x40, 0x40, 0x40),
    "M": (0x7F, 0x02, 0x0C, 0x02, 0x7F),
    "N": (0x7F, 0x04, 0x08, 0x10, 0x7F),
    "O": (0x3E, 0x41, 0x41, 0x41, 0x3E),
    "P": (0x7F, 0x09, 0x09, 0x09, 0x06),
    "Q": (0x3E, 0x41, 0x51, 0x21, 0x5E),
    "R": (0x7F, 0x09, 0x19, 0x29, 0x46),
    "S": (0x46, 0x49, 0x49, 0x49, 0x31),
    "T": (0x01, 0x01, 0x7F, 0x01, 0x01),
    "U": (0x3F, 0x40, 0x40, 0x40, 0x3F),
    "V": (0x1F, 0x20, 0x40, 0x20, 0x1F),
    "W": (0x3F, 0x40, 0x38, 0x40, 0x3F),
    "X": (0x63, 0x14, 0x08, 0x14, 0x63),
    "Y": (0x07, 0x08, 0x70, 0x08, 0x07),
    "Z": (0x61, 0x51, 0x49, 0x45, 0x43),
}


class _RasterPlot:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(b"\xff" * (width * height * 3))

    def pixel(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            index = (y * self.width + x) * 3
            self.pixels[index : index + 3] = bytes(color)

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: Tuple[int, int, int],
        width: int = 1,
    ) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        radius = max(0, width // 2)
        while True:
            for offset_x in range(-radius, radius + 1):
                for offset_y in range(-radius, radius + 1):
                    self.pixel(x0 + offset_x, y0 + offset_y, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def text(
        self,
        x: int,
        y: int,
        value: str,
        color: Tuple[int, int, int] = (17, 24, 39),
        scale: int = 2,
    ) -> None:
        cursor = x
        for character in value.upper():
            columns = _FONT_5X7.get(character, _FONT_5X7[" "])
            for column_index, bits in enumerate(columns):
                for row in range(7):
                    if bits & (1 << row):
                        for dx in range(scale):
                            for dy in range(scale):
                                self.pixel(
                                    cursor + column_index * scale + dx,
                                    y + row * scale + dy,
                                    color,
                                )
            cursor += 6 * scale

    def png(self) -> bytes:
        raw = bytearray()
        row_bytes = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * row_bytes
            raw.extend(self.pixels[start : start + row_bytes])

        def chunk(kind: bytes, payload: bytes) -> bytes:
            checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", checksum)
            )

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
            + chunk(b"IEND", b"")
        )


def _fallback_plot_png(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> bytes:
    width, height = 1024, 832
    left, right, top, bottom = 125, 55, 100, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    raster = _RasterPlot(width, height)
    ordered = sorted(rows, key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in ordered]
    x_min, x_max = min(thresholds), max(thresholds)
    if x_min == x_max:
        x_min -= 0.05
        x_max += 0.05

    def x_pos(value: float) -> int:
        return round(left + (value - x_min) / (x_max - x_min) * plot_width)

    def y_pos(value: float) -> int:
        return round(top + (1.0 - value) * plot_height)

    for index in range(6):
        value = index / 5.0
        y = y_pos(value)
        raster.line(left, y, left + plot_width, y, (225, 229, 235))
        raster.text(55, y - 7, format(value, ".1f"), scale=2)
    raster.line(left, top, left, top + plot_height, (17, 24, 39), width=2)
    raster.line(
        left,
        top + plot_height,
        left + plot_width,
        top + plot_height,
        (17, 24, 39),
        width=2,
    )
    for index in range(6):
        threshold = x_min + (x_max - x_min) * index / 5.0
        x = x_pos(threshold)
        raster.line(x, top, x, top + plot_height, (243, 244, 246))
        raster.text(x - 25, top + plot_height + 20, format(threshold, ".2g"), scale=2)

    colors = ((31, 119, 180), (214, 39, 40), (44, 160, 44))
    names = {
        "recall": "RECALL",
        "fpr": "FALSE POSITIVE RATE",
        "detection_rate": "DETECTION RATE",
    }
    for metric_index, metric in enumerate(metrics):
        color = colors[metric_index % len(colors)]
        points = [
            (x_pos(float(row["threshold"])), y_pos(float(row[metric])))
            for row in ordered
        ]
        previous_x, previous_y = points[0]
        for current_x, current_y in points[1:]:
            # Matplotlib's where="pre": the new value covers the interval
            # immediately before its x coordinate.
            raster.line(previous_x, previous_y, previous_x, current_y, color, 3)
            raster.line(previous_x, current_y, current_x, current_y, color, 3)
            previous_x, previous_y = current_x, current_y
        raster.text(
            left + metric_index * 260,
            62,
            names[metric],
            color=color,
            scale=2,
        )

    if len(metrics) == 1:
        title = "STAGE II {} VS THRESHOLD".format(names[metrics[0]])
    else:
        title = "KWS RECALL AND FPR VS THRESHOLD"
    raster.text(max(20, width // 2 - len(title) * 9), 25, title, scale=3)
    raster.text(width // 2 - 80, height - 55, "THRESHOLD", scale=3)
    return raster.png()


def _write_threshold_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    overwrite: bool,
) -> Dict[str, Any]:
    metric_labels = {
        "recall": "Recall",
        "fpr": "False Positive Rate",
        "detection_rate": "Detection Rate",
    }
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        _atomic_write_bytes(
            path,
            _fallback_plot_png(rows, metrics),
            overwrite=overwrite,
        )
        result = {
            "status": "generated",
            "metrics": list(metrics),
            "path": str(path.resolve()),
            "backend": "stdlib_png",
        }
        if len(metrics) == 1:
            result["metric"] = metrics[0]
        return result

    if path.exists() and not overwrite:
        raise FileExistsError(
            "output already exists (pass --overwrite): {}".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}-".format(path.stem), suffix=".png"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    ordered = sorted(rows, key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    try:
        for metric in metrics:
            axis.step(
                thresholds,
                [float(row[metric]) for row in ordered],
                where="pre",
                linewidth=2.0,
                label=metric_labels[metric] if len(metrics) > 1 else None,
            )
        x_min, x_max = min(thresholds), max(thresholds)
        if x_min == x_max:
            x_min -= 0.05
            x_max += 0.05
        if len(metrics) == 1:
            ylabel = metric_labels[metrics[0]]
            title = "Stage II {} vs Threshold".format(ylabel)
        else:
            ylabel = "Rate"
            title = "KWS Recall and FPR vs Threshold"
            axis.legend()
        axis.set(
            xlim=(x_min, x_max),
            ylim=(0.0, 1.0),
            xlabel="Threshold",
            ylabel=ylabel,
            title=title,
        )
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(temporary, dpi=160, bbox_inches="tight")
        os.replace(str(temporary), str(path))
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    result = {
        "status": "generated",
        "metrics": list(metrics),
        "path": str(path.resolve()),
        "backend": "matplotlib",
    }
    if len(metrics) == 1:
        result["metric"] = metrics[0]
    return result


def _normalize_records(
    records: Sequence[Mapping[str, Any]], thresholds: Sequence[Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    threshold_keys = sorted(
        {format_threshold(value) for value in thresholds}, key=float
    )
    if not threshold_keys:
        raise ArtifactError("at least one threshold is required")
    if not records:
        raise ArtifactError("at least one evaluation result is required")
    requested = set(threshold_keys)
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(records):
        row = dict(raw)
        required = {
            "audio_path",
            "keyword",
            "qbyt_score",
            "detected",
            "threshold",
            "skipped",
            "source_manifest_row",
            "detection_count",
            "detections",
        }
        missing = required.difference(row)
        if missing:
            raise ArtifactError(
                "result {} is missing field(s): {}".format(
                    index, ", ".join(sorted(missing))
                )
            )
        threshold_key = format_threshold(row["threshold"])
        if threshold_key not in requested:
            raise ArtifactError(
                "result {} uses unrequested threshold {}".format(index, threshold_key)
            )
        row["threshold"] = float(threshold_key)
        raw_row_number = row["source_manifest_row"]
        if isinstance(raw_row_number, bool) or not isinstance(raw_row_number, int):
            raise ArtifactError("source_manifest_row must be an integer")
        row_number = raw_row_number
        if row_number <= 0:
            raise ArtifactError("source_manifest_row must be positive")
        row["source_manifest_row"] = row_number
        identity = (threshold_key, row_number)
        if identity in seen:
            raise ArtifactError(
                "duplicate result for threshold {} and source_manifest_row {}".format(
                    threshold_key, row_number
                )
            )
        seen.add(identity)
        score = _finite_float(row["qbyt_score"], "qbyt_score")
        if not 0.0 <= score <= 1.0:
            raise ArtifactError("qbyt_score must be between 0 and 1")
        row["qbyt_score"] = score
        if not isinstance(row["detected"], bool):
            raise ArtifactError("detected must be a boolean")
        if not isinstance(row["skipped"], bool):
            raise ArtifactError("skipped must be a boolean")
        if row.get("label") is not None:
            label = row["label"]
            if isinstance(label, bool) or not isinstance(label, int):
                raise ArtifactError("label must be 0 or 1")
            if label not in (0, 1):
                raise ArtifactError("label must be 0 or 1")
            row["label"] = label
        if row.get("duration_sec") is not None:
            duration = _finite_float(row["duration_sec"], "duration_sec")
            if duration <= 0.0:
                raise ArtifactError("duration_sec must be positive")
            row["duration_sec"] = duration
        event_count = row["detection_count"]
        if isinstance(event_count, bool) or not isinstance(event_count, int):
            raise ArtifactError("detection_count must be an integer")
        if event_count < 0:
            raise ArtifactError("detection_count must be non-negative")
        raw_events = row["detections"]
        if not isinstance(raw_events, list):
            raise ArtifactError("detections must be a list")
        events = list(raw_events)
        row["detections"] = events
        if event_count != len(events):
            raise ArtifactError("detection_count must agree with detections")
        event_scores = []
        for event_index, event in enumerate(events):
            if not isinstance(event, Mapping) or "score" not in event:
                raise ArtifactError(
                    "detection {} must be an object with a score".format(
                        event_index
                    )
                )
            event_score = _finite_float(
                event["score"], "detection {} score".format(event_index)
            )
            if not 0.0 <= event_score <= 1.0:
                raise ArtifactError("detection score must be between 0 and 1")
            if event_score != event["score"]:
                event = dict(event)
                event["score"] = event_score
                events[event_index] = event
            try:
                json.dumps(event, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ArtifactError(
                    "detection {} must be JSON serializable".format(event_index)
                ) from error
            event_scores.append(event_score)
        expected_score = max(event_scores) if event_scores else 0.0
        if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise ArtifactError("qbyt_score must equal the maximum detection score")
        if bool(row["detected"]) != bool(event_count):
            raise ArtifactError("detected must agree with detection_count")
        if row.get("label") == 0:
            false_alarm_events = row.get("false_alarm_events")
            if isinstance(false_alarm_events, bool) or not isinstance(
                false_alarm_events, int
            ):
                raise ArtifactError(
                    "negative results require integer false_alarm_events"
                )
            if false_alarm_events != event_count:
                raise ArtifactError(
                    "false_alarm_events must agree with detection_count"
                )
        normalized.append(row)
    normalized.sort(
        key=lambda row: (int(row["source_manifest_row"]), float(row["threshold"]))
    )
    by_source: Dict[int, List[Mapping[str, Any]]] = {}
    for row in normalized:
        by_source.setdefault(int(row["source_manifest_row"]), []).append(row)
    requested_thresholds = set(threshold_keys)
    invariant_fields = (
        "audio_path",
        "audio_path_resolved",
        "keyword",
        "label",
        "duration_sec",
        "skipped",
        "error",
        "manifest_meta",
    )
    for source_row, source_records in sorted(by_source.items()):
        observed_thresholds = {
            format_threshold(row["threshold"]) for row in source_records
        }
        if observed_thresholds != requested_thresholds:
            missing = sorted(
                requested_thresholds.difference(observed_thresholds), key=float
            )
            raise ArtifactError(
                "source_manifest_row {} is missing threshold result(s): {}".format(
                    source_row, ", ".join(missing) or "none"
                )
            )
        reference = source_records[0]
        for row in source_records[1:]:
            changed = [
                name
                for name in invariant_fields
                if row.get(name) != reference.get(name)
            ]
            if changed:
                raise ArtifactError(
                    "source_manifest_row {} changes trial field(s) across "
                    "thresholds: {}".format(source_row, ", ".join(changed))
                )
    return normalized, threshold_keys


def _infer_mode(records: Sequence[Mapping[str, Any]], requested: str) -> str:
    if requested not in {"auto", "clips", "musan"}:
        raise ArtifactError("mode must be auto, clips, or musan")
    usable = [row for row in records if not bool(row.get("skipped", False))]
    labels = {
        int(row["label"])
        for row in usable
        if row.get("label") is not None
    }
    if requested == "musan":
        if not usable:
            raise ArtifactError("musan mode requires at least one usable trial")
        if any(row.get("label") != 0 for row in records):
            raise ArtifactError("musan mode requires negative-only labels")
        if any(row.get("duration_sec") is None for row in usable):
            raise ArtifactError("musan mode requires duration_sec for every trial")
    if requested != "auto":
        return requested
    if (
        labels == {0}
        and usable
        and all(row.get("label") == 0 for row in records)
        and all(row.get("duration_sec") for row in usable)
    ):
        return "musan"
    return "clips"


def write_evaluation_artifacts(
    *,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Any],
    manifest_path: Path,
    overwrite: bool,
    mode: str = "auto",
    summary_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Write the five standard KWS evaluation artifacts.

    The PNG always exists on success.  Matplotlib reproduces the referenced
    plotting style when installed; a small standard-library renderer preserves
    the same step-curve layout when evaluation hosts omit plotting packages.
    All five outputs are fully staged before any existing artifact is replaced;
    each final file replacement is atomic.
    """
    output_dir = output_dir.expanduser().resolve()
    paths = {}
    for name in ARTIFACT_FILENAMES:
        candidate = output_dir / name
        if candidate.is_symlink():
            raise ArtifactError(
                "evaluation artifact path must not be a symlink: {}".format(
                    candidate
                )
            )
        if candidate.exists() and not candidate.is_file():
            raise ArtifactError(
                "evaluation artifact path is not a regular file: {}".format(
                    candidate
                )
            )
        resolved = candidate.resolve()
        if resolved.parent != output_dir:
            raise ArtifactError(
                "evaluation artifact escapes output directory: {}".format(candidate)
            )
        paths[name] = resolved
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "output already exists (pass --overwrite): {}".format(existing[0])
            )

    normalized, threshold_keys = _normalize_records(records, thresholds)
    metadata = _json_ready_mapping(summary_metadata)
    resolved_mode = _infer_mode(normalized, mode)
    by_threshold: Dict[str, List[Mapping[str, Any]]] = {
        key: [] for key in threshold_keys
    }
    for row in normalized:
        by_threshold[format_threshold(row["threshold"])].append(row)

    category_slugs: Dict[str, str] = {}
    if resolved_mode == "musan":
        categories = sorted(
            {
                category
                for category in (_category_name(row) for row in normalized)
                if category is not None
            }
        )
        used: Dict[str, str] = {}
        for category in categories:
            slug = _category_slug(category, used)
            category_slugs[slug] = category

    metric_rows = []
    for threshold_key in reversed(threshold_keys):
        row = _threshold_metrics(by_threshold[threshold_key])
        row["threshold"] = float(threshold_key)
        if resolved_mode == "musan":
            _add_subset_metrics(row, by_threshold[threshold_key], category_slugs)
        metric_rows.append(row)

    if resolved_mode == "clips":
        csv_fields = list(CLIP_SCAN_FIELDS + CLIP_SCAN_EXTENSION_FIELDS)
    else:
        csv_fields = list(MUSAN_SCAN_FIELDS)
        csv_fields.extend(
            sorted(
                {
                    name
                    for row in metric_rows
                    for name in row
                    if name.startswith("subset_")
                    and name not in MUSAN_SCAN_FIELDS
                }
            )
        )

    results_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in normalized
    )

    usable = [row for row in normalized if not bool(row.get("skipped", False))]
    labeled = [row for row in usable if row.get("label") in (0, 1)]
    positives = sum(int(row["label"]) == 1 for row in labeled)
    negatives = sum(int(row["label"]) == 0 for row in labeled)
    if positives and negatives:
        plot_metrics = ("recall", "fpr")
    elif positives:
        plot_metrics = ("recall",)
    elif negatives:
        plot_metrics = ("fpr",)
    else:
        plot_metrics = ("detection_rate",)

    unique_trials = {
        int(row["source_manifest_row"]): row for row in normalized
    }
    skipped_trials = {
        int(row["source_manifest_row"])
        for row in normalized
        if bool(row.get("skipped", False))
    }
    unique_labeled_trials = {
        int(row["source_manifest_row"]): int(row["label"])
        for row in normalized
        if not bool(row.get("skipped", False)) and row.get("label") in (0, 1)
    }
    usable_trial_ids = set(unique_trials).difference(skipped_trials)
    total_hours = sum(
        float(row["duration_sec"])
        for source_row, row in unique_trials.items()
        if source_row in usable_trial_ids and row.get("duration_sec") is not None
    ) / 3600.0
    semantics = {
        "result_row": "one exact decoder threshold x input manifest trial",
        "detected": "one or more emitted keyword events at that exact threshold",
        "qbyt_score": (
            "maximum emitted mean keyword-token acoustic probability; zero for "
            "no event; never re-threshold this field"
        ),
        "fp_tn": "triggered/non-triggered negative trials; at most one per trial",
        "false_alarm_events": "all emitted events; multiple events per trial count",
        "fa_per_hour": "false_alarm_events / negative source-trial exposure hours",
        "scan": (
            "independent decoder state per threshold "
            "(exact, possibly non-monotonic)"
        ),
        "curve": (
            "only listed thresholds are measured; step segments between them "
            "are visualization, not inferred decoder results"
        ),
        "sampled_auc": (
            "trapezoidal area over observed operating points only; not a "
            "conventional score-ranked ROC AUC"
        ),
    }
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path.expanduser().resolve()),
        "num_samples": len(unique_trials),
        "num_result_rows": len(normalized),
        "output_dir": str(output_dir),
        "num_skipped": len(skipped_trials),
        "thresholds": [float(value) for value in threshold_keys],
        "mode": resolved_mode,
        "semantics": semantics,
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
    }
    if resolved_mode == "musan":
        summary["total_hours"] = total_hours
    if len(metric_rows) == 1:
        if unique_labeled_trials:
            summary["metrics"] = {
                name: metric_rows[0][name]
                for name in CLIP_SCAN_FIELDS
                if name in metric_rows[0]
            }
            summary["metrics"]["num_labeled"] = metric_rows[0]["num_labeled"]
        else:
            summary["detection_metrics"] = {
                name: metric_rows[0][name]
                for name in ("threshold", "num_usable", "detection_rate")
            }
        if resolved_mode == "musan" and "metrics" in summary:
            summary["metrics"].update(
                {
                    "fa_per_hour": metric_rows[0]["fa_per_hour"],
                    "fa_per_1000_hours": metric_rows[0]["fa_per_1000_hours"],
                    "false_alarm_events": metric_rows[0]["false_alarm_events"],
                    "negative_exposure_hours": metric_rows[0][
                        "negative_exposure_hours"
                    ],
                }
            )
    if metadata:
        summary["provenance"] = metadata

    scores = [float(row["qbyt_score"]) for row in usable]
    scan_summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "results": str(paths[RESULTS_FILENAME].resolve()),
        "source_summary": str(paths[SUMMARY_FILENAME].resolve()),
        "mode": resolved_mode,
        "num_samples": len(usable_trial_ids),
        "num_input_rows": len(unique_trials),
        "num_result_rows": len(normalized),
        "num_skipped_excluded": len(skipped_trials),
        "num_unlabeled_excluded": len(
            usable_trial_ids.difference(unique_labeled_trials)
        ),
        "positives": sum(value == 1 for value in unique_labeled_trials.values()),
        "negatives": sum(value == 0 for value in unique_labeled_trials.values()),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "num_thresholds": len(threshold_keys),
        "threshold_step": None,
        "workers": 1,
        "curve_csv": str(paths[THRESHOLD_SCAN_CSV_FILENAME].resolve()),
        "semantics": semantics,
        "warnings": _nonmonotonic_warnings(metric_rows),
    }
    if resolved_mode == "clips" and positives and negatives:
        best_youden = _best_row(metric_rows, "youden_j", "f1")
        best_f1 = _best_row(metric_rows, "f1", "youden_j")
        eer_row = min(
            metric_rows,
            key=lambda row: (
                abs(float(row["fpr"]) - float(row["fnr"])),
                -float(row["threshold"]),
            ),
        )
        sampled_auc, sampled_auc_span = _sampled_auc(metric_rows)
        scan_summary.update(
            {
                "selection_scope": "sampled_exact_decoder_thresholds_only",
                "sampled_auc": sampled_auc,
                "sampled_auc_method": (
                    "trapezoidal_observed_fpr_upper_envelope_no_endpoints"
                ),
                "sampled_auc_fpr_span": sampled_auc_span,
                "sampled_eer": (
                    float(eer_row["fpr"]) + float(eer_row["fnr"])
                )
                / 2.0,
                "sampled_eer_threshold": float(eer_row["threshold"]),
                "sampled_eer_method": "closest_sampled_operating_point",
                "best_youden": {
                    name: best_youden[name] for name in CLIP_SCAN_FIELDS
                },
                "best_f1": {name: best_f1[name] for name in CLIP_SCAN_FIELDS},
            }
        )
    if resolved_mode == "musan":
        subset_summary = {}
        first_threshold_rows = (
            by_threshold[threshold_keys[0]] if threshold_keys else []
        )
        for slug, category in sorted(category_slugs.items()):
            subset = [
                row
                for row in first_threshold_rows
                if not bool(row.get("skipped", False))
                and _category_name(row) == category
            ]
            subset_summary[slug] = {
                "name": category,
                "num_samples": len(subset),
                "total_hours": sum(
                    float(row["duration_sec"])
                    for row in subset
                    if row.get("duration_sec") is not None
                )
                / 3600.0,
            }
        scan_summary["total_hours"] = total_hours
        scan_summary["subsets"] = subset_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output_dir), prefix=".kws-eval-artifacts-"
    ) as staging_name:
        staging_dir = Path(staging_name)
        staging_paths = {
            name: staging_dir / name for name in ARTIFACT_FILENAMES
        }
        plot_summary = _write_threshold_plot(
            staging_paths[THRESHOLD_SCAN_PNG_FILENAME],
            metric_rows,
            plot_metrics,
            overwrite=False,
        )
        plot_summary["path"] = str(
            paths[THRESHOLD_SCAN_PNG_FILENAME].resolve()
        )
        scan_summary["plot"] = plot_summary
        _atomic_write_text(
            staging_paths[RESULTS_FILENAME], results_text, overwrite=False
        )
        _atomic_write_csv(
            staging_paths[THRESHOLD_SCAN_CSV_FILENAME],
            csv_fields,
            metric_rows,
            overwrite=False,
        )
        _atomic_write_json(
            staging_paths[SUMMARY_FILENAME], summary, overwrite=False
        )
        _atomic_write_json(
            staging_paths[THRESHOLD_SCAN_SUMMARY_FILENAME],
            scan_summary,
            overwrite=False,
        )
        if not overwrite:
            existing = [path for path in paths.values() if path.exists()]
            if existing:
                raise FileExistsError(
                    "output already exists (pass --overwrite): {}".format(
                        existing[0]
                    )
                )
        commit_order = (
            RESULTS_FILENAME,
            THRESHOLD_SCAN_CSV_FILENAME,
            THRESHOLD_SCAN_PNG_FILENAME,
            SUMMARY_FILENAME,
            THRESHOLD_SCAN_SUMMARY_FILENAME,
        )
        for name in commit_order:
            os.replace(str(staging_paths[name]), str(paths[name]))
    return {name: str(path.resolve()) for name, path in paths.items()}


__all__ = [
    "ARTIFACT_FILENAMES",
    "ArtifactError",
    "build_result_record",
    "format_threshold",
    "write_evaluation_artifacts",
]
