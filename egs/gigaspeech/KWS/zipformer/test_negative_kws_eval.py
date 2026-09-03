#!/usr/bin/env python3

import contextlib
import csv
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("negative_kws_eval.py")
SPEC = importlib.util.spec_from_file_location("negative_kws_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_wav(path, duration_sec, sample_rate=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack("<h", 0) * frame_count)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quiet_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        return MODULE.main(argv), stdout.getvalue(), stderr.getvalue()


class TestThresholdParsing(unittest.TestCase):
    def test_explicit_thresholds_are_sorted_and_deduplicated(self):
        self.assertEqual(
            MODULE.parse_thresholds(["0.8", "0.2", "0.80", "1", "0"]),
            ["0", "0.2", "0.8", "1"],
        )

    def test_comma_and_space_separated_thresholds(self):
        self.assertEqual(
            MODULE.parse_thresholds(["0.7,0.3", "0.5 0.3"]),
            ["0.3", "0.5", "0.7"],
        )

    def test_inclusive_ascending_and_descending_ranges(self):
        self.assertEqual(
            MODULE.parse_thresholds(["0.2:0.5:0.1", "0.8:0.6:-0.1"]),
            ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
        )

    def test_invalid_threshold_specs_are_rejected(self):
        cases = (
            [],
            [""],
            ["bad"],
            ["nan"],
            ["-0.01"],
            ["1.01"],
            ["0:1"],
            ["0:1:0"],
            ["0:1:-0.1"],
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(MODULE.EvaluationError):
                    MODULE.parse_thresholds(values)

    def test_sweep_owned_options_reject_child_abbreviations(self):
        for value in (
            "--man",
            "--output-m",
            "--overw",
            "--keywords-t",
            "--progress",
            "--no-progress",
        ):
            with self.subTest(value=value):
                with self.assertRaises(MODULE.EvaluationError):
                    MODULE._validate_decoder_args([value, "ignored"])
        self.assertEqual(
            MODULE._validate_decoder_args(["--model-config", "model.json"]),
            ["--model-config", "model.json"],
        )


class TestProgress(unittest.TestCase):
    def test_sweep_progress_flags_are_tristate(self):
        parser = MODULE.get_parser()
        base = [
            "sweep",
            "--manifest",
            "manifest.csv",
            "--output-dir",
            "output",
            "--thresholds",
            "0.2",
        ]

        self.assertIsNone(parser.parse_args(base).progress)
        self.assertTrue(parser.parse_args(base + ["--progress"]).progress)
        self.assertFalse(parser.parse_args(base + ["--no-progress"]).progress)

    def test_progress_event_parser_rejects_invalid_payloads(self):
        prefix = MODULE.PROGRESS_EVENT_PREFIX
        progress_module = sys.modules[MODULE.parse_progress_event.__module__]
        emitted = io.StringIO()
        progress_module.emit_progress_event(
            completed=2,
            total=3,
            stream=emitted,
            clips=4,
        )
        self.assertEqual(
            MODULE.parse_progress_event(emitted.getvalue()),
            {"completed": 2, "total": 3, "clips": 4},
        )
        for line in (
            "ordinary output\n",
            prefix + "not-json\n",
            prefix + '{"completed":-1,"total":3}\n',
            prefix + '{"completed":4,"total":3}\n',
            prefix + '{"completed":1.5,"total":3}\n',
            prefix + '{"completed":true,"total":3}\n',
        ):
            with self.subTest(line=line):
                self.assertIsNone(MODULE.parse_progress_event(line))

    def test_run_decoder_extracts_events_and_preserves_normal_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.py"
            child.write_text(
                "import os, sys\n"
                "print('stdout line', flush=True)\n"
                "print('stderr line', file=sys.stderr, flush=True)\n"
                "print('__ICEFALL_KWS_PROGRESS__='\n"
                "      '{\"completed\":1,\"total\":2,\"clips\":3}', flush=True)\n"
                "print('__ICEFALL_KWS_PROGRESS__=bad-json', flush=True)\n"
                "assert os.environ.get('ICEFALL_KWS_PROGRESS_EVENTS') == '1'\n",
                encoding="utf-8",
            )
            log_path = root / "runner.log"
            events = []

            return_code = MODULE._run_decoder(
                [sys.executable, str(child)],
                log_path,
                on_progress=events.append,
            )

            self.assertEqual(return_code, 0)
            self.assertEqual(events, [{"completed": 1, "total": 2, "clips": 3}])
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("stdout line", log)
            self.assertIn("stderr line", log)
            self.assertIn("bad-json", log)
            self.assertNotIn('"completed":1', log)

    def test_progress_callback_failure_does_not_stop_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.py"
            child.write_text(
                "print('__ICEFALL_KWS_PROGRESS__='\n"
                "      '{\"completed\":1,\"total\":1}', flush=True)\n"
                "print('decoder finished', flush=True)\n",
                encoding="utf-8",
            )
            log_path = root / "runner.log"

            def broken_callback(event):
                del event
                raise RuntimeError("display failed")

            return_code = MODULE._run_decoder(
                [sys.executable, str(child)],
                log_path,
                on_progress=broken_callback,
            )

            self.assertEqual(return_code, 0)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("progress display disabled", log)
            self.assertIn("decoder finished", log)

    def test_run_decoder_terminates_child_when_progress_stream_fails(self):
        class BrokenStdout:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def __iter__(self):
                raise KeyboardInterrupt

        class FakeProcess:
            def __init__(self):
                self.stdout = BrokenStdout()
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True
                return -15

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "runner.log"
            process = FakeProcess()
            with mock.patch.object(
                MODULE.subprocess, "Popen", return_value=process
            ):
                with self.assertRaises(KeyboardInterrupt):
                    MODULE._run_decoder(
                        [sys.executable, "decoder.py"],
                        log_path,
                        on_progress=lambda event: None,
                    )

            self.assertTrue(process.terminated)
            self.assertTrue(process.waited)
            self.assertFalse(log_path.exists())

    def test_no_progress_clears_private_event_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.py"
            child.write_text(
                "import os\n"
                "assert 'ICEFALL_KWS_PROGRESS_EVENTS' not in os.environ\n"
                "print('clean environment')\n",
                encoding="utf-8",
            )
            log_path = root / "runner.log"
            with mock.patch.dict(
                MODULE.os.environ,
                {MODULE.PROGRESS_EVENT_ENV: "1"},
                clear=False,
            ):
                return_code = MODULE._run_decoder(
                    [sys.executable, str(child)], log_path
                )

            self.assertEqual(return_code, 0)
            self.assertIn(
                "clean environment", log_path.read_text(encoding="utf-8")
            )

    def test_console_progress_auto_disable_and_plain_fallback(self):
        disabled_stream = io.StringIO()
        display = MODULE.ConsoleProgress(
            total=2,
            description="Hidden",
            enabled=None,
            stream=disabled_stream,
        )
        list(display.track(range(2)))
        display.complete()
        self.assertEqual(disabled_stream.getvalue(), "")

        real_import = __import__

        def without_rich(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("rich unavailable")
            return real_import(name, *args, **kwargs)

        fallback_stream = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=without_rich):
            display = MODULE.ConsoleProgress(
                total=2,
                description="Fallback",
                enabled=True,
                stream=fallback_stream,
                unit="trials",
            )
            display.start()
            display.update(completed=1, status="clips=0")
            display.complete(status="done")
            display.stop()

        output = fallback_stream.getvalue()
        self.assertIn("Fallback", output)
        self.assertIn("2/2 trials", output)
        self.assertIn("done", output)

    def test_console_progress_output_failures_are_nonfatal(self):
        class BrokenStream:
            def isatty(self):
                return False

            def write(self, value):
                del value
                raise RuntimeError("stream unavailable")

            def flush(self):
                raise RuntimeError("stream unavailable")

        real_import = __import__

        def without_rich(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("rich unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=without_rich):
            display = MODULE.ConsoleProgress(
                total=1,
                description="Broken",
                enabled=True,
                stream=BrokenStream(),
            )
            display.start()
            display.complete(status="done")
            display.stop()

        self.assertFalse(display.enabled)

    def test_console_progress_stops_when_tracked_loop_raises(self):
        class SpyProgress(MODULE.ConsoleProgress):
            def __init__(self):
                super().__init__(
                    total=2,
                    description="Spy",
                    enabled=False,
                    stream=io.StringIO(),
                )
                self.stop_count = 0

            def stop(self):
                self.stop_count += 1
                super().stop()

        display = SpyProgress()

        with self.assertRaisesRegex(RuntimeError, "stop now"):
            for _ in display.track(range(2)):
                raise RuntimeError("stop now")

        self.assertEqual(display.stop_count, 1)


class TestPrepare(unittest.TestCase):
    def test_prepare_records_category_duration_and_each_keyword(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "musan"
            write_wav(input_dir / "music" / "a.wav", 0.25)
            write_wav(input_dir / "noise" / "b.wav", 0.5)
            output = root / "generated" / "manifest.csv"

            return_code, _, stderr = quiet_main(
                [
                    "prepare",
                    "--input-dir",
                    str(input_dir),
                    "--output-manifest",
                    str(output),
                    "--keyword",
                    "HEY EVA",
                    "--keyword",
                    "bpe_ids:3 4",
                ]
            )

            self.assertEqual(return_code, 0, stderr)
            rows = read_csv(output)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["label"] for row in rows}, {"0"})
            self.assertEqual({row["keyword"] for row in rows}, {MODULE.DEVICE_KEYWORD})
            self.assertEqual(
                {tuple(json.loads(row["keywords"])) for row in rows},
                {("HEY EVA", "bpe_ids:3 4")},
            )
            self.assertEqual({row["category"] for row in rows}, {"music", "noise"})
            duration_by_category = {
                row["category"]: float(row["source_duration_sec"]) for row in rows
            }
            self.assertAlmostEqual(duration_by_category["music"], 0.25)
            self.assertAlmostEqual(duration_by_category["noise"], 0.5)
            for row in rows:
                audio = (output.parent / row["audio_path"]).resolve()
                self.assertTrue(audio.is_file())

            with self.assertRaises(MODULE.EvaluationError):
                MODULE._build_exposure(
                    output,
                    category_field="catgory",
                    duration_field="source_duration_sec",
                    assume_negative=False,
                )
            exposure, device_keywords = MODULE._build_exposure(
                output,
                category_field="",
                duration_field="source_duration_sec",
                assume_negative=False,
            )
            self.assertEqual(device_keywords, ["HEY EVA", "bpe_ids:3 4"])
            self.assertEqual(
                {row["category"] for row in exposure}, {MODULE.UNCATEGORIZED}
            )

    def test_prepare_single_keyword_keeps_readable_keyword_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "wavs"
            write_wav(input_dir / "a.wav", 0.2)
            output = root / "manifest.csv"

            return_code, _, stderr = quiet_main(
                [
                    "prepare",
                    "--input-dir",
                    str(input_dir),
                    "--output-manifest",
                    str(output),
                    "--keyword",
                    "HEY EVA",
                ]
            )

            self.assertEqual(return_code, 0, stderr)
            rows = read_csv(output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["keyword"], "HEY EVA")
            self.assertEqual(json.loads(rows[0]["keywords"]), ["HEY EVA"])

    def test_prepare_rejects_reserved_device_keyword(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "wavs"
            write_wav(input_dir / "a.wav", 0.2)
            output = root / "manifest.csv"

            return_code, _, stderr = quiet_main(
                [
                    "prepare",
                    "--input-dir",
                    str(input_dir),
                    "--output-manifest",
                    str(output),
                    "--keyword",
                    "DEVICE",
                ]
            )

            self.assertEqual(return_code, 2)
            self.assertIn("reserved", stderr)

    def test_build_exposure_rejects_duplicate_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "a.wav"
            write_wav(audio, 0.2)
            manifest = root / "manifest.csv"
            write_csv(
                manifest,
                ("audio_path", "keyword", "label"),
                [
                    {"audio_path": audio.name, "keyword": "ONE", "label": 0},
                    {"audio_path": audio.name, "keyword": "TWO", "label": 0},
                ],
            )
            with self.assertRaisesRegex(MODULE.EvaluationError, "duplicate audio"):
                MODULE._build_exposure(
                    manifest,
                    category_field="",
                    duration_field="source_duration_sec",
                    assume_negative=False,
                )

    def test_prepare_features_deduplicates_resolved_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"placeholder")
            manifest = root / "manifest.csv"
            write_csv(
                manifest,
                ("audio_path", "keyword", "label"),
                [
                    {"audio_path": audio.name, "keyword": "ONE", "label": 0},
                    {"audio_path": audio.name, "keyword": "TWO", "label": 0},
                ],
            )
            cache_dir = root / "features"
            with mock.patch.object(
                MODULE,
                "prepare_cached_fbank",
                return_value={"status": "computed"},
            ) as prepare_one:
                return_code, stdout, stderr = quiet_main(
                    [
                        "prepare-features",
                        "--manifest",
                        str(manifest),
                        "--feature-cache-dir",
                        str(cache_dir),
                        "--num-workers",
                        "1",
                        "--no-progress",
                    ]
                )

            self.assertEqual(return_code, 0, stderr)
            self.assertIn("1 unique files", stdout)
            prepare_one.assert_called_once_with(
                str(audio.resolve()), str(cache_dir), False
            )

    def test_prepare_features_rejects_nonpositive_worker_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            write_csv(manifest, ("audio_path", "keyword"), [])

            return_code, _, stderr = quiet_main(
                [
                    "prepare-features",
                    "--manifest",
                    str(manifest),
                    "--feature-cache-dir",
                    str(root / "features"),
                    "--num-workers",
                    "0",
                ]
            )

            self.assertEqual(return_code, 2)
            self.assertIn("--num-workers must be positive", stderr)


class TestOutputSafety(unittest.TestCase):
    def test_managed_outputs_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory).resolve()
            with self.assertRaises(MODULE.EvaluationError):
                MODULE._validate_managed_output_paths(
                    output_dir=output_dir,
                    managed_files=(output_dir / "run.json",),
                    managed_roots=(output_dir / "inference", output_dir / "report"),
                    protected_inputs=(output_dir / "run.json",),
                )


class TestReport(unittest.TestCase):
    def test_legacy_v1_exact_results_normalize_only_for_assume_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            legacy_v1_record = MODULE.build_evaluation_result(
                source_manifest_row=2,
                audio_path="noise.wav",
                audio_path_resolved="/dataset/noise.wav",
                keyword="WAKE",
                threshold=0.5,
                events=[{"score": 0.8}],
                duration_sec=10.0,
            )
            legacy_v1_record.update(
                {
                    "qbyt_score": 0.8,
                    "score_semantics": "max_emitted_keyword_acoustic_score",
                }
            )
            write_jsonl(path, [legacy_v1_record])
            exposure = {
                2: {
                    "source_manifest_row": 2,
                    "audio_path": "noise.wav",
                    "audio_path_resolved": "/dataset/noise.wav",
                    "keyword": "WAKE",
                    "label": "0",
                    "category": "noise",
                    "duration_sec": 10.0,
                }
            }

            with self.assertRaisesRegex(
                MODULE.EvaluationError, "negative label"
            ):
                MODULE._load_exact_results(
                    path,
                    thresholds=("0.5",),
                    exposure_by_row=exposure,
                )
            records, hits, semantics = MODULE._load_exact_results(
                path,
                thresholds=("0.5",),
                exposure_by_row=exposure,
                allow_missing_label=True,
            )

            self.assertEqual(records[0]["label"], 0)
            self.assertEqual(records[0]["false_alarm_events"], 1)
            self.assertNotIn("qbyt_score", records[0])
            self.assertNotIn("score_semantics", records[0])
            self.assertEqual(
                hits[("0.5", 2)],
                [MODULE.Hit(score=0.8, detected_keyword="WAKE")],
            )
            self.assertEqual(semantics, "exact_results_jsonl_v1")

    def test_exact_results_reject_mixed_v1_and_v2_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            records = [
                MODULE.build_evaluation_result(
                    source_manifest_row=2,
                    audio_path="noise.wav",
                    audio_path_resolved="/dataset/noise.wav",
                    keyword="WAKE",
                    label=0,
                    threshold=threshold,
                    events=[{"score": 0.8}],
                    duration_sec=10.0,
                )
                for threshold in (0.2, 0.8)
            ]
            records[0]["qbyt_score"] = 0.8
            write_jsonl(path, records)
            exposure = {
                2: {
                    "source_manifest_row": 2,
                    "audio_path": "noise.wav",
                    "audio_path_resolved": "/dataset/noise.wav",
                    "keyword": "WAKE",
                    "label": "0",
                    "category": "noise",
                    "duration_sec": 10.0,
                }
            }

            with self.assertRaisesRegex(MODULE.EvaluationError, "mix schema"):
                MODULE._load_exact_results(
                    path,
                    thresholds=("0.2", "0.8"),
                    exposure_by_row=exposure,
                )

    def test_report_preserves_zero_hits_and_counts_events_and_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            exposure_path = run_dir / "exposure.csv"
            event_path = run_dir / "inference" / "manifest.csv"
            write_csv(
                exposure_path,
                MODULE.EXPOSURE_FIELDS,
                [
                    {
                        "source_manifest_row": 2,
                        "audio_path": "music.wav",
                        "audio_path_resolved": "/dataset/music.wav",
                        "keyword": "WAKE",
                        "label": 0,
                        "category": "music",
                        "duration_sec": 1800,
                    },
                    {
                        "source_manifest_row": 3,
                        "audio_path": "noise.wav",
                        "audio_path_resolved": "/dataset/noise.wav",
                        "keyword": "WAKE",
                        "label": 0,
                        "category": "noise",
                        "duration_sec": 1800,
                    },
                ],
            )
            write_csv(
                event_path,
                ("source_manifest_row", "keywords_threshold", "score"),
                [
                    {
                        "source_manifest_row": 2,
                        "keywords_threshold": "0.2",
                        "score": "0.91",
                    },
                    {
                        "source_manifest_row": 2,
                        "keywords_threshold": "0.2",
                        "score": "0.88",
                    },
                    {
                        "source_manifest_row": 3,
                        "keywords_threshold": "0.2",
                        "score": "0.75",
                    },
                ],
            )
            inference_results_path = run_dir / "inference" / "results.jsonl"
            write_jsonl(
                inference_results_path,
                [
                    MODULE.build_evaluation_result(
                        source_manifest_row=2,
                        audio_path="music.wav",
                        audio_path_resolved="/dataset/music.wav",
                        keyword="WAKE",
                        label=0,
                        threshold=0.2,
                        events=[
                            {
                                "score": 0.91,
                                "timestamp_frames": [10, 11],
                                "clip_exported": False,
                                "raw_event_id": "music-0",
                            },
                            {
                                "score": 0.88,
                                "timestamp_frames": [20, 21],
                                "clip_exported": False,
                                "raw_event_id": "music-1",
                            },
                        ],
                        duration_sec=1800,
                        manifest_meta={"category": "music"},
                    ),
                    MODULE.build_evaluation_result(
                        source_manifest_row=2,
                        audio_path="music.wav",
                        audio_path_resolved="/dataset/music.wav",
                        keyword="WAKE",
                        label=0,
                        threshold=0.8,
                        duration_sec=1800,
                        manifest_meta={"category": "music"},
                    ),
                    MODULE.build_evaluation_result(
                        source_manifest_row=3,
                        audio_path="noise.wav",
                        audio_path_resolved="/dataset/noise.wav",
                        keyword="WAKE",
                        label=0,
                        threshold=0.2,
                        events=[
                            {
                                "score": 0.75,
                                "timestamp_frames": [30, 31],
                                "clip_exported": False,
                                "raw_event_id": "noise-0",
                            }
                        ],
                        duration_sec=1800,
                        manifest_meta={"category": "noise"},
                    ),
                    MODULE.build_evaluation_result(
                        source_manifest_row=3,
                        audio_path="noise.wav",
                        audio_path_resolved="/dataset/noise.wav",
                        keyword="WAKE",
                        label=0,
                        threshold=0.8,
                        duration_sec=1800,
                        manifest_meta={"category": "noise"},
                    ),
                ],
            )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "schema_version": MODULE.RUN_SCHEMA_VERSION,
                        "status": "inference_complete",
                        "manifest": "/dataset/manifest.csv",
                        "thresholds": ["0.2", "0.8"],
                        "exposure_csv": "exposure.csv",
                        "exposure_csv_sha256": MODULE._sha256(exposure_path),
                        "event_manifest": "inference/manifest.csv",
                        "event_manifest_sha256": MODULE._sha256(event_path),
                        "inference_results_jsonl": "inference/results.jsonl",
                        "inference_results_jsonl_sha256": MODULE._sha256(
                            inference_results_path
                        ),
                        "report_dir": "report",
                    }
                ),
                encoding="utf-8",
            )

            return_code, _, stderr = quiet_main(
                ["report", "--run-dir", str(run_dir), "--overwrite"]
            )

            self.assertEqual(return_code, 0, stderr)
            updated_run = json.loads(
                (run_dir / "run.json").read_text(encoding="utf-8")
            )
            with self.subTest("run_report persists rebuilt artifacts"):
                self.assertIn("artifacts", updated_run)
                self.assertEqual(
                    updated_run["artifacts"]["results_jsonl"], "results.jsonl"
                )
                self.assertEqual(
                    updated_run["artifacts"]["threshold_scan_csv"],
                    "threshold_scan.csv",
                )
                self.assertEqual(
                    updated_run["artifacts"]["report_html"], "report/report.html"
                )
            metrics = read_csv(run_dir / "report" / "threshold_metrics.csv")
            indexed = {
                (row["threshold"], row["keyword"], row["category"]): row
                for row in metrics
            }
            low_all = indexed[("0.2", "WAKE", "ALL")]
            self.assertEqual(int(low_all["false_alarm_events"]), 3)
            self.assertEqual(int(low_all["triggered_source_trials"]), 2)
            self.assertAlmostEqual(float(low_all["exposure_hours"]), 1.0)
            self.assertAlmostEqual(float(low_all["fa_per_hour"]), 3.0)
            self.assertAlmostEqual(float(low_all["source_trial_trigger_rate"]), 1.0)

            low_music = indexed[("0.2", "WAKE", "music")]
            self.assertEqual(int(low_music["false_alarm_events"]), 2)
            self.assertEqual(int(low_music["triggered_source_trials"]), 1)
            self.assertAlmostEqual(float(low_music["fa_per_hour"]), 4.0)
            low_noise = indexed[("0.2", "WAKE", "noise")]
            self.assertAlmostEqual(float(low_noise["fa_per_hour"]), 2.0)

            high_all = indexed[("0.8", "WAKE", "ALL")]
            self.assertEqual(int(high_all["false_alarm_events"]), 0)
            self.assertEqual(int(high_all["triggered_source_trials"]), 0)
            self.assertAlmostEqual(float(high_all["fa_per_hour"]), 0.0)
            self.assertAlmostEqual(float(high_all["source_trial_trigger_rate"]), 0.0)
            source_rows = read_csv(run_dir / "report" / "source_metrics.csv")
            high_source_rows = [row for row in source_rows if row["threshold"] == "0.8"]
            self.assertEqual(len(high_source_rows), 2)
            self.assertEqual(
                {row["false_alarm_events"] for row in high_source_rows}, {"0"}
            )

            summary = json.loads(
                (run_dir / "report" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["event_count"], 3)
            self.assertEqual(summary["source_trials"], 2)
            self.assertEqual(
                summary["semantics"]["observation_source"],
                "exact_results_jsonl_v2",
            )
            for filename in (
                "results.jsonl",
                "summary.json",
                "threshold_scan_summary.json",
                "threshold_scan.csv",
                "threshold_scan.png",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)
            standard_results = [
                json.loads(line)
                for line in (run_dir / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(standard_results), 4)
            self.assertTrue(
                all(
                    "qbyt_score" not in row and "score_semantics" not in row
                    for row in standard_results
                )
            )
            low_music_result = next(
                row
                for row in standard_results
                if row["source_manifest_row"] == 2 and row["threshold"] == 0.2
            )
            self.assertEqual(low_music_result["detection_count"], 2)
            with self.subTest("report preserves raw inference events"):
                self.assertEqual(
                    [
                        event["raw_event_id"]
                        for event in low_music_result["detections"]
                    ],
                    ["music-0", "music-1"],
                )
                self.assertEqual(
                    [
                        event["timestamp_frames"]
                        for event in low_music_result["detections"]
                    ],
                    [[10, 11], [20, 21]],
                )
                self.assertFalse(
                    any(
                        event["clip_exported"]
                        for event in low_music_result["detections"]
                    )
                )
            scan_rows = read_csv(run_dir / "threshold_scan.csv")
            self.assertEqual([row["threshold"] for row in scan_rows], ["0.8", "0.2"])
            self.assertEqual(int(scan_rows[1]["fp"]), 2)
            self.assertAlmostEqual(float(scan_rows[1]["fa_per_hour"]), 3.0)
            self.assertIn("category_music_fp", scan_rows[0])
            self.assertIn("category_noise_fp", scan_rows[0])
            self.assertFalse(
                any(name.startswith("subset_") for name in scan_rows[0])
            )
            standard_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            scan_summary = json.loads(
                (run_dir / "threshold_scan_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(standard_summary["schema_version"], 2)
            self.assertEqual(standard_summary["evaluation_type"], "negative")
            self.assertEqual(scan_summary["schema_version"], 2)
            self.assertEqual(scan_summary["evaluation_type"], "negative")
            self.assertEqual(
                {value["name"] for value in scan_summary["categories"].values()},
                {"music", "noise"},
            )
            for removed in (
                "mode",
                "source_summary",
                "score_min",
                "score_max",
                "threshold_step",
                "workers",
                "subsets",
            ):
                self.assertNotIn(removed, scan_summary)
            self.assertEqual(
                (run_dir / "threshold_scan.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            svg_paths = sorted((run_dir / "report").glob("*.svg"))
            self.assertEqual(len(svg_paths), 2)
            for path in svg_paths:
                self.assertEqual(ET.parse(path).getroot().tag.split("}")[-1], "svg")
            report_html = (run_dir / "report" / "report.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("KWS negative-set evaluation", report_html)
            self.assertIn("WAKE", report_html)
            for chart in summary["charts"]:
                self.assertIn(chart["path"], report_html)

            exposure_rows = read_csv(exposure_path)
            exposure_rows[0]["duration_sec"] = "1"
            write_csv(exposure_path, MODULE.EXPOSURE_FIELDS, exposure_rows)
            return_code, _, stderr = quiet_main(
                ["report", "--run-dir", str(run_dir), "--overwrite"]
            )
            self.assertEqual(return_code, 2)
            self.assertIn("exposure CSV content differs", stderr)


class TestDeviceKeywordMetrics(unittest.TestCase):
    def test_threshold_metrics_split_detected_keyword_with_shared_exposure(self):
        source_rows = [
            {
                "threshold": "0.2",
                "source_manifest_row": 2,
                "audio_path": "a.wav",
                "audio_path_resolved": "/data/a.wav",
                "keyword": MODULE.DEVICE_KEYWORD,
                "category": "noise",
                "duration_sec": "3600",
                "false_alarm_events": 2,
                "triggered": 1,
                "max_score": "0.9",
            },
            {
                "threshold": "0.2",
                "source_manifest_row": 3,
                "audio_path": "b.wav",
                "audio_path_resolved": "/data/b.wav",
                "keyword": MODULE.DEVICE_KEYWORD,
                "category": "noise",
                "duration_sec": "3600",
                "false_alarm_events": 1,
                "triggered": 1,
                "max_score": "0.8",
            },
        ]
        hits = {
            ("0.2", 2): [
                MODULE.Hit(score=0.9, detected_keyword="HEY EVA"),
                MODULE.Hit(score=0.7, detected_keyword="OK GOOGLE"),
            ],
            ("0.2", 3): [MODULE.Hit(score=0.8, detected_keyword="HEY EVA")],
        }
        rows = MODULE._threshold_metrics(
            ["0.2"],
            source_rows,
            hits=hits,
            device_keywords=["HEY EVA", "OK GOOGLE"],
        )
        indexed = {
            (row["keyword"], row["category"]): row for row in rows
        }
        device_all = indexed[(MODULE.DEVICE_KEYWORD, MODULE.ALL_CATEGORY)]
        self.assertEqual(int(device_all["false_alarm_events"]), 3)
        self.assertEqual(int(device_all["triggered_source_trials"]), 2)
        self.assertAlmostEqual(float(device_all["exposure_hours"]), 2.0)
        self.assertAlmostEqual(float(device_all["fa_per_hour"]), 1.5)

        hey = indexed[("HEY EVA", MODULE.ALL_CATEGORY)]
        ok = indexed[("OK GOOGLE", MODULE.ALL_CATEGORY)]
        self.assertEqual(int(hey["false_alarm_events"]), 2)
        self.assertEqual(int(ok["false_alarm_events"]), 1)
        self.assertAlmostEqual(float(hey["exposure_hours"]), 2.0)
        self.assertAlmostEqual(float(ok["exposure_hours"]), 2.0)
        self.assertAlmostEqual(
            float(hey["fa_per_hour"]) + float(ok["fa_per_hour"]),
            float(device_all["fa_per_hour"]),
        )
        self.assertEqual(int(hey["triggered_source_trials"]), 2)
        self.assertEqual(int(ok["triggered_source_trials"]), 1)

    def test_threshold_metrics_map_text_and_bpe_specs_to_detected_phrases(self):
        source_rows = [
            {
                "threshold": "0.2",
                "source_manifest_row": 2,
                "audio_path": "a.wav",
                "audio_path_resolved": "/data/a.wav",
                "keyword": MODULE.DEVICE_KEYWORD,
                "category": "noise",
                "duration_sec": "3600",
                "false_alarm_events": 2,
                "triggered": 1,
                "max_score": "0.9",
            }
        ]
        hits = {
            ("0.2", 2): [
                MODULE.Hit(score=0.9, detected_keyword="HEY EVA"),
                MODULE.Hit(score=0.7, detected_keyword="[ALARM]"),
            ]
        }
        rows = MODULE._threshold_metrics(
            ["0.2"],
            source_rows,
            hits=hits,
            device_keywords=["hey eva", "bpe_ids:3 4"],
            keyword_phrases={"hey eva": "HEY EVA", "bpe_ids:3 4": "[ALARM]"},
        )
        indexed = {
            (row["keyword"], row["category"]): row
            for row in rows
            if row["category"] == MODULE.ALL_CATEGORY
        }
        self.assertEqual(int(indexed[("HEY EVA", MODULE.ALL_CATEGORY)]["false_alarm_events"]), 1)
        self.assertEqual(
            int(indexed[("[ALARM]", MODULE.ALL_CATEGORY)]["false_alarm_events"]), 1
        )
        self.assertNotIn(("hey eva", MODULE.ALL_CATEGORY), indexed)
        self.assertNotIn(("bpe_ids:3 4", MODULE.ALL_CATEGORY), indexed)

    def test_threshold_metrics_uppercase_text_specs_without_phrase_map(self):
        source_rows = [
            {
                "threshold": "0.2",
                "source_manifest_row": 2,
                "audio_path": "a.wav",
                "audio_path_resolved": "/data/a.wav",
                "keyword": MODULE.DEVICE_KEYWORD,
                "category": "noise",
                "duration_sec": "3600",
                "false_alarm_events": 1,
                "triggered": 1,
                "max_score": "0.9",
            }
        ]
        hits = {
            ("0.2", 2): [MODULE.Hit(score=0.9, detected_keyword="HEY EVA")],
        }
        rows = MODULE._threshold_metrics(
            ["0.2"],
            source_rows,
            hits=hits,
            device_keywords=["hey eva", "OK GOOGLE"],
        )
        indexed = {
            (row["keyword"], row["category"]): row
            for row in rows
            if row["category"] == MODULE.ALL_CATEGORY
        }
        self.assertEqual(int(indexed[("HEY EVA", MODULE.ALL_CATEGORY)]["false_alarm_events"]), 1)
        self.assertEqual(
            int(indexed[("OK GOOGLE", MODULE.ALL_CATEGORY)]["false_alarm_events"]), 0
        )
        self.assertNotIn(("hey eva", MODULE.ALL_CATEGORY), indexed)

    def test_resume_device_keywords_rejects_unrecoverable_device_rows(self):
        with self.assertRaisesRegex(MODULE.EvaluationError, "cannot reconstruct"):
            MODULE._resume_device_keywords(
                {},
                [{"keyword": MODULE.DEVICE_KEYWORD}],
            )


class TestSweep(unittest.TestCase):
    def test_sweep_invokes_decoder_once_and_resume_checks_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            write_wav(audio, 1.0)
            manifest = root / "manifest.csv"
            write_csv(
                manifest,
                (
                    "audio_path",
                    "keyword",
                    "label",
                    "category",
                    "source_duration_sec",
                ),
                [
                    {
                        "audio_path": audio.name,
                        "keyword": "WAKE",
                        "label": 0,
                        "category": "noise",
                        "source_duration_sec": 1,
                    }
                ],
            )
            fake_decoder = root / "fake_decoder.py"
            fake_decoder.write_text(
                """
import csv
import json
import sys
from pathlib import Path

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

counter = Path(value("--counter"))
count = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
counter.write_text(str(count + 1), encoding="utf-8")
Path(value("--argv-log")).write_text(json.dumps(args), encoding="utf-8")
output = Path(value("--output-manifest"))
output.parent.mkdir(parents=True, exist_ok=True)
thresholds = value("--keywords-thresholds").split(",")
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("source_manifest_row", "keywords_threshold", "score"),
    )
    writer.writeheader()
    for threshold in thresholds:
        writer.writerow(
            {
                "source_manifest_row": 2,
                "keywords_threshold": threshold,
                "score": 0.9,
            }
        )
""".lstrip(),
                encoding="utf-8",
            )
            output_dir = root / "run"
            counter = root / "calls.txt"
            argv_log = root / "argv.json"
            checkpoint = root / "model.pt"
            bpe_model = root / "bpe.model"
            feature_cache = root / "features"
            checkpoint.write_bytes(b"model")
            bpe_model.write_bytes(b"bpe")
            feature_cache.mkdir()

            def sweep_args(existing_flag=None):
                values = [
                    "sweep",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(output_dir),
                    "--thresholds",
                    "0.2:0.4:0.2",
                    "--decode-script",
                    str(fake_decoder),
                    "--python",
                    sys.executable,
                ]
                if existing_flag is not None:
                    values.append(existing_flag)
                values.extend(
                    [
                        "--",
                        "--counter",
                        str(counter),
                        "--argv-log",
                        str(argv_log),
                        "--checkpoint",
                        str(checkpoint),
                        "--bpe-model",
                        str(bpe_model),
                        "--max-token-gap-sec",
                        "0.8",
                        "--max-keyword-duration-sec",
                        "2.0",
                        "--feature-cache-dir",
                        str(feature_cache),
                        "--stream-batch-size",
                        "8",
                    ]
                )
                return values

            return_code, _, stderr = quiet_main(sweep_args())
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
            decoder_argv = json.loads(argv_log.read_text(encoding="utf-8"))
            threshold_index = decoder_argv.index("--keywords-thresholds")
            self.assertEqual(decoder_argv[threshold_index + 1], "0.2,0.4")
            gap_index = decoder_argv.index("--max-token-gap-sec")
            self.assertEqual(decoder_argv[gap_index + 1], "0.8")
            duration_index = decoder_argv.index("--max-keyword-duration-sec")
            self.assertEqual(decoder_argv[duration_index + 1], "2.0")
            cache_index = decoder_argv.index("--feature-cache-dir")
            self.assertEqual(decoder_argv[cache_index + 1], str(feature_cache))
            batch_index = decoder_argv.index("--stream-batch-size")
            self.assertEqual(decoder_argv[batch_index + 1], "8")
            keyword_index = decoder_argv.index("--keywords")
            self.assertEqual(decoder_argv[keyword_index + 1], "WAKE")
            event_rows = read_csv(output_dir / "inference" / "manifest.csv")
            self.assertEqual(
                {row["keywords_threshold"] for row in event_rows}, {"0.2", "0.4"}
            )
            run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "complete")
            self.assertEqual(run["decoder_return_code"], 0)
            for filename in (
                "results.jsonl",
                "summary.json",
                "threshold_scan_summary.json",
                "threshold_scan.csv",
                "threshold_scan.png",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)
            standard_results = [
                json.loads(line)
                for line in (output_dir / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(
                all(
                    "qbyt_score" not in row and "score_semantics" not in row
                    for row in standard_results
                )
            )
            standard_summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            scan_summary = json.loads(
                (output_dir / "threshold_scan_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(standard_summary["schema_version"], 2)
            self.assertEqual(standard_summary["evaluation_type"], "negative")
            self.assertEqual(scan_summary["schema_version"], 2)
            self.assertEqual(scan_summary["evaluation_type"], "negative")

            resume_args = sweep_args("--resume")
            resume_args.insert(resume_args.index("--"), "--no-progress")
            return_code, _, stderr = quiet_main(resume_args)
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

            run_path = output_dir / "run.json"
            run_payload = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(run_payload["device_keywords"], ["WAKE"])
            del run_payload["device_keywords"]
            run_path.write_text(
                json.dumps(run_payload, ensure_ascii=False), encoding="utf-8"
            )
            return_code, _, stderr = quiet_main(resume_args)
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
            restored = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(restored["device_keywords"], ["WAKE"])

            event_manifest = output_dir / "inference" / "manifest.csv"
            original_events = event_manifest.read_text(encoding="utf-8")
            event_manifest.write_text(
                "source_manifest_row,keywords_threshold,score\n",
                encoding="utf-8",
            )
            return_code, _, stderr = quiet_main(sweep_args("--resume"))
            self.assertEqual(return_code, 2)
            self.assertIn("event manifest content differs", stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
            event_manifest.write_text(original_events, encoding="utf-8")

            with manifest.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            return_code, _, stderr = quiet_main(sweep_args("--resume"))
            self.assertEqual(return_code, 2)
            self.assertIn("manifest_sha256", stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")


if __name__ == "__main__":
    unittest.main()
