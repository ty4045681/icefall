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
        for value in ("--man", "--output-m", "--overw", "--keywords-t"):
            with self.subTest(value=value):
                with self.assertRaises(MODULE.EvaluationError):
                    MODULE._validate_decoder_args([value, "ignored"])
        self.assertEqual(
            MODULE._validate_decoder_args(["--model-config", "model.json"]),
            ["--model-config", "model.json"],
        )


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
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["label"] for row in rows}, {"0"})
            self.assertEqual(
                {row["keyword"] for row in rows},
                {"HEY EVA", "bpe_ids:3 4"},
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
            exposure = MODULE._build_exposure(
                output,
                category_field="",
                duration_field="source_duration_sec",
                assume_negative=False,
            )
            self.assertEqual(
                {row["category"] for row in exposure}, {MODULE.UNCATEGORIZED}
            )


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
                        "report_dir": "report",
                    }
                ),
                encoding="utf-8",
            )

            return_code, _, stderr = quiet_main(
                ["report", "--run-dir", str(run_dir), "--overwrite"]
            )

            self.assertEqual(return_code, 0, stderr)
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
            checkpoint.write_bytes(b"model")
            bpe_model.write_bytes(b"bpe")

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
                    ]
                )
                return values

            return_code, _, stderr = quiet_main(sweep_args())
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
            decoder_argv = json.loads(argv_log.read_text(encoding="utf-8"))
            threshold_index = decoder_argv.index("--keywords-thresholds")
            self.assertEqual(decoder_argv[threshold_index + 1], "0.2,0.4")
            event_rows = read_csv(output_dir / "inference" / "manifest.csv")
            self.assertEqual(
                {row["keywords_threshold"] for row in event_rows}, {"0.2", "0.4"}
            )
            run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "complete")
            self.assertEqual(run["decoder_return_code"], 0)

            return_code, _, stderr = quiet_main(sweep_args("--resume"))
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

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
