#!/usr/bin/env python3

import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from egs.gigaspeech.KWS.zipformer import kws_eval_artifacts as MODULE


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestEvaluationArtifacts(unittest.TestCase):
    def _record(
        self,
        *,
        source_row,
        threshold,
        label,
        scores=(),
        category="noise",
        duration_sec=1800.0,
    ):
        return MODULE.build_result_record(
            source_manifest_row=source_row,
            audio_path="audio-{}.wav".format(source_row),
            audio_path_resolved="/dataset/audio-{}.wav".format(source_row),
            keyword="WAKE",
            label=label,
            threshold=threshold,
            events=[{"score": score} for score in scores],
            duration_sec=duration_sec,
            manifest_meta={"category": category},
        )

    def test_negative_exact_scan_counts_trials_and_all_events(self):
        records = [
            self._record(source_row=2, threshold=0.2, label=0, scores=(0.9, 0.8)),
            self._record(source_row=2, threshold=0.8, label=0),
            self._record(
                source_row=3,
                threshold=0.2,
                label=0,
                scores=(0.7,),
                category="music",
            ),
            self._record(
                source_row=3,
                threshold=0.8,
                label=0,
                category="music",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            paths = MODULE.write_evaluation_artifacts(
                output_dir=output_dir,
                records=records,
                thresholds=(0.2, 0.8),
                manifest_path=Path("/dataset/manifest.csv"),
                overwrite=False,
                mode="musan",
            )

            self.assertEqual(set(paths), set(MODULE.ARTIFACT_FILENAMES))
            for name in MODULE.ARTIFACT_FILENAMES:
                self.assertTrue((output_dir / name).is_file(), name)
            self.assertEqual(
                (output_dir / "threshold_scan.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )

            result_rows = [
                json.loads(line)
                for line in (output_dir / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(result_rows), 4)
            low_first = next(
                row
                for row in result_rows
                if row["source_manifest_row"] == 2 and row["threshold"] == 0.2
            )
            self.assertEqual(low_first["detection_count"], 2)
            self.assertEqual(low_first["qbyt_score"], 0.9)
            self.assertTrue(low_first["detected"])

            curve = read_csv(output_dir / "threshold_scan.csv")
            self.assertEqual([row["threshold"] for row in curve], ["0.8", "0.2"])
            self.assertEqual(int(curve[0]["fp"]), 0)
            self.assertEqual(int(curve[1]["fp"]), 2)
            self.assertAlmostEqual(float(curve[1]["fpr"]), 1.0)
            # Three events over one source-trial exposure hour.
            self.assertAlmostEqual(float(curve[1]["fa_per_hour"]), 3.0)

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["num_samples"], 2)
            self.assertEqual(summary["num_result_rows"], 4)
            self.assertAlmostEqual(summary["total_hours"], 1.0)
            scan = json.loads(
                (output_dir / "threshold_scan_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(scan["mode"], "musan")
            self.assertEqual(scan["plot"]["metrics"], ["fpr"])
            self.assertEqual(
                {value["name"] for value in scan["subsets"].values()},
                {"music", "noise"},
            )

    def test_generation_failure_does_not_replace_existing_artifact_set(self):
        record = self._record(
            source_row=2,
            threshold=0.5,
            label=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            output_dir.mkdir()
            before = {}
            for index, name in enumerate(MODULE.ARTIFACT_FILENAMES):
                payload = "old-{}-{}".format(index, name).encode("utf-8")
                (output_dir / name).write_bytes(payload)
                before[name] = payload

            with mock.patch.object(
                MODULE,
                "_write_threshold_plot",
                side_effect=RuntimeError("plot failed"),
            ), self.assertRaisesRegex(RuntimeError, "plot failed"):
                MODULE.write_evaluation_artifacts(
                    output_dir=output_dir,
                    records=[record],
                    thresholds=(0.5,),
                    manifest_path=Path("/dataset/manifest.csv"),
                    overwrite=True,
                    mode="musan",
                )

            self.assertEqual(
                {
                    name: (output_dir / name).read_bytes()
                    for name in MODULE.ARTIFACT_FILENAMES
                },
                before,
            )

    def test_writer_rejects_internally_inconsistent_result_records(self):
        valid = self._record(
            source_row=2,
            threshold=0.5,
            label=0,
            scores=(0.8,),
        )
        cases = (
            ("count", lambda row: row.update(detection_count=0)),
            ("score", lambda row: row.update(qbyt_score=0.1)),
            ("events", lambda row: row.update(false_alarm_events=0)),
            ("row", lambda row: row.update(source_manifest_row=True)),
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                record = copy.deepcopy(valid)
                mutate(record)
                with self.assertRaises(MODULE.ArtifactError):
                    MODULE.write_evaluation_artifacts(
                        output_dir=Path(directory) / "output",
                        records=[record],
                        thresholds=(0.5,),
                        manifest_path=Path("/dataset/manifest.csv"),
                        overwrite=False,
                        mode="musan",
                    )

    def test_writer_rejects_empty_result_set(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            MODULE.ArtifactError, "at least one evaluation result"
        ):
            MODULE.write_evaluation_artifacts(
                output_dir=Path(directory) / "output",
                records=[],
                thresholds=(0.5,),
                manifest_path=Path("/dataset/manifest.csv"),
                overwrite=False,
            )

    def test_mixed_labels_use_reference_confusion_fields(self):
        records = [
            self._record(source_row=2, threshold=0.2, label=1, scores=(0.9,)),
            self._record(source_row=2, threshold=0.8, label=1, scores=(0.9,)),
            self._record(source_row=3, threshold=0.2, label=0, scores=(0.7,)),
            self._record(source_row=3, threshold=0.8, label=0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MODULE.write_evaluation_artifacts(
                output_dir=output_dir,
                records=records,
                thresholds=(0.2, 0.8),
                manifest_path=Path("/dataset/manifest.csv"),
                overwrite=False,
                mode="clips",
            )
            curve = read_csv(output_dir / "threshold_scan.csv")
            self.assertEqual(
                tuple(curve[0]),
                MODULE.CLIP_SCAN_FIELDS + MODULE.CLIP_SCAN_EXTENSION_FIELDS,
            )
            high, low = curve
            self.assertEqual(
                tuple(int(high[name]) for name in ("tp", "tn", "fp", "fn")),
                (1, 1, 0, 0),
            )
            self.assertEqual(
                tuple(int(low[name]) for name in ("tp", "tn", "fp", "fn")),
                (1, 0, 1, 0),
            )
            scan = json.loads(
                (output_dir / "threshold_scan_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(scan["best_f1"]["threshold"], 0.8)
            self.assertEqual(scan["plot"]["metrics"], ["recall", "fpr"])
            for misleading_name in ("auc", "eer", "eer_threshold"):
                self.assertNotIn(misleading_name, scan)
            self.assertAlmostEqual(scan["sampled_auc"], 1.0)
            self.assertAlmostEqual(scan["sampled_eer"], 0.0)
            self.assertEqual(scan["sampled_eer_threshold"], 0.8)

    def test_unlabeled_single_threshold_summary_uses_detection_metrics(self):
        record = self._record(
            source_row=2,
            threshold=0.5,
            label=None,
            scores=(0.9,),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MODULE.write_evaluation_artifacts(
                output_dir=output_dir,
                records=[record],
                thresholds=(0.5,),
                manifest_path=Path("/dataset/manifest.csv"),
                overwrite=False,
                mode="clips",
            )

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("metrics", summary)
            self.assertNotIn("total_hours", summary)
            self.assertIn("detection_metrics", summary)
            self.assertAlmostEqual(
                summary["detection_metrics"]["detection_rate"], 1.0
            )

    def test_existing_artifact_is_rejected_before_any_write(self):
        record = self._record(source_row=2, threshold=0.5, label=0)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            existing = output_dir / "threshold_scan.csv"
            existing.write_text("owned\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                MODULE.write_evaluation_artifacts(
                    output_dir=output_dir,
                    records=[record],
                    thresholds=(0.5,),
                    manifest_path=Path("/dataset/manifest.csv"),
                    overwrite=False,
                )
            self.assertEqual(existing.read_text(encoding="utf-8"), "owned\n")
            self.assertFalse((output_dir / "results.jsonl").exists())

    def test_every_trial_requires_a_consistent_threshold_cartesian_product(self):
        low = self._record(source_row=2, threshold=0.2, label=0)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(MODULE.ArtifactError, "missing threshold"):
                MODULE.write_evaluation_artifacts(
                    output_dir=Path(directory),
                    records=[low],
                    thresholds=(0.2, 0.8),
                    manifest_path=Path("/dataset/manifest.csv"),
                    overwrite=False,
                )

        high = self._record(source_row=2, threshold=0.8, label=1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(MODULE.ArtifactError, "changes trial field"):
                MODULE.write_evaluation_artifacts(
                    output_dir=Path(directory),
                    records=[low, high],
                    thresholds=(0.2, 0.8),
                    manifest_path=Path("/dataset/manifest.csv"),
                    overwrite=False,
                )

    def test_auto_mode_does_not_drop_unlabeled_or_failed_positive_trials(self):
        negative = self._record(source_row=2, threshold=0.5, label=0)
        unlabeled = MODULE.build_result_record(
            source_manifest_row=3,
            audio_path="unlabeled.wav",
            keyword="WAKE",
            threshold=0.5,
            duration_sec=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MODULE.write_evaluation_artifacts(
                output_dir=output_dir,
                records=[negative, unlabeled],
                thresholds=(0.5,),
                manifest_path=Path("/dataset/manifest.csv"),
                overwrite=False,
                mode="auto",
            )
            summary = json.loads(
                (output_dir / "threshold_scan_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["mode"], "clips")
            self.assertEqual(summary["num_unlabeled_excluded"], 1)

        failed_positive = MODULE.build_result_record(
            source_manifest_row=3,
            audio_path="positive.wav",
            keyword="WAKE",
            label=1,
            threshold=0.5,
            duration_sec=1.0,
            skipped=True,
            error="decode failed",
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MODULE.write_evaluation_artifacts(
                output_dir=output_dir,
                records=[negative, failed_positive],
                thresholds=(0.5,),
                manifest_path=Path("/dataset/manifest.csv"),
                overwrite=False,
                mode="auto",
            )
            summary = json.loads(
                (output_dir / "threshold_scan_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["mode"], "clips")
            self.assertEqual(summary["num_skipped_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
