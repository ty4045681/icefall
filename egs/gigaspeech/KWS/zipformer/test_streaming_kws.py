#!/usr/bin/env python3

import contextlib
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).with_name("streaming_kws.py")
SPEC = importlib.util.spec_from_file_location("streaming_kws", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CONTEXT_GRAPH_SCRIPT = SCRIPT.parents[4] / "icefall" / "context_graph.py"
CONTEXT_GRAPH_SPEC = importlib.util.spec_from_file_location(
    "streaming_kws_test_context_graph", CONTEXT_GRAPH_SCRIPT
)
CONTEXT_GRAPH_MODULE = importlib.util.module_from_spec(CONTEXT_GRAPH_SPEC)
assert CONTEXT_GRAPH_SPEC.loader is not None
CONTEXT_GRAPH_SPEC.loader.exec_module(CONTEXT_GRAPH_MODULE)


class FakeSentencePiece:
    def __init__(self):
        # Matches this recipe's train_bpe_model.py special-token layout.
        self.pieces = [
            "<blk>",
            "<sos/eos>",
            "<unk>",
            "\u2581HEY",
            "\u2581EVA",
            "\u2581[ALARM]",
        ]
        self.encode_calls = []

    def unk_id(self):
        return 2

    def is_unknown(self, token_id):
        return token_id == self.unk_id()

    def get_piece_size(self):
        return len(self.pieces)

    def piece_to_id(self, piece):
        try:
            return self.pieces.index(piece)
        except ValueError:
            return self.unk_id()

    def id_to_piece(self, token_id):
        return self.pieces[token_id]

    def encode(self, phrase):
        self.encode_calls.append(phrase)
        values = {"HEY EVA": [3, 4], "[ALARM]": [5]}
        return values.get(phrase, [self.unk_id()])

    def decode(self, token_ids):
        return (
            "".join(self.pieces[token_id] for token_id in token_ids)
            .replace("\u2581", " ")
            .strip()
        )


class TestManifestAndTimestamps(unittest.TestCase):
    def test_keyword_timing_policy_uses_inclusive_duration_and_exact_gap(self):
        policy = MODULE.KeywordTimingPolicy(
            max_token_gap_sec=0.08,
            max_keyword_duration_sec=0.12,
        )

        self.assertTrue(policy.can_continue([0, 1], current_frame=2))
        self.assertTrue(policy.accepts([0, 2]))
        self.assertFalse(policy.can_continue([0, 1], current_frame=3))
        self.assertFalse(policy.accepts([0, 3]))

    def test_keyword_timing_values_must_be_finite_and_non_negative(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    MODULE._resolve_optional_positive_seconds(
                        value, {}, "max_token_gap_sec", "--max-token-gap-sec"
                    )

        for cli_value, config, expected in (
            (None, {}, None),
            (None, {"max_token_gap_sec": "0.8"}, 0.8),
            (0.4, {"max_token_gap_sec": "0.8"}, 0.4),
            (0.0, {"max_token_gap_sec": "0.8"}, None),
            (None, {"max_token_gap_sec": 0}, None),
        ):
            with self.subTest(cli_value=cli_value, config=config):
                self.assertEqual(
                    MODULE._resolve_optional_positive_seconds(
                        cli_value,
                        config,
                        "max_token_gap_sec",
                        "--max-token-gap-sec",
                    ),
                    expected,
                )

    def test_internal_keyword_timing_policy_rejects_zero(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            MODULE.KeywordTimingPolicy(max_token_gap_sec=0.0)
        with self.assertRaisesRegex(ValueError, "at least one encoder frame"):
            MODULE.KeywordTimingPolicy(max_keyword_duration_sec=0.02)

    def test_keyword_timing_config_string_is_resolved(self):
        self.assertEqual(
            MODULE._resolve_optional_positive_seconds(
                None,
                {"max_token_gap_sec": "0.8"},
                "max_token_gap_sec",
                "--max-token-gap-sec",
            ),
            0.8,
        )

    def test_keyword_threshold_list_is_sorted_and_numerically_deduplicated(self):
        self.assertEqual(
            MODULE.parse_keywords_thresholds(
                "0.8, 0.25 0.80,0.3,0.30000000000000004,0,1"
            ),
            [0.0, 0.25, 0.3, 0.8, 1.0],
        )

    def test_keyword_threshold_list_rejects_empty_invalid_and_out_of_range(self):
        for value in ("", "not-a-number", "nan", "inf", "-0.01", "1.01"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MODULE.parse_keywords_thresholds(value)

    def test_manifest_paths_are_relative_to_csv_and_metadata_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio" / "sample.wav"
            audio.parent.mkdir()
            audio.touch()
            manifest = root / "manifest.csv"
            manifest.write_text(
                "\ufeffaudio_path,keyword,label,text_variant\n"
                "audio/sample.wav,hey eva,1,hey ever\n",
                encoding="utf-8",
            )

            entries, fields = MODULE.load_manifest(manifest)

            self.assertEqual(fields, ["audio_path", "keyword", "label", "text_variant"])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].audio_path, audio.resolve())
            self.assertEqual(entries[0].keyword, "hey eva")
            self.assertEqual(entries[0].label, 1)
            self.assertEqual(entries[0].row["text_variant"], "hey ever")

    def test_manifest_rejects_non_binary_label(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                "audio_path,keyword,label\na.wav,hey eva,2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "label must be 0 or 1"):
                MODULE.load_manifest(manifest)

    def test_manifest_rejects_empty_trial_set(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text("audio_path,keyword,label\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contains no trials"):
                MODULE.load_manifest(manifest)

    def test_manifest_trims_header_whitespace_like_dma_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.wav").touch()
            manifest = root / "manifest.csv"
            manifest.write_text(
                " audio_path , keyword , label \na.wav,hey eva,0\n",
                encoding="utf-8",
            )
            entries, fields = MODULE.load_manifest(manifest)
            self.assertEqual(fields, ["audio_path", "keyword", "label"])
            self.assertEqual(entries[0].label, 0)

    def test_manifest_retains_csv_quoted_bpe_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.wav").touch()
            manifest = root / "manifest.csv"
            manifest.write_text(
                'audio_path,keyword,label\na.wav,"bpe_ids:[3,4]",1\n',
                encoding="utf-8",
            )

            entries, _ = MODULE.load_manifest(manifest)

            self.assertEqual(entries[0].keyword, "bpe_ids:[3,4]")

    def test_frame_span_uses_last_frame_plus_one_and_exact_samples(self):
        detection = MODULE.Detection(
            phrase="HEY EVA", timestamp_frames=[25, 40], score=0.9
        )

        bounds = MODULE.compute_clip_bounds(
            detection,
            num_samples=40000,
            sample_rate=16000,
            pre_roll_sec=0.15,
            post_roll_sec=0.20,
        )

        self.assertIsNotNone(bounds)
        self.assertEqual(bounds.raw_start_sample, 16000)
        self.assertEqual(bounds.raw_end_sample, 26240)
        self.assertEqual(bounds.start_sample, 13600)
        self.assertEqual(bounds.end_sample, 29440)
        self.assertAlmostEqual(bounds.raw_start_sec, 1.0)
        self.assertAlmostEqual(bounds.raw_end_sec, 1.64)

    def test_frame_span_clamps_to_audio(self):
        detection = MODULE.Detection(phrase="WAKE", timestamp_frames=[0], score=1.0)
        bounds = MODULE.compute_clip_bounds(
            detection,
            num_samples=400,
            pre_roll_sec=1.0,
            post_roll_sec=1.0,
        )
        self.assertEqual((bounds.start_sample, bounds.end_sample), (0, 400))

    def test_hit_emitted_in_synthetic_tail_is_anchored_to_real_eof(self):
        detection = MODULE.Detection(
            phrase="WAKE", timestamp_frames=[10, 11], score=0.9
        )
        bounds = MODULE.compute_clip_bounds(
            detection,
            num_samples=6400,
            pre_roll_sec=0.0,
            post_roll_sec=0.0,
        )

        self.assertIsNotNone(bounds)
        self.assertTrue(bounds.tail_anchored)
        self.assertEqual(bounds.raw_start_sample, 5120)
        self.assertEqual(bounds.raw_end_sample, 6400)
        self.assertEqual(bounds.end_sample, 6400)
        payload = MODULE.detection_json(
            detection,
            duration_samples=6400,
            sample_rate=16000,
            decode_mode="streaming",
        )
        self.assertEqual(payload["start_time"], 0.32)
        self.assertEqual(payload["end_time"], 0.4)

    def test_dma_json_uses_seconds_and_never_ambiguous_timestamps(self):
        detection = MODULE.Detection(
            phrase="HEY EVA", timestamp_frames=[25, 40], score=0.75
        )
        payload = MODULE.detection_json(
            detection,
            duration_samples=40000,
            sample_rate=16000,
            decode_mode="streaming",
        )

        self.assertNotIn("timestamps", payload)
        self.assertEqual(payload["timestamp_frames"], [25, 40])
        self.assertEqual(payload["start_time"], 1.0)
        self.assertEqual(payload["end_time"], 1.64)

    def test_clip_names_do_not_collide_for_same_stem(self):
        row = {"audio_path": "same.wav", "keyword": "wake"}
        first = MODULE.ManifestEntry(2, row, Path("/a/same.wav"), "wake", None)
        second = MODULE.ManifestEntry(2, row, Path("/b/same.wav"), "wake", None)
        bounds = MODULE.ClipBounds(0, 640, 0, 640, 16000)

        self.assertNotEqual(
            MODULE.make_clip_filename(first, 0, bounds),
            MODULE.make_clip_filename(second, 0, bounds),
        )

    def test_multi_threshold_clip_name_contains_threshold_tag(self):
        row = {"audio_path": "sample.wav", "keyword": "wake"}
        entry = MODULE.ManifestEntry(2, row, Path("/a/sample.wav"), "wake", 0)
        bounds = MODULE.ClipBounds(0, 640, 0, 640, 16000)

        scalar_name = MODULE.make_clip_filename(entry, 0, bounds)
        scanned_name = MODULE.make_clip_filename(entry, 0, bounds, 0.625)

        self.assertNotIn("_thr_", scalar_name)
        self.assertIn("_thr_0p625_hit_000_", scanned_name)

    def test_manifest_hit_row_records_the_threshold_used(self):
        entry = MODULE.ManifestEntry(
            3,
            {
                "audio_path": "sample.wav",
                "keyword": "wake",
                "label": "0",
                "max_token_gap_sec": "stale",
                "max_keyword_duration_sec": "stale",
            },
            Path("/input/sample.wav"),
            "wake",
            0,
        )
        detection = MODULE.Detection("WAKE", [1, 2], 0.9)
        bounds = MODULE.ClipBounds(640, 1920, 0, 2560, 16000)
        runtime = SimpleNamespace(
            decode_mode="streaming",
            params=SimpleNamespace(chunk_size=16, left_context_frames=64),
            max_token_gap_sec=0.8,
            max_keyword_duration_sec=2.0,
        )

        row = MODULE._manifest_output_row(
            entry=entry,
            detection=detection,
            hit_index=0,
            bounds=bounds,
            clip_path=Path("/output/wavs/clip.wav"),
            output_manifest=Path("/output/manifest.csv"),
            runtime=runtime,
            keyword_threshold=0.625,
        )

        self.assertEqual(row["keywords_threshold"], "0.625")
        self.assertEqual(row["max_token_gap_sec"], "0.8")
        self.assertEqual(row["max_keyword_duration_sec"], "2")

        runtime.max_token_gap_sec = None
        runtime.max_keyword_duration_sec = None
        disabled_row = MODULE._manifest_output_row(
            entry=entry,
            detection=detection,
            hit_index=0,
            bounds=bounds,
            clip_path=Path("/output/wavs/clip.wav"),
            output_manifest=Path("/output/manifest.csv"),
            runtime=runtime,
            keyword_threshold=0.625,
        )
        self.assertEqual(disabled_row["max_token_gap_sec"], "")
        self.assertEqual(disabled_row["max_keyword_duration_sec"], "")

    def test_empty_output_manifest_has_a_valid_header(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.csv"
            MODULE.write_output_manifest(
                output,
                [],
                ["audio_path", "keyword", "label", "text_variant"],
                overwrite=False,
            )
            with output.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader), [])
                self.assertEqual(
                    reader.fieldnames[:4],
                    [
                        "audio_path",
                        "keyword",
                        "label",
                        "text_variant",
                    ],
                )

    def test_atomic_manifest_write_does_not_clobber_fixed_tmp_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.csv"
            sidecar = Path(directory) / "manifest.csv.tmp"
            sidecar.write_text("source data", encoding="utf-8")

            MODULE.write_output_manifest(
                output,
                [],
                ["audio_path", "keyword"],
                overwrite=False,
            )

            self.assertEqual(sidecar.read_text(encoding="utf-8"), "source data")
            self.assertTrue(output.is_file())

    def test_all_input_files_are_protected_from_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_manifest = root / "input.csv"
            first_audio = root / "first.wav"
            second_audio = root / "second.wav"
            output_manifest = root / "output.csv"
            protected = {
                path.resolve()
                for path in (
                    input_manifest,
                    first_audio,
                    second_audio,
                    output_manifest,
                )
            }

            for target in protected:
                with self.assertRaisesRegex(ValueError, "protected file"):
                    MODULE._ensure_output_is_not_protected(
                        target, protected, "generated output"
                    )

            safe = root / "wavs" / "generated.wav"
            MODULE._ensure_output_is_not_protected(safe, protected, "generated output")

    def test_evaluation_artifact_names_cannot_become_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            base = {
                "output_dir": output_dir,
                "output_manifest": None,
                "wav_subdir": "wavs",
                "pre_roll_sec": 0.0,
                "post_roll_sec": 0.0,
                "overwrite": True,
            }
            cases = (
                {"wav_subdir": "results.jsonl/wavs"},
                {"output_manifest": output_dir / "summary.json" / "nested.csv"},
                {"output_manifest": output_dir},
            )
            for overrides in cases:
                with self.subTest(overrides=overrides), self.assertRaisesRegex(
                    ValueError, "reserved evaluation artifact|directory"
                ):
                    MODULE._validate_output_layout(
                        SimpleNamespace(**{**base, **overrides})
                    )

    def test_evaluation_records_keep_emitted_hit_without_exported_clip(self):
        entry = MODULE.ManifestEntry(
            row_number=2,
            row={"audio_path": "audio.wav", "keyword": "WAKE", "label": "0"},
            audio_path=Path("/dataset/audio.wav"),
            keyword="WAKE",
            label=0,
        )
        detection = MODULE.Detection("WAKE", [10, 11], 0.9)
        records = []
        MODULE._append_evaluation_trial_results(
            records=records,
            entry=entry,
            thresholds=(0.5,),
            output_rows=(),
            output_manifest=Path("/output/manifest.csv"),
            artifact_output_dir=Path("/output"),
            detections_by_threshold={0.5: [detection]},
            num_samples=16000,
        )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["detected"])
        self.assertEqual(records[0]["qbyt_score"], 0.9)
        self.assertFalse(records[0]["detections"][0]["clip_exported"])
        self.assertEqual(records[0]["detections"][0]["timestamp_frames"], [10, 11])

    def test_evaluation_clip_path_is_resolved_from_output_manifest(self):
        detection = MODULE.Detection("WAKE", [10, 11], 0.9)
        events = MODULE._evaluation_events(
            [detection],
            [
                {
                    "hit_index": 0,
                    "score": "0.9",
                    "audio_path": "../../output/wavs/hit.wav",
                    "start_sec": "0.1",
                    "end_sec": "0.3",
                }
            ],
            Path("/evaluation/manifests/nested/manifest.csv"),
            Path("/evaluation/output"),
        )

        self.assertEqual(events[0]["clip_audio_path"], "wavs/hit.wav")
        self.assertEqual(
            events[0]["clip_manifest_audio_path"],
            "../../output/wavs/hit.wav",
        )
        self.assertEqual(
            events[0]["clip_audio_path_resolved"],
            "/evaluation/output/wavs/hit.wav",
        )
        self.assertTrue(events[0]["clip_exported"])

    def test_training_policy_requires_explicit_deployment_point(self):
        parser = MODULE.get_parser()
        args = parser.parse_args(["--wav", "a.wav", "--checkpoint", "a.pt"])
        checkpoint = {
            "causal": True,
            "chunk_size": "16,32,64,-1",
            "left_context_frames": "64,128,256,-1",
        }
        with self.assertRaisesRegex(ValueError, "choose one explicitly"):
            MODULE._merge_model_args(checkpoint, {}, args)

        args.chunk_size = 16
        args.left_context_frames = 64
        values = MODULE._merge_model_args(checkpoint, {}, args)
        self.assertEqual(values["chunk_size"], "16")
        self.assertEqual(values["left_context_frames"], "64")
        self.assertEqual(MODULE.validate_streaming_configuration(values), "streaming")

    def test_manifest_progress_flags_are_tristate(self):
        parser = MODULE.get_parser()
        base = ["--manifest", "manifest.csv"]

        defaults = parser.parse_args(base)
        self.assertIsNone(defaults.progress)
        self.assertIsNone(defaults.feature_cache_dir)
        self.assertEqual(defaults.stream_batch_size, 1)
        self.assertTrue(parser.parse_args(base + ["--progress"]).progress)
        self.assertFalse(parser.parse_args(base + ["--no-progress"]).progress)
        batched = parser.parse_args(
            base
            + [
                "--feature-cache-dir",
                "features",
                "--stream-batch-size",
                "8",
            ]
        )
        self.assertEqual(batched.feature_cache_dir, Path("features"))
        self.assertEqual(batched.stream_batch_size, 8)

    def test_manifest_batch_requires_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "--fail-fast"):
            MODULE.main(
                [
                    "--manifest",
                    "manifest.csv",
                    "--feature-cache-dir",
                    "features",
                    "--stream-batch-size",
                    "2",
                ]
            )

    def test_manifest_progress_counts_rows_and_reports_nonfatal_errors(self):
        class FakeWaveform:
            def numel(self):
                return 16000

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input.csv"
            manifest.touch()
            audio_paths = [root / "one.wav", root / "two.wav"]
            for path in audio_paths:
                path.touch()
            entries = [
                MODULE.ManifestEntry(
                    row_number=index + 2,
                    row={"audio_path": path.name, "keyword": "HEY EVA"},
                    audio_path=path,
                    keyword="HEY EVA",
                    label=0,
                )
                for index, path in enumerate(audio_paths)
            ]
            args = SimpleNamespace(
                manifest=manifest,
                output_dir=root / "output",
                output_manifest=None,
                wav_subdir="wavs",
                pre_roll_sec=0.15,
                post_roll_sec=0.15,
                overwrite=True,
                keywords_thresholds="0.2,0.8",
                keywords_threshold=None,
                progress=True,
                fail_fast=False,
            )
            runtime = SimpleNamespace(
                model=SimpleNamespace(
                    parameters=lambda: iter([SimpleNamespace(device="cpu")])
                ),
                sp=object(),
                keywords_score=1.5,
                keywords_threshold=0.5,
            )
            events = []
            fake_torch = SimpleNamespace(
                inference_mode=lambda: contextlib.nullcontext()
            )

            with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.object(
                MODULE,
                "load_manifest",
                return_value=(entries, ["audio_path", "keyword"]),
            ), mock.patch.object(
                MODULE,
                "build_keywords_graphs",
                return_value={0.2: object(), 0.8: object()},
            ), mock.patch.object(
                MODULE,
                "load_audio",
                side_effect=[ValueError("bad audio"), FakeWaveform()],
            ), mock.patch.object(
                MODULE, "compute_fbank", return_value=object()
            ), mock.patch.object(
                MODULE,
                "run_keyword_inference_multi_threshold",
                return_value={0.2: [], 0.8: []},
            ), mock.patch.object(
                MODULE, "write_output_manifest"
            ) as write_manifest, mock.patch.object(
                MODULE, "progress_events_requested", return_value=True
            ), mock.patch.object(
                MODULE,
                "emit_progress_event",
                side_effect=lambda **values: events.append(values),
            ):
                return_code = MODULE.run_manifest(args, runtime)

            self.assertEqual(return_code, 2)
            write_manifest.assert_called_once()
            self.assertEqual(
                events,
                [
                    {"completed": 0, "total": 2, "clips": 0, "misses": 0, "errors": 0},
                    {"completed": 1, "total": 2, "clips": 0, "misses": 0, "errors": 1},
                    {"completed": 2, "total": 2, "clips": 0, "misses": 2, "errors": 1},
                ],
            )
            for filename in (
                "results.jsonl",
                "summary.json",
                "threshold_scan_summary.json",
                "threshold_scan.csv",
                "threshold_scan.png",
            ):
                self.assertTrue((args.output_dir / filename).is_file(), filename)
            result_rows = [
                json.loads(line)
                for line in (args.output_dir / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(result_rows), 4)
            self.assertEqual(sum(bool(row["skipped"]) for row in result_rows), 2)

    def test_manifest_fail_fast_emits_error_progress_and_stops_display(self):
        class SpyProgress(MODULE.ConsoleProgress):
            instances = []

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.stop_count = 0
                self.__class__.instances.append(self)

            def stop(self):
                self.stop_count += 1
                super().stop()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input.csv"
            manifest.touch()
            audio_path = root / "bad.wav"
            audio_path.touch()
            entry = MODULE.ManifestEntry(
                row_number=2,
                row={"audio_path": audio_path.name, "keyword": "HEY EVA"},
                audio_path=audio_path,
                keyword="HEY EVA",
                label=0,
            )
            args = SimpleNamespace(
                manifest=manifest,
                output_dir=root / "output",
                output_manifest=None,
                wav_subdir="wavs",
                pre_roll_sec=0.15,
                post_roll_sec=0.15,
                overwrite=True,
                keywords_thresholds="0.2,0.8",
                keywords_threshold=None,
                progress=True,
                fail_fast=True,
            )
            runtime = SimpleNamespace(
                model=SimpleNamespace(
                    parameters=lambda: iter([SimpleNamespace(device="cpu")])
                ),
                sp=object(),
                keywords_score=1.5,
                keywords_threshold=0.5,
            )
            events = []
            fake_torch = SimpleNamespace(
                inference_mode=lambda: contextlib.nullcontext()
            )

            with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.object(
                MODULE,
                "load_manifest",
                return_value=([entry], ["audio_path", "keyword"]),
            ), mock.patch.object(
                MODULE,
                "build_keywords_graphs",
                return_value={0.2: object(), 0.8: object()},
            ), mock.patch.object(
                MODULE, "load_audio", side_effect=ValueError("bad audio")
            ), mock.patch.object(
                MODULE, "progress_events_requested", return_value=True
            ), mock.patch.object(
                MODULE,
                "emit_progress_event",
                side_effect=lambda **values: events.append(values),
            ), mock.patch.object(MODULE, "ConsoleProgress", SpyProgress):
                with self.assertRaisesRegex(ValueError, "bad audio"):
                    MODULE.run_manifest(args, runtime)

            self.assertEqual(SpyProgress.instances[0].stop_count, 1)
            self.assertEqual(events[-1]["completed"], 1)
            self.assertEqual(events[-1]["errors"], 1)

    def test_oov_keyword_is_rejected_before_graph_construction(self):
        class FakeSentencePiece:
            def unk_id(self):
                return 3

            def encode(self, phrase):
                del phrase
                return [1, 3]

        with self.assertRaisesRegex(ValueError, "outside the SentencePiece vocabulary"):
            MODULE.encode_keywords(FakeSentencePiece(), ["unknown wake word"])

    def test_text_keyword_remains_backward_compatible(self):
        sp = FakeSentencePiece()

        phrases, token_ids = MODULE.encode_keywords(sp, ["hey eva"])

        self.assertEqual(phrases, ["HEY EVA"])
        self.assertEqual(token_ids, [[3, 4]])
        self.assertEqual(sp.encode_calls, ["HEY EVA"])

    def test_bpe_id_keywords_bypass_sentencepiece_encoding(self):
        for value in ("[3,4]", "bpe_ids:3 4", "bpe_ids:[3,4]"):
            with self.subTest(value=value):
                sp = FakeSentencePiece()
                phrases, token_ids = MODULE.encode_keywords(sp, [value])
                self.assertEqual(phrases, ["HEY EVA"])
                self.assertEqual(token_ids, [[3, 4]])
                self.assertEqual(sp.encode_calls, [])

    def test_bpe_piece_keywords_bypass_sentencepiece_encoding(self):
        values = (
            "bpe_pieces:\u2581HEY \u2581EVA",
            'bpe_pieces:["\u2581HEY","\u2581EVA"]',
            '["\u2581HEY","\u2581EVA"]',
        )
        for value in values:
            with self.subTest(value=value):
                sp = FakeSentencePiece()
                phrases, token_ids = MODULE.encode_keywords(sp, [value])
                self.assertEqual(phrases, ["HEY EVA"])
                self.assertEqual(token_ids, [[3, 4]])
                self.assertEqual(sp.encode_calls, [])

    def test_direct_bpe_rejects_invalid_or_special_ids(self):
        cases = (
            ("[0,3]", "<blk>"),
            ("[1,3]", "<sos/eos>"),
            ("[2,3]", "<unk>"),
            ("[3,99]", "outside"),
            ("[true]", "integer IDs"),
        )
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.encode_keywords(FakeSentencePiece(), [value])

    def test_reserved_sos_eos_piece_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "<sos/eos>"):
            MODULE.encode_keywords(FakeSentencePiece(), ["bpe_pieces:<sos/eos>"])

    def test_unknown_bpe_piece_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in the SentencePiece vocabulary"):
            MODULE.encode_keywords(
                FakeSentencePiece(), ["bpe_pieces:\u2581HEY \u2581MISSING"]
            )

    def test_missing_blank_cannot_alias_to_unk(self):
        class MissingBlankSentencePiece:
            pieces = ["<unk>", "\u2581HEY"]

            def piece_to_id(self, piece):
                try:
                    return self.pieces.index(piece)
                except ValueError:
                    return 0

            def id_to_piece(self, token_id):
                return self.pieces[token_id]

        sp = MissingBlankSentencePiece()
        self.assertEqual(sp.piece_to_id("<blk>"), sp.piece_to_id("<unk>"))
        with self.assertRaisesRegex(ValueError, "no exact '<blk>' token"):
            MODULE._require_sentencepiece_piece(sp, "<blk>")

    def test_custom_semantic_unk_does_not_alias_literal_unk_piece(self):
        class CustomUnkSentencePiece:
            pieces = ["<blk>", "<unk>", "\u2581HEY", "[UNK]"]

            def unk_id(self):
                return 3

            def get_piece_size(self):
                return len(self.pieces)

            def piece_to_id(self, piece):
                try:
                    return self.pieces.index(piece)
                except ValueError:
                    return self.unk_id()

            def id_to_piece(self, token_id):
                return self.pieces[token_id]

            def is_unknown(self, token_id):
                return token_id == self.unk_id()

        sp = CustomUnkSentencePiece()
        self.assertEqual(MODULE._require_sentencepiece_unk_id(sp), 3)
        self.assertEqual(MODULE._exact_sentencepiece_piece_id(sp, "<unk>"), 1)
        with self.assertRaisesRegex(ValueError, "<unk>"):
            MODULE.encode_keywords(sp, ["bpe_ids:3"])
        with self.assertRaisesRegex(ValueError, "reserved token <unk>"):
            MODULE.encode_keywords(sp, ["bpe_ids:1"])

    def test_equivalent_text_ids_and_pieces_are_deduplicated(self):
        sp = FakeSentencePiece()

        phrases, token_ids = MODULE.encode_keywords(
            sp, ["hey eva", "bpe_ids:3 4", "bpe_pieces:\u2581HEY \u2581EVA"]
        )

        self.assertEqual(phrases, ["HEY EVA"])
        self.assertEqual(token_ids, [[3, 4]])

    def test_text_prefix_escapes_a_bracketed_keyword(self):
        sp = FakeSentencePiece()

        phrases, token_ids = MODULE.encode_keywords(sp, ["text:[alarm]"])

        self.assertEqual(phrases, ["[ALARM]"])
        self.assertEqual(token_ids, [[5]])
        self.assertEqual(sp.encode_calls, ["[ALARM]"])

    def test_malformed_bare_json_is_not_silently_encoded_as_text(self):
        sp = FakeSentencePiece()
        with self.assertRaisesRegex(ValueError, "invalid JSON BPE keyword"):
            MODULE.encode_keywords(sp, ["[3, nope]"])
        self.assertEqual(sp.encode_calls, [])

    def test_misspelled_bpe_prefix_is_not_silently_encoded_as_text(self):
        sp = FakeSentencePiece()
        with self.assertRaisesRegex(ValueError, "unknown BPE keyword prefix"):
            MODULE.encode_keywords(sp, ["bpe_id:3 4"])
        self.assertEqual(sp.encode_calls, [])

    def test_non_causal_checkpoint_uses_full_context(self):
        parser = MODULE.get_parser()
        args = parser.parse_args(
            ["--wav", "a.wav", "--checkpoint", "a.pt", "--causal", "0"]
        )
        values = MODULE._merge_model_args(
            {
                "causal": True,
                "chunk_size": "16,32,64,-1",
                "left_context_frames": "64,128,256,-1",
            },
            {},
            args,
        )
        self.assertEqual(values["chunk_size"], "-1")
        self.assertEqual(values["left_context_frames"], "-1")
        self.assertEqual(MODULE.validate_streaming_configuration(values), "offline")


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TestStatefulKeywordDecoder(unittest.TestCase):
    class State:
        def __init__(
            self,
            tokens=(),
            *,
            phrase="",
            threshold=0.0,
            node_score=0.0,
            state_id=0,
        ):
            self.tokens = tuple(tokens)
            self.token = -1 if not tokens else tokens[-1]
            self.level = len(tokens)
            self.phrase = phrase
            self.ac_threshold = threshold
            self.node_score = node_score
            self.id = state_id
            self.fail = None

    class Graph:
        def __init__(self, threshold=0.5):
            self.root = TestStatefulKeywordDecoder.State(state_id=0)
            self.first = TestStatefulKeywordDecoder.State(
                (1,), node_score=1.0, state_id=1
            )
            self.matched = TestStatefulKeywordDecoder.State(
                (1, 2),
                phrase="WAKE",
                threshold=threshold,
                node_score=2.0,
                state_id=2,
            )
            self.root.fail = self.root
            self.first.fail = self.root
            self.matched.fail = self.root

        def forward_one_step(self, state, token):
            if state is self.root and token == 1:
                return 1.0, self.first, None
            if state is self.first and token == 2:
                return 1.0, self.matched, self.matched
            return -state.node_score, self.root, None

        def is_matched(self, state):
            return (
                state is self.matched,
                self.matched if state is self.matched else None,
            )

        def finalize(self, state):
            return -state.node_score, self.root

    class ThreeTokenGraph:
        def __init__(self):
            State = TestStatefulKeywordDecoder.State
            self.root = State(state_id=10)
            self.first = State((1,), node_score=1.0, state_id=11)
            self.second = State((1, 2), node_score=2.0, state_id=12)
            self.matched = State(
                (1, 2, 1),
                phrase="LONG",
                threshold=0.5,
                node_score=3.0,
                state_id=13,
            )
            self.root.fail = self.root
            self.first.fail = self.root
            self.second.fail = self.root
            self.matched.fail = self.first

        def forward_one_step(self, state, token):
            if state is self.root and token == 1:
                return 1.0, self.first, None
            if state is self.first and token == 2:
                return 1.0, self.second, None
            if state is self.second and token == 1:
                return 1.0, self.matched, self.matched
            return -state.node_score, self.root, None

        def is_matched(self, state):
            return (
                state is self.matched,
                self.matched if state is self.matched else None,
            )

    class OverlapGraph:
        def __init__(self):
            State = TestStatefulKeywordDecoder.State
            self.root = State(state_id=20)
            self.a = State((1,), node_score=1.0, state_id=21)
            self.b = State((2,), node_score=1.0, state_id=22)
            self.ab = State((1, 2), node_score=2.0, state_id=23)
            self.ba = State(
                (2, 1),
                phrase="SHORT",
                threshold=0.5,
                node_score=2.0,
                state_id=24,
            )
            self.root.fail = self.root
            self.a.fail = self.root
            self.b.fail = self.root
            self.ab.fail = self.b
            self.ba.fail = self.a

        def forward_one_step(self, state, token):
            if state is self.root and token == 1:
                return 1.0, self.a, None
            if state is self.root and token == 2:
                return 1.0, self.b, None
            if state is self.a and token == 2:
                return 1.0, self.ab, None
            if state in (self.b, self.ab) and token == 1:
                score = self.ba.node_score - state.node_score
                return score, self.ba, self.ba
            return -state.node_score, self.root, None

        def is_matched(self, state):
            return state is self.ba, self.ba if state is self.ba else None

    if torch is not None:

        class Decoder(torch.nn.Module):
            blank_id = 0
            context_size = 1

            def forward(self, decoder_input, need_pad=False):
                del need_pad
                return torch.zeros(
                    decoder_input.size(0), 1, 4, device=decoder_input.device
                )

        class Joiner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_proj_calls = 0

            def encoder_proj(self, value):
                self.encoder_proj_calls += 1
                return value

            def decoder_proj(self, value):
                return value

            def forward(self, encoder, decoder, project_input=False):
                del decoder, project_input
                return encoder

        class Model(torch.nn.Module):
            unk_id = 3

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.decoder = TestStatefulKeywordDecoder.Decoder()
                self.joiner = TestStatefulKeywordDecoder.Joiner()
                self.forward_encoder_calls = 0

            def forward_encoder(self, features, feature_lens):
                self.forward_encoder_calls += 1
                return features, feature_lens

    def test_keyword_and_trailing_blanks_can_cross_chunk_boundaries(self):
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=self.Graph(),
            beam=1,
            num_tailing_blanks=1,
            blank_penalty=0.0,
        )
        # Each tensor is a separate encoder chunk. The keyword tokens 1 and 2
        # are split across chunks, as are the two blanks required by strict > 1.
        chunks = [
            torch.tensor([[[-8.0, 8.0, -8.0, -8.0]]]),
            torch.tensor([[[-8.0, -8.0, 8.0, -8.0]]]),
            torch.tensor([[[8.0, -8.0, -8.0, -8.0]]]),
            torch.tensor([[[8.0, -8.0, -8.0, -8.0]]]),
        ]

        hits = []
        for chunk in chunks:
            hits.extend(decoder.advance(chunk))

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].phrase, "WAKE")
        self.assertEqual(hits[0].timestamp_frames, [0, 1])
        self.assertGreater(hits[0].score, 0.99)

    @staticmethod
    def _frame(token):
        values = torch.full((1, 1, 4), -8.0)
        values[0, 0, token] = 8.0
        return values

    def _decode_frames(self, decoder, tokens):
        hits = []
        for token in tokens:
            hits.extend(decoder.advance(self._frame(token)))
        return hits

    def test_disabled_timing_limits_preserve_long_blank_gap(self):
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=self.Graph(),
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
        )

        hits = self._decode_frames(decoder, [1, 0, 0, 0, 0, 0, 2, 0])

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp_frames, [0, 6])

    def test_token_gap_exact_boundary_is_allowed_but_next_frame_expires(self):
        for tokens, expected in (([1, 0, 2, 0], 1), ([1, 0, 0, 2, 0], 0)):
            with self.subTest(tokens=tokens):
                decoder = MODULE.StatefulKeywordDecoder(
                    model=self.Model(),
                    keywords_graph=self.Graph(),
                    beam=1,
                    num_tailing_blanks=0,
                    blank_penalty=0.0,
                    max_token_gap_sec=0.08,
                )

                hits = self._decode_frames(decoder, tokens)

                self.assertEqual(len(hits), expected)
                if hits:
                    self.assertEqual(hits[0].timestamp_frames, [0, 2])

    def test_total_duration_expires_even_when_each_gap_is_valid(self):
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=self.ThreeTokenGraph(),
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            max_token_gap_sec=0.08,
            max_keyword_duration_sec=0.08,
        )

        hits = self._decode_frames(decoder, [1, 2, 1, 0])

        self.assertEqual(hits, [])

    def test_duration_timeout_keeps_the_longest_valid_fail_suffix(self):
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=self.OverlapGraph(),
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            max_keyword_duration_sec=0.08,
        )

        hits = self._decode_frames(decoder, [1, 2, 1, 0])

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].phrase, "SHORT")
        self.assertEqual(hits[0].timestamp_frames, [1, 2])

    def test_duration_timeout_uses_real_context_graph_fail_suffix(self):
        graph = CONTEXT_GRAPH_MODULE.ContextGraph(
            context_score=1.0, ac_threshold=0.5
        )
        graph.build(
            token_ids=[[1, 2, 1], [2, 1]],
            phrases=["LONG", "SHORT"],
        )
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=graph,
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            max_keyword_duration_sec=0.08,
        )

        hits = self._decode_frames(decoder, [1, 2, 1, 0])

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].phrase, "SHORT")
        self.assertEqual(hits[0].timestamp_frames, [1, 2])

    def test_completed_output_suffix_cannot_extend_after_timeout(self):
        graph = CONTEXT_GRAPH_MODULE.ContextGraph(
            context_score=1.0, ac_threshold=0.5
        )
        graph.build(
            token_ids=[[2], [1, 2, 1]],
            phrases=["SHORT", "LONG"],
        )
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=graph,
            beam=1,
            num_tailing_blanks=100,
            blank_penalty=0.0,
            max_token_gap_sec=0.04,
        )

        hits = self._decode_frames(decoder, [1, 2, 0, 1, 0, 0])

        self.assertEqual(hits, [])
        top = decoder.hypotheses.most_probable()
        self.assertIs(top.context_state, graph.root)
        # Preserve SHORT's completed output bonus, but neither the expired
        # LONG prefix bonus nor an invalid LONG completion bonus.
        self.assertAlmostEqual(float(top.log_prob), 1.0, places=4)

    def test_expired_retry_merges_with_existing_root_before_topk(self):
        graph = self.Graph()
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=graph,
            beam=2,
            num_tailing_blanks=100,
            blank_penalty=0.0,
            max_token_gap_sec=0.04,
        )
        decoder.frame_offset = 3
        decoder.hypotheses = MODULE._HypothesisList()
        decoder.hypotheses.add(
            MODULE._DecoderHypothesis(
                ys=[0],
                log_prob=torch.tensor(0.0),
                context_state=graph.root,
                use_timing_key=True,
                score_length=3,
            )
        )
        decoder.hypotheses.add(
            MODULE._DecoderHypothesis(
                ys=[0, 1, 2],
                log_prob=torch.tensor(2.0),
                context_state=graph.matched,
                timestamp=[0, 1],
                ac_probs=[0.9, 0.9],
                use_timing_key=True,
                score_length=3,
            )
        )
        frame = torch.tensor([[[0.0, -0.4, -8.0, -8.0]]])

        decoder.advance(frame)

        self.assertTrue(
            any(
                hyp.context_state is graph.first
                for hyp in decoder.hypotheses.values()
            )
        )

    def test_expired_nonblank_uses_fallback_predictor_history(self):
        class RecordingDecoder(self.Decoder):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, decoder_input, need_pad=False):
                self.inputs.append(decoder_input.detach().cpu().tolist())
                return super().forward(decoder_input, need_pad=need_pad)

        model = self.Model()
        model.decoder = RecordingDecoder()
        graph = self.Graph()
        decoder = MODULE.StatefulKeywordDecoder(
            model=model,
            keywords_graph=graph,
            beam=1,
            num_tailing_blanks=100,
            blank_penalty=0.0,
            max_token_gap_sec=0.04,
        )
        decoder.frame_offset = 3
        decoder.hypotheses = MODULE._HypothesisList()
        decoder.hypotheses.add(
            MODULE._DecoderHypothesis(
                ys=[0, 1, 2],
                log_prob=torch.tensor(2.0),
                context_state=graph.matched,
                timestamp=[0, 1],
                ac_probs=[0.9, 0.9],
                use_timing_key=True,
                score_length=3,
            )
        )

        decoder.advance(self._frame(1))

        self.assertEqual({tuple(row) for row in model.decoder.inputs[-1]}, {(2,), (0,)})
        top = decoder.hypotheses.most_probable()
        self.assertIs(top.context_state, graph.first)
        self.assertEqual(top.timestamp, [3])

    def test_timing_mode_canonicalizes_root_without_resetting_score_length(self):
        graph = self.Graph()
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=graph,
            beam=4,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            max_token_gap_sec=0.04,
        )

        decoder.advance(self._frame(2))

        root_hypotheses = [
            hyp
            for hyp in decoder.hypotheses.values()
            if hyp.context_state is graph.root
        ]
        self.assertEqual(len(root_hypotheses), 2)
        self.assertEqual({tuple(hyp.ys) for hyp in root_hypotheses}, {(0,)})
        self.assertEqual(
            {hyp.effective_score_length for hyp in root_hypotheses}, {1, 2}
        )
        for hyp in root_hypotheses:
            self.assertAlmostEqual(
                MODULE._hypothesis_rank(hyp),
                float(hyp.log_prob) / hyp.effective_score_length,
            )

    def test_gap_timeout_retries_the_current_token_from_root(self):
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=self.OverlapGraph(),
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            max_token_gap_sec=0.04,
        )

        hits = self._decode_frames(decoder, [1, 0, 2, 1, 0])

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].phrase, "SHORT")
        self.assertEqual(hits[0].timestamp_frames, [2, 3])

    def test_completed_keyword_can_wait_for_tail_blanks_past_gap_limit(self):
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=self.Graph(),
            beam=1,
            num_tailing_blanks=3,
            blank_penalty=0.0,
            max_token_gap_sec=0.04,
            max_keyword_duration_sec=0.08,
        )

        hits = self._decode_frames(decoder, [1, 2, 0, 0, 0, 0])

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp_frames, [0, 1])

    def test_timeout_rolls_back_active_context_bonus(self):
        graph = self.Graph()
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=graph,
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            max_token_gap_sec=0.04,
        )
        hyp = MODULE._DecoderHypothesis(
            ys=[0, 1],
            log_prob=torch.tensor(5.0),
            context_state=graph.first,
            timestamp=[0],
            ac_probs=[0.9],
            use_timing_key=True,
            score_length=7,
        )

        normalized = decoder._normalize_hypothesis_time(hyp, absolute_t=2)

        self.assertIs(normalized.context_state, graph.root)
        self.assertAlmostEqual(float(normalized.log_prob), 4.0)
        self.assertEqual(normalized.timestamp, [])
        self.assertEqual(normalized.ac_probs, [])
        self.assertEqual(normalized.ys, [0])
        self.assertEqual(normalized.effective_score_length, 7)

    def test_timing_merge_key_keeps_distinct_token_alignments(self):
        graph = self.Graph()
        hypotheses = MODULE._HypothesisList()
        for timestamp in (0, 1):
            hypotheses.add(
                MODULE._DecoderHypothesis(
                    ys=[0, 1],
                    log_prob=torch.tensor(0.0),
                    context_state=graph.first,
                    timestamp=[timestamp],
                    ac_probs=[0.9],
                    use_timing_key=True,
                )
            )

        self.assertEqual(len(hypotheses.data), 2)

    def test_finalize_skips_only_the_higher_invalid_alignment(self):
        graph = self.Graph()
        decoder = MODULE.StatefulKeywordDecoder(
            model=self.Model(),
            keywords_graph=graph,
            beam=2,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            max_token_gap_sec=0.08,
            max_keyword_duration_sec=0.12,
        )
        decoder.hypotheses = MODULE._HypothesisList()
        for log_prob, timestamps in ((5.0, [0, 3]), (4.0, [1, 2])):
            decoder.hypotheses.add(
                MODULE._DecoderHypothesis(
                    ys=[0, 1, 2],
                    log_prob=torch.tensor(log_prob),
                    context_state=graph.matched,
                    timestamp=timestamps,
                    ac_probs=[0.9, 0.9],
                    use_timing_key=True,
                )
            )

        hits = decoder.finalize()

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp_frames, [1, 2])

    def test_multi_threshold_decoder_projects_once_and_keeps_independent_state(self):
        model = self.Model()
        low_graph = self.Graph(0.5)
        high_graph = self.Graph(1.0)
        decoder = MODULE.MultiThresholdKeywordDecoder(
            model=model,
            keywords_graphs={0.5: low_graph, 1.0: high_graph},
            beam=1,
            num_tailing_blanks=1,
            blank_penalty=0.0,
        )
        chunks = [
            torch.tensor([[[-8.0, 8.0, -8.0, -8.0]]]),
            torch.tensor([[[-8.0, -8.0, 8.0, -8.0]]]),
            torch.tensor([[[8.0, -8.0, -8.0, -8.0]]]),
            torch.tensor([[[8.0, -8.0, -8.0, -8.0]]]),
        ]

        hits = {0.5: [], 1.0: []}
        for chunk in chunks:
            for threshold, values in decoder.advance(chunk).items():
                hits[threshold].extend(values)

        self.assertEqual(model.joiner.encoder_proj_calls, len(chunks))
        self.assertEqual(len(hits[0.5]), 1)
        self.assertEqual(hits[1.0], [])
        # The accepted decoder resets after its hit; the rejected decoder
        # remains at the matched graph state. Their search state is not shared.
        self.assertIs(
            decoder.decoders[0.5].hypotheses.most_probable().context_state,
            low_graph.root,
        )
        self.assertIs(
            decoder.decoders[1.0].hypotheses.most_probable().context_state,
            high_graph.matched,
        )

    def test_multi_threshold_timing_limits_still_project_once(self):
        model = self.Model()
        low_graph = self.Graph(0.5)
        high_graph = self.Graph(1.0)
        decoder = MODULE.MultiThresholdKeywordDecoder(
            model=model,
            keywords_graphs={0.5: low_graph, 1.0: high_graph},
            beam=1,
            num_tailing_blanks=1,
            blank_penalty=0.0,
            max_token_gap_sec=0.08,
            max_keyword_duration_sec=0.08,
        )
        chunks = [self._frame(token) for token in (1, 2, 0, 0)]

        hits = {0.5: [], 1.0: []}
        for chunk in chunks:
            for threshold, values in decoder.advance(chunk).items():
                hits[threshold].extend(values)

        self.assertEqual(model.joiner.encoder_proj_calls, len(chunks))
        self.assertEqual(len(hits[0.5]), 1)
        self.assertEqual(hits[1.0], [])
        self.assertIs(
            decoder.decoders[0.5].hypotheses.most_probable().context_state,
            low_graph.root,
        )
        self.assertIs(
            decoder.decoders[1.0].hypotheses.most_probable().context_state,
            high_graph.root,
        )

    def test_streaming_state_stack_unstack_round_trip(self):
        def one_stream(value):
            return [
                torch.full((2, 1, 3), value),
                torch.full((1, 1, 2, 4), value),
                torch.full((2, 1, 3), value),
                torch.full((2, 1, 3), value),
                torch.full((1, 3, 2), value),
                torch.full((1, 3, 2), value),
                torch.full((1, 2, 3, 4), value),
                torch.tensor([int(value)], dtype=torch.int32),
            ]

        original = [one_stream(1.0), one_stream(2.0)]
        restored = MODULE.unstack_states(MODULE.stack_states(original))

        self.assertEqual(len(restored), len(original))
        for expected_states, actual_states in zip(original, restored):
            for expected, actual in zip(expected_states, actual_states):
                self.assertTrue(torch.equal(expected, actual))

    @staticmethod
    def _fake_streaming_states(batch_size, device):
        return [
            torch.zeros(2, batch_size, 3, device=device),
            torch.zeros(1, batch_size, 2, 4, device=device),
            torch.zeros(2, batch_size, 3, device=device),
            torch.zeros(2, batch_size, 3, device=device),
            torch.zeros(batch_size, 3, 2, device=device),
            torch.zeros(batch_size, 3, 2, device=device),
            torch.zeros(batch_size, 2, 3, 4, device=device),
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        ]

    @staticmethod
    def _token_features(tokens):
        values = torch.full((len(tokens), 4), -8.0)
        for index, token in enumerate(tokens):
            values[index, token] = 8.0
        return values

    def test_batched_streaming_matches_scalar_and_projects_once(self):
        model = self.Model()
        runtime = SimpleNamespace(
            model=model,
            params=SimpleNamespace(chunk_size=16, left_context_frames=64),
            decode_mode="streaming",
            max_token_gap_sec=None,
            max_keyword_duration_sec=None,
        )
        args = SimpleNamespace(
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            tail_padding_sec=0.0,
        )
        features = [
            self._token_features([1, 2, 0]),
            self._token_features([1, 0, 2, 0]),
        ]
        observed_processed_lens = []

        def fake_get_init_states(model, device, batch_size=1):
            del model
            return self._fake_streaming_states(batch_size, device)

        def fake_streaming_forward(**kwargs):
            feature_batch = kwargs["features"]
            states = kwargs["states"]
            observed_processed_lens.append(states[-1].tolist())
            output = feature_batch[:, :, :4].clone()
            padding = torch.isclose(
                output,
                torch.tensor(MODULE.LOG_EPS, device=output.device),
            ).all(dim=-1)
            output[padding] = torch.tensor(
                [8.0, -8.0, -8.0, -8.0], device=output.device
            )
            new_states = [value.clone() for value in states]
            new_states[-1] = states[-1] + 1
            lengths = torch.full(
                (output.size(0),),
                output.size(1),
                dtype=torch.int64,
                device=output.device,
            )
            return output, lengths, new_states

        with mock.patch.object(
            MODULE, "get_init_states", side_effect=fake_get_init_states
        ), mock.patch.object(
            MODULE, "streaming_forward", side_effect=fake_streaming_forward
        ):
            scalar = []
            for values in features:
                scalar.append(
                    MODULE.run_keyword_inference_multi_threshold(
                        runtime=runtime,
                        features=values,
                        keywords_graphs={0.5: self.Graph(0.5), 0.9: self.Graph(0.9)},
                        args=args,
                    )
                )

            model.joiner.encoder_proj_calls = 0
            items = [
                MODULE.KeywordBatchItem(
                    item_id=index,
                    features=values,
                    keywords_graphs={0.5: self.Graph(0.5), 0.9: self.Graph(0.9)},
                )
                for index, values in enumerate(features)
            ]
            batched = dict(
                MODULE.run_keyword_inference_multi_threshold_batched(
                    runtime=runtime,
                    items=items,
                    args=args,
                    batch_size=2,
                )
            )

        self.assertEqual(model.joiner.encoder_proj_calls, 1)
        for index, expected_by_threshold in enumerate(scalar):
            for threshold, expected in expected_by_threshold.items():
                actual = batched[index][threshold]
                self.assertEqual(
                    [(hit.phrase, hit.timestamp_frames) for hit in actual],
                    [(hit.phrase, hit.timestamp_frames) for hit in expected],
                )
                self.assertEqual(len(actual), len(expected))
                for actual_hit, expected_hit in zip(actual, expected):
                    self.assertAlmostEqual(actual_hit.score, expected_hit.score)
        self.assertEqual(observed_processed_lens[-1], [0, 0])

    def test_batched_streaming_refills_with_fresh_state_and_no_cross_stream_hit(self):
        model = self.Model()
        runtime = SimpleNamespace(
            model=model,
            params=SimpleNamespace(chunk_size=16, left_context_frames=64),
            decode_mode="streaming",
            max_token_gap_sec=None,
            max_keyword_duration_sec=None,
        )
        args = SimpleNamespace(
            beam=1,
            num_tailing_blanks=0,
            blank_penalty=0.0,
            tail_padding_sec=0.0,
        )
        observed_processed_lens = []

        def fake_get_init_states(model, device, batch_size=1):
            del model
            return self._fake_streaming_states(batch_size, device)

        def fake_streaming_forward(**kwargs):
            feature_batch = kwargs["features"]
            states = kwargs["states"]
            observed_processed_lens.append(states[-1].tolist())
            output = feature_batch[:, :, :4].clone()
            padding = torch.isclose(
                output,
                torch.tensor(MODULE.LOG_EPS, device=output.device),
            ).all(dim=-1)
            output[padding] = torch.tensor(
                [8.0, -8.0, -8.0, -8.0], device=output.device
            )
            new_states = [value.clone() for value in states]
            new_states[-1] = states[-1] + 1
            lengths = torch.full(
                (output.size(0),),
                output.size(1),
                dtype=torch.int64,
                device=output.device,
            )
            return output, lengths, new_states

        items = [
            MODULE.KeywordBatchItem(
                "only-first",
                self._token_features([1, 0]),
                {0.5: self.Graph()},
            ),
            MODULE.KeywordBatchItem(
                "long",
                self._token_features([0] * 33),
                {0.5: self.Graph()},
            ),
            MODULE.KeywordBatchItem(
                "only-second",
                self._token_features([2, 0]),
                {0.5: self.Graph()},
            ),
        ]
        with mock.patch.object(
            MODULE, "get_init_states", side_effect=fake_get_init_states
        ), mock.patch.object(
            MODULE, "streaming_forward", side_effect=fake_streaming_forward
        ):
            results = dict(
                MODULE.run_keyword_inference_multi_threshold_batched(
                    runtime=runtime,
                    items=items,
                    args=args,
                    batch_size=2,
                )
            )

        self.assertEqual(observed_processed_lens, [[0, 0], [1, 0]])
        self.assertEqual(results["only-first"][0.5], [])
        self.assertEqual(results["only-second"][0.5], [])

    def test_cached_manifest_without_hits_never_reloads_waveform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input.csv"
            manifest.touch()
            audio = root / "audio.wav"
            audio.write_bytes(b"source")
            cache_dir = root / "features"
            MODULE.FbankFeatureCache(cache_dir).store(
                audio,
                torch.zeros(3, 80),
                num_samples=1600,
            )
            entry = MODULE.ManifestEntry(
                row_number=2,
                row={"audio_path": audio.name, "keyword": "HEY EVA"},
                audio_path=audio,
                keyword="HEY EVA",
                label=0,
            )
            args = SimpleNamespace(
                manifest=manifest,
                output_dir=root / "output",
                output_manifest=None,
                wav_subdir="wavs",
                pre_roll_sec=0.15,
                post_roll_sec=0.15,
                overwrite=True,
                keywords_thresholds=None,
                keywords_threshold=None,
                progress=False,
                fail_fast=True,
                feature_cache_dir=cache_dir,
                stream_batch_size=1,
            )
            runtime = SimpleNamespace(
                model=self.Model(),
                params=SimpleNamespace(chunk_size=16, left_context_frames=64),
                sp=object(),
                keywords_score=1.5,
                keywords_threshold=0.5,
                decode_mode="streaming",
                max_token_gap_sec=None,
                max_keyword_duration_sec=None,
            )

            with mock.patch.object(
                MODULE,
                "load_manifest",
                return_value=([entry], ["audio_path", "keyword"]),
            ), mock.patch.object(
                MODULE, "build_keywords_graphs", return_value={0.5: self.Graph()}
            ), mock.patch.object(
                MODULE,
                "run_keyword_inference_multi_threshold",
                return_value={0.5: []},
            ), mock.patch.object(
                MODULE, "load_audio", side_effect=AssertionError("unexpected read")
            ) as load_audio, mock.patch.object(
                MODULE, "compute_fbank", side_effect=AssertionError("unexpected fbank")
            ), mock.patch.object(MODULE, "write_output_manifest") as write_manifest:
                return_code = MODULE.run_manifest(args, runtime)

            self.assertEqual(return_code, 0)
            load_audio.assert_not_called()
            write_manifest.assert_called_once()

    def test_cached_manifest_loads_waveform_once_for_hits_at_many_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input.csv"
            manifest.touch()
            audio = root / "audio.wav"
            audio.write_bytes(b"source")
            cache_dir = root / "features"
            MODULE.FbankFeatureCache(cache_dir).store(
                audio,
                torch.zeros(3, 80),
                num_samples=1600,
            )
            entry = MODULE.ManifestEntry(
                row_number=2,
                row={"audio_path": audio.name, "keyword": "HEY EVA"},
                audio_path=audio,
                keyword="HEY EVA",
                label=0,
            )
            args = SimpleNamespace(
                manifest=manifest,
                output_dir=root / "output",
                output_manifest=None,
                wav_subdir="wavs",
                pre_roll_sec=0.0,
                post_roll_sec=0.0,
                overwrite=True,
                keywords_thresholds="0.5,0.8",
                keywords_threshold=None,
                progress=False,
                fail_fast=True,
                feature_cache_dir=cache_dir,
                stream_batch_size=1,
            )
            runtime = SimpleNamespace(
                model=self.Model(),
                params=SimpleNamespace(chunk_size=16, left_context_frames=64),
                sp=object(),
                keywords_score=1.5,
                keywords_threshold=0.5,
                decode_mode="streaming",
                max_token_gap_sec=None,
                max_keyword_duration_sec=None,
            )
            detection = MODULE.Detection("WAKE", [1, 2], 0.9)

            with mock.patch.object(
                MODULE,
                "load_manifest",
                return_value=([entry], ["audio_path", "keyword"]),
            ), mock.patch.object(
                MODULE,
                "build_keywords_graphs",
                return_value={0.5: self.Graph(), 0.8: self.Graph(0.8)},
            ), mock.patch.object(
                MODULE,
                "run_keyword_inference_multi_threshold",
                return_value={0.5: [detection], 0.8: [detection]},
            ), mock.patch.object(
                MODULE, "load_audio", return_value=torch.zeros(1600)
            ) as load_audio, mock.patch.object(
                MODULE, "compute_fbank", side_effect=AssertionError("unexpected fbank")
            ), mock.patch.object(
                MODULE, "_write_wav_atomic"
            ) as write_wav, mock.patch.object(
                MODULE, "write_output_manifest"
            ) as write_manifest:
                return_code = MODULE.run_manifest(args, runtime)

            self.assertEqual(return_code, 0)
            load_audio.assert_called_once_with(audio)
            self.assertEqual(write_wav.call_count, 2)
            rows = write_manifest.call_args.args[1]
            self.assertEqual(
                [row["keywords_threshold"] for row in rows], ["0.5", "0.8"]
            )

    def test_cached_manifest_batch_restores_row_and_threshold_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input.csv"
            manifest.touch()
            cache_dir = root / "features"
            entries = []
            audio_paths = []
            for index in range(3):
                audio = root / "audio-{}.wav".format(index)
                audio.write_bytes("source-{}".format(index).encode("ascii"))
                audio_paths.append(audio)
                MODULE.FbankFeatureCache(cache_dir).store(
                    audio,
                    torch.full((index + 1, 80), float(index)),
                    num_samples=1600,
                )
                entries.append(
                    MODULE.ManifestEntry(
                        row_number=index + 2,
                        row={"audio_path": audio.name, "keyword": "HEY EVA"},
                        audio_path=audio,
                        keyword="HEY EVA",
                        label=0,
                    )
                )

            args = SimpleNamespace(
                manifest=manifest,
                output_dir=root / "output",
                output_manifest=None,
                wav_subdir="wavs",
                pre_roll_sec=0.0,
                post_roll_sec=0.0,
                overwrite=True,
                keywords_thresholds="0.75,0.25",
                keywords_threshold=None,
                progress=False,
                fail_fast=True,
                feature_cache_dir=cache_dir,
                stream_batch_size=2,
            )
            runtime = SimpleNamespace(
                model=self.Model(),
                params=SimpleNamespace(chunk_size=16, left_context_frames=64),
                sp=object(),
                keywords_score=1.5,
                keywords_threshold=0.5,
                decode_mode="streaming",
                max_token_gap_sec=None,
                max_keyword_duration_sec=None,
            )
            low_hit = MODULE.Detection("WAKE", [1, 2], 0.8)
            high_hit = MODULE.Detection("WAKE", [2, 3], 0.9)
            loaded_audio = []

            def fake_batched(*, runtime, items, args, batch_size):
                del runtime, args
                materialized = list(items)
                self.assertEqual(batch_size, 2)
                self.assertEqual([item.item_id for item in materialized], [0, 1, 2])
                self.assertEqual(
                    [item.features.size(0) for item in materialized], [1, 2, 3]
                )
                # Simulate dynamic streams finishing in a different order.
                for item_id in (2, 0, 1):
                    yield item_id, {0.75: [high_hit], 0.25: [low_hit]}

            def fake_load_audio(path):
                loaded_audio.append(path)
                return torch.zeros(1600)

            with mock.patch.object(
                MODULE,
                "load_manifest",
                return_value=(entries, ["audio_path", "keyword"]),
            ), mock.patch.object(
                MODULE,
                "build_keywords_graphs",
                return_value={0.25: self.Graph(0.25), 0.75: self.Graph(0.75)},
            ), mock.patch.object(
                MODULE,
                "run_keyword_inference_multi_threshold_batched",
                side_effect=fake_batched,
            ) as batched_inference, mock.patch.object(
                MODULE,
                "run_keyword_inference_multi_threshold",
                side_effect=AssertionError("scalar inference should not run"),
            ), mock.patch.object(
                MODULE, "load_audio", side_effect=fake_load_audio
            ), mock.patch.object(
                MODULE, "compute_fbank", side_effect=AssertionError("unexpected fbank")
            ), mock.patch.object(
                MODULE, "_write_wav_atomic"
            ) as write_wav, mock.patch.object(
                MODULE, "write_output_manifest"
            ) as write_manifest:
                return_code = MODULE.run_manifest(args, runtime)

            self.assertEqual(return_code, 0)
            batched_inference.assert_called_once()
            self.assertEqual(write_wav.call_count, 6)
            self.assertEqual(len(loaded_audio), len(audio_paths))
            for audio in audio_paths:
                self.assertEqual(loaded_audio.count(audio), 1)

            rows = write_manifest.call_args.args[1]
            self.assertEqual(
                [
                    (row["source_manifest_row"], row["keywords_threshold"])
                    for row in rows
                ],
                [
                    (2, "0.25"),
                    (2, "0.75"),
                    (3, "0.25"),
                    (3, "0.75"),
                    (4, "0.25"),
                    (4, "0.75"),
                ],
            )

    def test_offline_multi_threshold_inference_runs_encoder_once(self):
        model = self.Model()
        runtime = MODULE.LoadedRuntime(
            model=model,
            sp=None,
            params=SimpleNamespace(causal=False),
            checkpoint_path=Path("checkpoint.pt"),
            decode_mode="offline",
            keywords_score=1.0,
            keywords_threshold=0.5,
            config={},
        )
        args = SimpleNamespace(
            beam=1,
            num_tailing_blanks=1,
            blank_penalty=0.0,
            tail_padding_sec=0.0,
        )
        features = torch.full((MODULE.ENCODER_EMBED_PAD, 4), -8.0)
        features[:, 0] = 8.0
        features[0] = torch.tensor([-8.0, 8.0, -8.0, -8.0])
        features[1] = torch.tensor([-8.0, -8.0, 8.0, -8.0])

        hits = MODULE.run_keyword_inference_multi_threshold(
            runtime=runtime,
            features=features,
            keywords_graphs={0.5: self.Graph(0.5), 1.0: self.Graph(1.0)},
            args=args,
        )

        self.assertEqual(model.forward_encoder_calls, 1)
        self.assertEqual(model.joiner.encoder_proj_calls, 1)
        self.assertEqual(len(hits[0.5]), 1)
        self.assertEqual(hits[1.0], [])


if __name__ == "__main__":
    unittest.main()
