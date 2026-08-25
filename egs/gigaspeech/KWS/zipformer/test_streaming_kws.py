#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("streaming_kws.py")
SPEC = importlib.util.spec_from_file_location("streaming_kws", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


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
        return "".join(self.pieces[token_id] for token_id in token_ids).replace(
            "\u2581", " "
        ).strip()


class TestManifestAndTimestamps(unittest.TestCase):
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
        detection = MODULE.Detection(
            phrase="WAKE", timestamp_frames=[0], score=1.0
        )
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
                self.assertEqual(reader.fieldnames[:4], [
                    "audio_path",
                    "keyword",
                    "label",
                    "text_variant",
                ])

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
            MODULE._ensure_output_is_not_protected(
                safe, protected, "generated output"
            )

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
        def __init__(self, tokens=(), *, phrase="", threshold=0.0):
            self.tokens = tuple(tokens)
            self.token = -1 if not tokens else tokens[-1]
            self.level = len(tokens)
            self.phrase = phrase
            self.ac_threshold = threshold

    class Graph:
        def __init__(self):
            self.root = TestStatefulKeywordDecoder.State()
            self.first = TestStatefulKeywordDecoder.State((1,))
            self.matched = TestStatefulKeywordDecoder.State(
                (1, 2), phrase="WAKE", threshold=0.5
            )

        def forward_one_step(self, state, token):
            if state is self.root and token == 1:
                return 0.0, self.first, None
            if state is self.first and token == 2:
                return 0.0, self.matched, self.matched
            return 0.0, self.root, None

        def is_matched(self, state):
            return (state is self.matched, self.matched if state is self.matched else None)

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
            def encoder_proj(self, value):
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


if __name__ == "__main__":
    unittest.main()
