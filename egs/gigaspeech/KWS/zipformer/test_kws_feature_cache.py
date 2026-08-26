#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("kws_feature_cache.py")
SPEC = importlib.util.spec_from_file_location("kws_feature_cache_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


@unittest.skipIf(torch is None or np is None, "torch and numpy are required")
class TestFbankFeatureCache(unittest.TestCase):
    def test_store_load_and_source_or_frontend_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"first")
            cache = MODULE.FbankFeatureCache(root / "cache", spec={"frontend": 1})
            features = torch.arange(240, dtype=torch.float32).reshape(3, 80)

            self.assertIsNone(cache.load(audio))
            metadata = cache.store(audio, features, num_samples=1600)
            loaded = cache.load(audio)

            self.assertIsNotNone(loaded)
            self.assertTrue(torch.equal(loaded.features, features))
            self.assertEqual(loaded.num_samples, 1600)
            self.assertEqual(loaded.key, metadata["key"])

            changed_frontend = MODULE.FbankFeatureCache(
                root / "cache", spec={"frontend": 2}
            )
            self.assertIsNone(changed_frontend.load(audio))

            audio.write_bytes(b"second version")
            self.assertIsNone(cache.load(audio))

    def test_incomplete_or_corrupt_entry_is_never_a_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            cache = MODULE.FbankFeatureCache(root / "cache", spec={"frontend": 1})
            _, key = cache._current_identity(audio)
            data_path, metadata_path = cache._paths(key)
            data_path.parent.mkdir(parents=True, exist_ok=True)

            with data_path.open("wb") as handle:
                np.save(handle, np.zeros((2, 80), dtype=np.float32))
            self.assertIsNone(cache.lookup_metadata(audio))

            metadata_path.write_text("{broken", encoding="utf-8")
            self.assertIsNone(cache.lookup_metadata(audio))

            cache.store(audio, torch.zeros(2, 80), num_samples=800)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["shape"] = [3, 80]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertIsNone(cache.lookup_metadata(audio))

            cache.store(audio, torch.zeros(2, 80), num_samples=800)
            data_path.write_bytes(b"truncated")
            self.assertIsNone(cache.lookup_metadata(audio))

    def test_resolved_path_prevents_same_name_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "same.wav"
            second = root / "b" / "same.wav"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            cache = MODULE.FbankFeatureCache(root / "cache", spec={"frontend": 1})

            _, first_key = cache._current_identity(first)
            _, second_key = cache._current_identity(second)

            self.assertNotEqual(first_key, second_key)

    def test_invalid_feature_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            cache = MODULE.FbankFeatureCache(root / "cache", spec={"frontend": 1})

            with self.assertRaisesRegex(ValueError, "shape"):
                cache.store(audio, torch.zeros(2, 79), num_samples=800)
            values = torch.zeros(2, 80)
            values[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                cache.store(audio, values, num_samples=800)
            with self.assertRaisesRegex(ValueError, "num_samples"):
                cache.store(audio, torch.zeros(2, 80), num_samples=0)
            with self.assertRaisesRegex(ValueError, "num_samples"):
                cache.store(audio, torch.zeros(2, 80), num_samples=1.5)

    def test_read_only_cache_does_not_create_or_modify_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            cache_root = root / "missing-cache"

            cache = MODULE.FbankFeatureCache(
                cache_root,
                spec={"frontend": 1},
                writable=False,
            )

            self.assertIsNone(cache.lookup_metadata(audio))
            self.assertFalse(cache_root.exists())
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                cache.store(audio, torch.zeros(2, 80), num_samples=800)

    def test_invalid_numeric_metadata_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            cache = MODULE.FbankFeatureCache(root / "cache", spec={"frontend": 1})
            cache.store(audio, torch.zeros(2, 80), num_samples=800)
            _, key = cache._current_identity(audio)
            _, metadata_path = cache._paths(key)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            metadata["num_samples"] = "800"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertIsNone(cache.lookup_metadata(audio))

            cache.store(audio, torch.zeros(2, 80), num_samples=800)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["data_size"] = None
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertIsNone(cache.lookup_metadata(audio))

    def test_prepare_rejects_audio_changed_during_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"before")
            cache_root = root / "cache"

            def mutate_source(_path):
                audio.write_bytes(b"after-is-different")
                return torch.zeros(1600)

            with mock.patch.object(
                MODULE,
                "load_audio",
                side_effect=mutate_source,
            ), mock.patch.object(
                MODULE,
                "compute_fbank",
                return_value=torch.zeros(2, 80),
            ):
                with self.assertRaisesRegex(RuntimeError, "audio changed"):
                    MODULE.prepare_cached_fbank(
                        str(audio),
                        str(cache_root),
                    )

            cache = MODULE.FbankFeatureCache(
                cache_root,
                spec=MODULE.frontend_spec(),
                writable=False,
            )
            self.assertIsNone(cache.lookup_metadata(audio))


if __name__ == "__main__":
    unittest.main()
