import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).with_name("prepare-calibration-features.py")
SPEC = importlib.util.spec_from_file_location("prepare_calibration_features", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare)


def test_normalize_features() -> None:
    features = np.ones((3, 80), dtype=np.float64)
    actual = prepare.normalize_features(features, "sample", num_mel_bins=80)
    assert actual.shape == (3, 80)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous

    with pytest.raises(ValueError, match="not floating point"):
        prepare.normalize_features(
            np.ones((3, 80), dtype=np.int32), "sample", num_mel_bins=80
        )
    with pytest.raises(ValueError, match="expected"):
        prepare.normalize_features(
            np.ones((3, 40), dtype=np.float32), "sample", num_mel_bins=80
        )
    with pytest.raises(ValueError, match="empty"):
        prepare.normalize_features(
            np.empty((0, 80), dtype=np.float32), "sample", num_mel_bins=80
        )
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        prepare.normalize_features(features, "sample", num_mel_bins=80)


def test_discover_audio_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.WAV").touch()
    (tmp_path / "nested" / "a.flac").touch()
    (tmp_path / "ignored.txt").touch()

    actual = prepare.discover_audio_files(tmp_path)
    assert actual == sorted([tmp_path / "b.WAV", tmp_path / "nested" / "a.flac"])


def test_iter_audio_features(tmp_path: Path, monkeypatch) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "sample.wav").touch()

    class FakeRecording:
        sampling_rate = 8000
        duration = 1.0
        channel_ids = [0, 1]

        @staticmethod
        def from_file(filename: Path, recording_id: str):
            assert filename == audio_dir / "sample.wav"
            assert recording_id.startswith("calibration-")
            return FakeRecording()

        def resample(self, sample_rate: int):
            self.sampling_rate = sample_rate
            return self

    class FakeMonoCut:
        def __init__(self, **kwargs) -> None:
            assert kwargs["channel"] == 0

        def compute_features(self, extractor: object) -> np.ndarray:
            assert extractor is fake_extractor
            return np.ones((3, 80), dtype=np.float32)

    fake_lhotse = types.ModuleType("lhotse")
    fake_lhotse.Recording = FakeRecording
    fake_lhotse.MonoCut = FakeMonoCut
    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)
    fake_extractor = object()
    monkeypatch.setattr(
        prepare,
        "create_kaldifeat_extractor",
        lambda *args: fake_extractor,
    )

    records = list(
        prepare.iter_audio_features(
            audio_dir=audio_dir,
            max_utterances=1,
            sample_rate=16000,
            num_mel_bins=80,
            device="cpu",
        )
    )
    assert records[0][0] == "sample.wav"
    assert records[0][1].shape == (3, 80)
    assert records[0][2] == "computed"


def test_generate_feature_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "features"
    records = [
        ("speaker/a.wav", np.ones((3, 80), dtype=np.float64), "computed"),
        ("cut-b", np.full((2, 80), 0.5, dtype=np.float32), "stored"),
    ]

    manifest = prepare.generate_feature_directory(
        records=records,
        output_dir=output_dir,
        num_mel_bins=80,
        log_interval=10,
        source_mode="test",
        source_path=tmp_path,
        sample_rate=16000,
    )

    assert manifest["num_utterances"] == 2
    assert manifest["origin_counts"] == {"computed": 1, "stored": 1}
    npz_files = sorted(output_dir.glob("*.npz"))
    assert len(npz_files) == 2
    with np.load(npz_files[0], allow_pickle=False) as archive:
        assert archive.files == ["features"]
        assert archive["features"].shape == (3, 80)
        assert archive["features"].dtype == np.float32

    saved_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert saved_manifest == manifest

    with pytest.raises(FileExistsError, match="must be empty"):
        prepare.prepare_output_dir(output_dir)


class FakeCut:
    def __init__(
        self,
        cut_id: str,
        features: np.ndarray,
        has_features: bool,
        sampling_rate: int = 16000,
    ) -> None:
        self.id = cut_id
        self.features = features
        self.has_features = has_features
        self.has_recording = True
        self.sampling_rate = sampling_rate
        self.resampled_to = None
        self.extractor = None

    def load_features(self) -> np.ndarray:
        return self.features

    def resample(self, sample_rate: int):
        self.resampled_to = sample_rate
        self.sampling_rate = sample_rate
        return self

    def compute_features(self, extractor: object) -> np.ndarray:
        self.extractor = extractor
        return self.features


def test_iter_cut_features_loads_or_computes_lazily() -> None:
    stored = FakeCut("stored", np.ones((2, 80), dtype=np.float32), True)
    computed = FakeCut(
        "computed", np.ones((3, 80), dtype=np.float32), False, sampling_rate=8000
    )
    extractor = object()
    factory_calls = 0

    def extractor_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return extractor

    records = list(
        prepare.iter_cut_features(
            cuts=[stored, computed],
            max_utterances=2,
            sample_rate=16000,
            extractor_factory=extractor_factory,
        )
    )

    assert [record[2] for record in records] == ["stored", "computed"]
    assert factory_calls == 1
    assert computed.resampled_to == 16000
    assert computed.extractor is extractor
