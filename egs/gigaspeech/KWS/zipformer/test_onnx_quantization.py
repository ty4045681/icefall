import json
import shutil
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
quantization = pytest.importorskip("onnx_quantization")

TensorProto = onnx.TensorProto
helper = onnx.helper
numpy_helper = onnx.numpy_helper
NpzCalibrationDataReader = quantization.NpzCalibrationDataReader
calibration_bundle_hash = quantization.calibration_bundle_hash
compare_models_on_npz = quantization.compare_models_on_npz
compare_streaming_model_sets = quantization.compare_streaming_model_sets
generate_calibration_bundle = quantization.generate_calibration_bundle
load_calibration_manifest = quantization.load_calibration_manifest
load_model_io_schema = quantization.load_model_io_schema
model_artifact_hash = quantization.model_artifact_hash
normalize_static_quantization_op_types = (
    quantization.normalize_static_quantization_op_types
)
quantize_static_int8 = quantization.quantize_static_int8
quantize_weight_only_int8 = quantization.quantize_weight_only_int8
resolve_quantization_mode = quantization.resolve_quantization_mode
resolve_static_quantization_op_types = quantization.resolve_static_quantization_op_types
validate_calibration_manifest = quantization.validate_calibration_manifest
validate_quantized_model_artifact = quantization.validate_quantized_model_artifact
_load_feature_array = quantization._load_feature_array


@pytest.mark.parametrize(
    "mode,legacy,expected",
    [
        (None, 1, "dynamic"),
        (None, 0, "none"),
        ("none", 1, "none"),
        ("dynamic", 0, "dynamic"),
        ("weight-only", 1, "weight-only"),
        ("static", 1, "static"),
        ("all", 0, "all"),
    ],
)
def test_resolve_quantization_mode(mode, legacy, expected) -> None:
    assert resolve_quantization_mode(mode, legacy) == expected


@pytest.mark.parametrize(
    "profile,component,expected",
    [
        ("legacy", "encoder", ("MatMul", "Conv")),
        ("balanced", "encoder", ("MatMul", "Gemm", "Conv")),
        ("balanced", "decoder", ("Gather", "MatMul", "Gemm", "Conv")),
        ("balanced", "joiner", ("MatMul", "Gemm")),
    ],
)
def test_static_quantization_profiles(profile, component, expected) -> None:
    assert resolve_static_quantization_op_types(component, profile) == expected


def test_static_quantization_extended_profile_and_override() -> None:
    extended = resolve_static_quantization_op_types("encoder", "extended")
    assert extended[:3] == ("MatMul", "Gemm", "Conv")
    assert {
        "Transpose",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Flatten",
        "Slice",
        "Split",
        "Add",
        "Mul",
    }.issubset(extended)
    assert "Gather" in resolve_static_quantization_op_types("decoder", "extended")
    assert normalize_static_quantization_op_types(" MatMul, Gather,MatMul ") == (
        "MatMul",
        "Gather",
    )
    assert resolve_static_quantization_op_types(
        "decoder", "legacy", "Gather,MatMul"
    ) == ("Gather", "MatMul")

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_static_quantization_op_types("")
    with pytest.raises(ValueError, match="Invalid ONNX op type"):
        normalize_static_quantization_op_types("MatMul,com.microsoft::Foo")
    with pytest.raises(ValueError, match="component"):
        resolve_static_quantization_op_types("frontend", "balanced")
    with pytest.raises(ValueError, match="profile"):
        resolve_static_quantization_op_types("encoder", "aggressive")


def _make_weight_model(
    filename: Path, external_data: bool = False, large: bool = False
) -> None:
    if large:
        matmul_weight = np.linspace(-1, 1, 3 * 512, dtype=np.float32).reshape(3, 512)
    else:
        matmul_weight = np.array(
            [[0.2, 0.0], [-0.5, 0.0], [1.0, 0.0]], dtype=np.float32
        )
    conv_weight = np.linspace(-0.9, 0.9, 18, dtype=np.float32).reshape(2, 1, 3, 3)
    nodes = [
        helper.make_node("MatMul", ["x", "matmul_weight"], ["matmul_out"]),
        helper.make_node("Conv", ["image", "conv_weight"], ["conv_out"]),
        helper.make_node("Identity", ["ids"], ["ids_out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "weight_only_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3]),
            helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 1, 4, 4]),
            helper.make_tensor_value_info("ids", TensorProto.INT64, ["N"]),
        ],
        [
            helper.make_tensor_value_info(
                "matmul_out", TensorProto.FLOAT, ["N", matmul_weight.shape[1]]
            ),
            helper.make_tensor_value_info(
                "conv_out", TensorProto.FLOAT, ["N", 2, 2, 2]
            ),
            helper.make_tensor_value_info("ids_out", TensorProto.INT64, ["N"]),
        ],
        [
            numpy_helper.from_array(matmul_weight, "matmul_weight"),
            numpy_helper.from_array(conv_weight, "conv_weight"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="icefall-test",
        ir_version=9,
    )
    metadata = model.metadata_props.add()
    metadata.key = "test_metadata"
    metadata.value = "preserved"
    if external_data:
        onnx.save_model(
            model,
            str(filename),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{filename.stem}.weights",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, str(filename))


def _run_model(filename: Path, inputs):
    session = ort.InferenceSession(str(filename), providers=["CPUExecutionProvider"])
    return session.run(None, inputs)


def test_weight_only_preserves_schema_and_outputs(tmp_path: Path) -> None:
    fp32_filename = tmp_path / "model.onnx"
    int8_filename = tmp_path / "model.weight-only-int8.onnx"
    _make_weight_model(fp32_filename)
    original_schema = load_model_io_schema(fp32_filename)

    report = quantize_weight_only_int8(fp32_filename, int8_filename)

    assert report["quantized_weight_count"] == 2
    assert load_model_io_schema(int8_filename) == original_schema
    quantized_model = onnx.load(str(int8_filename))
    assert [
        item.value
        for item in quantized_model.metadata_props
        if item.key == "test_metadata"
    ] == ["preserved"]
    assert quantized_model.opset_import[0].version == 13
    assert all(node.domain == "" for node in quantized_model.graph.node)
    assert not any(
        node.op_type == "QuantizeLinear" for node in quantized_model.graph.node
    )
    assert (
        sum(node.op_type == "DequantizeLinear" for node in quantized_model.graph.node)
        == 2
    )
    initializer_types = {
        initializer.name: initializer.data_type
        for initializer in quantized_model.graph.initializer
    }
    assert "matmul_weight" not in initializer_types
    assert "conv_weight" not in initializer_types
    assert sum(value == TensorProto.INT8 for value in initializer_types.values()) == 4

    inputs = {
        "x": np.array([[0.1, -0.2, 0.3], [0.5, 0.4, -0.1]], dtype=np.float32),
        "image": np.linspace(-1, 1, 32, dtype=np.float32).reshape(2, 1, 4, 4),
        "ids": np.array([3, 7], dtype=np.int64),
    }
    expected = _run_model(fp32_filename, inputs)
    actual = _run_model(int8_filename, inputs)
    np.testing.assert_allclose(actual[0], expected[0], atol=0.01, rtol=0.01)
    np.testing.assert_allclose(actual[1], expected[1], atol=0.01, rtol=0.01)
    np.testing.assert_array_equal(actual[2], expected[2])


def test_weight_only_shares_dequantized_weight(tmp_path: Path) -> None:
    filename = tmp_path / "shared.onnx"
    output_filename = tmp_path / "shared.weight-only-int8.onnx"
    weight = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x1", "weight"], ["y1"]),
            helper.make_node("MatMul", ["x2", "weight"], ["y2"]),
        ],
        "shared_weight",
        [
            helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, 3]),
        ],
        [
            helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 4]),
        ],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
    )
    onnx.save_model(model, str(filename))

    report = quantize_weight_only_int8(filename, output_filename)

    assert report["quantized_weight_count"] == 1
    output_model = onnx.load(str(output_filename))
    assert (
        sum(node.op_type == "DequantizeLinear" for node in output_model.graph.node) == 1
    )
    matmul_weights = [
        node.input[1] for node in output_model.graph.node if node.op_type == "MatMul"
    ]
    assert len(set(matmul_weights)) == 1


def test_weight_only_axes_zero_channel_and_shared_consumer(tmp_path: Path) -> None:
    fp32_filename = tmp_path / "shared-consumer.onnx"
    int8_filename = tmp_path / "shared-consumer.weight-only-int8.onnx"
    weight = np.array([[0.2, 0.0], [-0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "weight"], ["matmul_out"]),
            helper.make_node("Identity", ["weight"], ["weight_out"]),
        ],
        "shared_consumer",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [
            helper.make_tensor_value_info("matmul_out", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("weight_out", TensorProto.FLOAT, [3, 2]),
        ],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
    )
    onnx.save_model(model, str(fp32_filename))

    quantize_weight_only_int8(fp32_filename, int8_filename)

    output_model = onnx.load(str(int8_filename))
    assert any(item.name == "weight" for item in output_model.graph.initializer)
    dequantize = next(
        node for node in output_model.graph.node if node.op_type == "DequantizeLinear"
    )
    axis = next(item.i for item in dequantize.attribute if item.name == "axis")
    assert axis == 1
    scale_initializer = next(
        item
        for item in output_model.graph.initializer
        if item.name == dequantize.input[1]
    )
    scale = numpy_helper.to_array(scale_initializer)
    assert scale[1] == 1


def test_weight_only_rejects_nonfinite_weight(tmp_path: Path) -> None:
    fp32_filename = tmp_path / "nonfinite.onnx"
    int8_filename = tmp_path / "nonfinite.weight-only-int8.onnx"
    weight = np.array([[np.nan], [1.0]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="bad_matmul")],
        "nonfinite",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
    )
    onnx.save_model(model, str(fp32_filename))

    with pytest.raises(ValueError, match="weight contains NaN/Inf"):
        quantize_weight_only_int8(fp32_filename, int8_filename)


def test_quantized_artifact_allows_existing_custom_domain(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        quantization.ort,
        "InferenceSession",
        lambda *args, **kwargs: object(),
    )
    fp32_filename = tmp_path / "custom.onnx"
    int8_filename = tmp_path / "custom.weight-only-int8.onnx"
    _make_weight_model(fp32_filename)
    model = onnx.load(str(fp32_filename))
    model.graph.node.append(
        helper.make_node(
            "CustomIdentity",
            ["ids"],
            ["unused_custom_output"],
            domain="example.custom",
        )
    )
    model.opset_import.append(helper.make_opsetid("example.custom", 1))
    onnx.save_model(model, str(fp32_filename))

    quantize_weight_only_int8(fp32_filename, int8_filename)
    output_model = onnx.load(str(int8_filename), load_external_data=False)
    assert sum(node.domain == "example.custom" for node in output_model.graph.node) == 1


def test_quantized_artifact_rejects_new_contrib_domain(tmp_path: Path) -> None:
    fp32_filename = tmp_path / "model.onnx"
    int8_filename = tmp_path / "model.weight-only-int8.onnx"
    _make_weight_model(fp32_filename)
    quantize_weight_only_int8(fp32_filename, int8_filename)
    model = onnx.load(str(int8_filename))
    identity = next(node for node in model.graph.node if node.op_type == "Identity")
    identity.domain = "com.microsoft"
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save_model(model, str(int8_filename))

    with pytest.raises(ValueError, match="non-standard ONNX content"):
        validate_quantized_model_artifact(
            fp32_filename,
            int8_filename,
            {"icefall.quantization.mode": "weight_only_int8"},
        )


def test_quantized_artifact_rejects_changed_existing_custom_content(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        quantization.ort,
        "InferenceSession",
        lambda *args, **kwargs: object(),
    )
    fp32_filename = tmp_path / "custom.onnx"
    int8_filename = tmp_path / "custom.weight-only-int8.onnx"
    _make_weight_model(fp32_filename)
    source = onnx.load(str(fp32_filename))
    source.graph.node.append(
        helper.make_node(
            "CustomIdentity",
            ["ids"],
            ["unused_custom_output"],
            domain="example.custom",
        )
    )
    source.opset_import.append(helper.make_opsetid("example.custom", 1))
    onnx.save_model(source, str(fp32_filename))

    quantize_weight_only_int8(fp32_filename, int8_filename)
    candidate = onnx.load(str(int8_filename))
    custom_node = next(
        node for node in candidate.graph.node if node.domain == "example.custom"
    )
    custom_node.op_type = "ChangedCustomIdentity"
    onnx.save_model(candidate, str(int8_filename))

    with pytest.raises(ValueError, match="non-standard ONNX content"):
        validate_quantized_model_artifact(
            fp32_filename,
            int8_filename,
            {"icefall.quantization.mode": "weight_only_int8"},
        )


def _make_subgraph_external_data_model(filename: Path) -> Path:
    subgraph_weight = np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8)
    then_graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "subgraph_weight"], ["then_out"])],
        "then_graph",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(subgraph_weight, "subgraph_weight")],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["else_out"])],
        "else_graph",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [1, 8])],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["y"],
                then_branch=then_graph,
                else_branch=else_graph,
            )
        ],
        "subgraph_external",
        [
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
    )
    location = f"{filename.stem}.weights"
    onnx.save_model(
        model,
        str(filename),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=0,
    )
    return filename.parent / location


def test_subgraph_external_data_is_hashed_and_required(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    candidate_dir = tmp_path / "candidate"
    source_dir.mkdir()
    candidate_dir.mkdir()
    source_filename = source_dir / "subgraph.onnx"
    weights_filename = _make_subgraph_external_data_model(source_filename)
    original_hash = model_artifact_hash(source_filename)

    weights = bytearray(weights_filename.read_bytes())
    weights[-1] ^= 1
    weights_filename.write_bytes(weights)
    assert model_artifact_hash(source_filename) != original_hash
    weights[-1] ^= 1
    weights_filename.write_bytes(weights)

    candidate_filename = candidate_dir / source_filename.name
    shutil.copy2(source_filename, candidate_filename)
    with pytest.raises(ValueError, match="Missing external ONNX data file"):
        validate_quantized_model_artifact(source_filename, candidate_filename, {})


def test_weight_only_external_data(tmp_path: Path) -> None:
    fp32_filename = tmp_path / "external.onnx"
    int8_filename = tmp_path / "external.weight-only-int8.onnx"
    _make_weight_model(fp32_filename, external_data=True)

    quantize_weight_only_int8(
        fp32_filename,
        int8_filename,
        use_external_data=True,
    )

    onnx.checker.check_model(str(int8_filename))
    output_model = onnx.load(str(int8_filename), load_external_data=False)
    external_initializers = {
        initializer.name
        for initializer in output_model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL
    }
    locations = {
        item.value
        for initializer in output_model.graph.initializer
        for item in initializer.external_data
        if item.key == "location"
    }
    assert len(external_initializers) == len(output_model.graph.initializer)
    assert locations == {f"{int8_filename.stem}.weights"}
    weights_filename = tmp_path / f"{int8_filename.stem}.weights"
    assert weights_filename.is_file()
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    moved_model = moved_dir / int8_filename.name
    shutil.copy2(int8_filename, moved_model)
    shutil.copy2(weights_filename, moved_dir / weights_filename.name)
    ort.InferenceSession(str(moved_model), providers=["CPUExecutionProvider"])

    original_hash = model_artifact_hash(int8_filename)
    weights = bytearray(weights_filename.read_bytes())
    weights[-1] ^= 1
    weights_filename.write_bytes(weights)
    assert model_artifact_hash(int8_filename) != original_hash


def _calibration_inputs(batch_size: int = 1):
    return {
        "x": np.full((batch_size, 3), 0.25, dtype=np.float32),
        "image": np.full((batch_size, 1, 4, 4), -0.5, dtype=np.float32),
        "ids": np.arange(batch_size, dtype=np.int64),
    }


def test_npz_calibration_reader_validation_and_rewind(tmp_path: Path) -> None:
    model_filename = tmp_path / "model.onnx"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    _make_weight_model(model_filename)
    np.savez(sample_dir / "000001.npz", **_calibration_inputs())
    np.savez(sample_dir / "000000.npz", **_calibration_inputs(2))

    reader = NpzCalibrationDataReader(model_filename, sample_dir)

    assert [filename.name for filename in reader.files] == ["000000.npz", "000001.npz"]
    assert reader.get_next()["x"].shape == (2, 3)
    assert reader.get_next()["x"].shape == (1, 3)
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next()["x"].shape == (2, 3)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    values = _calibration_inputs()
    values["x"] = values["x"].astype(np.float64)
    np.savez(bad_dir / "000000.npz", **values)
    with pytest.raises(ValueError, match="dtype"):
        NpzCalibrationDataReader(model_filename, bad_dir)


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_key", "Input names"),
        ("extra_key", "Input names"),
        ("integer_dtype", "dtype"),
        ("rank", "rank"),
        ("fixed_dimension", "expected dimension"),
        ("nonfinite", "NaN/Inf"),
    ],
)
def test_npz_calibration_reader_rejects_invalid_samples(
    tmp_path: Path, case: str, match: str
) -> None:
    model_filename = tmp_path / "model.onnx"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    _make_weight_model(model_filename)
    values = _calibration_inputs()
    if case == "missing_key":
        values.pop("ids")
    elif case == "extra_key":
        values["extra"] = np.zeros((1,), dtype=np.float32)
    elif case == "integer_dtype":
        values["ids"] = values["ids"].astype(np.float32)
    elif case == "rank":
        values["x"] = values["x"][0]
    elif case == "fixed_dimension":
        values["x"] = np.zeros((1, 4), dtype=np.float32)
    elif case == "nonfinite":
        values["x"][0, 0] = np.inf
    np.savez(sample_dir / "000000.npz", **values)

    with pytest.raises(ValueError, match=match):
        NpzCalibrationDataReader(model_filename, sample_dir)


def test_npz_calibration_reader_rejects_empty_dir_and_invalid_limit(
    tmp_path: Path,
) -> None:
    model_filename = tmp_path / "model.onnx"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    _make_weight_model(model_filename)
    with pytest.raises(ValueError, match="No calibration NPZ"):
        NpzCalibrationDataReader(model_filename, sample_dir)

    np.savez(sample_dir / "000000.npz", **_calibration_inputs())
    with pytest.raises(ValueError, match="max_samples must be positive"):
        NpzCalibrationDataReader(model_filename, sample_dir, max_samples=0)


@pytest.mark.parametrize("calibration_method", ["minmax", "percentile"])
@pytest.mark.parametrize("external_data", [False, True])
def test_static_qdq_quantization(
    tmp_path: Path, calibration_method: str, external_data: bool
) -> None:
    model_filename = tmp_path / "model.onnx"
    output_filename = tmp_path / f"model.{calibration_method}.static-int8.onnx"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    _make_weight_model(model_filename, large=external_data)
    for index in range(3):
        values = _calibration_inputs(1)
        values["x"] *= index + 1
        np.savez(sample_dir / f"{index:06d}.npz", **values)

    report = quantize_static_int8(
        model_input=model_filename,
        model_output=output_filename,
        sample_dir=sample_dir,
        calibration_method=calibration_method,
        use_external_data=external_data,
        calibration_bundle_hash_value="a" * 64,
    )

    assert report["sample_count"] == 3
    assert report["op_types_to_quantize"] == ["MatMul", "Conv"]
    assert report["matmul_const_b_only"] is False
    assert report["preprocess"] is False
    assert load_model_io_schema(output_filename) == load_model_io_schema(model_filename)
    model = onnx.load(str(output_filename))
    assert any(node.op_type == "QuantizeLinear" for node in model.graph.node)
    assert any(node.op_type == "DequantizeLinear" for node in model.graph.node)
    initializer_types = {
        initializer.name: initializer.data_type
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            assert initializer_types[node.input[2]] == TensorProto.INT8
    producers = {output: node for node in model.graph.node for output in node.output}
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Conv"):
            continue
        dequantize = producers.get(node.input[1])
        assert dequantize is not None and dequantize.op_type == "DequantizeLinear"
        axis = next(
            attribute.i
            for attribute in dequantize.attribute
            if attribute.name == "axis"
        )
        assert axis == (1 if node.op_type == "MatMul" else 0)
    metadata = {item.key: item.value for item in model.metadata_props}
    assert metadata["test_metadata"] == "preserved"
    assert metadata["icefall.quantization.mode"] == "static_int8"
    assert metadata["icefall.quantization.op_types"] == "MatMul,Conv"
    assert metadata["icefall.quantization.matmul_const_b_only"] == "0"
    assert metadata["icefall.quantization.preprocess"] == "0"
    assert metadata["icefall.quantization.calibration_bundle_sha256"] == "a" * 64
    assert "icefall.quantization.calibration_manifest_sha256" not in metadata
    if external_data:
        external_model = onnx.load(str(output_filename), load_external_data=False)
        assert any(
            initializer.data_location == TensorProto.EXTERNAL
            for initializer in external_model.graph.initializer
        )
        locations = {
            item.value
            for initializer in external_model.graph.initializer
            for item in initializer.external_data
            if item.key == "location"
        }
        moved_dir = tmp_path / f"moved-{calibration_method}"
        moved_dir.mkdir()
        moved_model = moved_dir / output_filename.name
        shutil.copy2(output_filename, moved_model)
        for location in locations:
            shutil.copy2(output_filename.parent / location, moved_dir / location)
        ort.InferenceSession(str(moved_model), providers=["CPUExecutionProvider"])
    assert metadata["icefall.quantization.matmul_axis"] == "1"
    stats = compare_models_on_npz(
        model_filename, output_filename, sample_dir, max_samples=2
    )
    assert set(stats) == {"matmul_out", "conv_out", "ids_out"}
    assert stats["ids_out"]["max_abs"] == 0


@pytest.mark.parametrize("external_data", [False, True])
def test_static_qdq_configurable_preprocess_and_matmul_policy(
    tmp_path: Path, monkeypatch, external_data: bool
) -> None:
    model_filename = tmp_path / "model.onnx"
    output_filename = tmp_path / "model.configured.static-int8.onnx"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    _make_weight_model(
        model_filename,
        external_data=external_data,
        large=external_data,
    )
    np.savez(sample_dir / "000000.npz", **_calibration_inputs())

    calls = {}
    original_preprocess = quantization.quant_pre_process
    original_quantize_static = quantization.quantize_static

    def record_preprocess(*args, **kwargs):
        calls["preprocess_args"] = args
        calls["preprocess_kwargs"] = kwargs
        return original_preprocess(*args, **kwargs)

    def record_quantize_static(*args, **kwargs):
        calls["quantize_kwargs"] = kwargs
        return original_quantize_static(*args, **kwargs)

    monkeypatch.setattr(quantization, "quant_pre_process", record_preprocess)
    monkeypatch.setattr(quantization, "quantize_static", record_quantize_static)

    report = quantize_static_int8(
        model_input=model_filename,
        model_output=output_filename,
        sample_dir=sample_dir,
        calibration_method="minmax",
        use_external_data=external_data,
        op_types_to_quantize="MatMul,Conv,MatMul",
        matmul_const_b_only=True,
        preprocess=True,
        preprocess_skip_symbolic_shape=True,
    )

    preprocess_input = Path(calls["preprocess_args"][0])
    if external_data:
        assert preprocess_input.name == "materialized-input.onnx"
    else:
        assert preprocess_input == model_filename
    assert calls["preprocess_kwargs"]["skip_optimization"] is False
    assert calls["preprocess_kwargs"]["skip_onnx_shape"] is False
    assert calls["preprocess_kwargs"]["skip_symbolic_shape"] is True
    assert calls["preprocess_kwargs"]["save_as_external_data"] is external_data
    quantize_kwargs = calls["quantize_kwargs"]
    assert Path(quantize_kwargs["model_input"]).name == "preprocessed.onnx"
    assert quantize_kwargs["op_types_to_quantize"] == ["MatMul", "Conv"]
    assert quantize_kwargs["extra_options"]["MatMulConstBOnly"] is True
    assert report["op_types_to_quantize"] == ["MatMul", "Conv"]
    assert report["matmul_const_b_only"] is True
    assert report["preprocess"] is True

    metadata = {
        item.key: item.value for item in onnx.load(str(output_filename)).metadata_props
    }
    output_model = onnx.load(str(output_filename), load_external_data=False)
    assert all(item.domain in ("", "ai.onnx") for item in output_model.opset_import)
    assert (
        any(
            initializer.data_location == TensorProto.EXTERNAL
            for initializer in output_model.graph.initializer
        )
        is external_data
    )
    assert metadata["icefall.quantization.matmul_const_b_only"] == "1"
    assert metadata["icefall.quantization.preprocess"] == "1"
    assert metadata["icefall.quantization.preprocess_skip_symbolic_shape"] == "1"


def _save_streaming_models(
    model_dir: Path,
    dynamic_batch: bool,
    fixed_batch_size: int = 1,
    integer_type: int = TensorProto.INT64,
):
    batch = "N" if dynamic_batch else fixed_batch_size
    integer_dtype = np.int32 if integer_type == TensorProto.INT32 else np.int64
    encoder_graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["encoder_out"]),
            helper.make_node(
                "Add", ["processed_lens", "state_increment"], ["new_processed_lens"]
            ),
        ],
        "encoder",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3, 2]),
            helper.make_tensor_value_info("processed_lens", integer_type, [batch]),
        ],
        [
            helper.make_tensor_value_info(
                "encoder_out", TensorProto.FLOAT, [batch, 3, 2]
            ),
            helper.make_tensor_value_info("new_processed_lens", integer_type, [batch]),
        ],
        [
            numpy_helper.from_array(
                np.array([2], dtype=integer_dtype), "state_increment"
            )
        ],
    )
    decoder_graph = helper.make_graph(
        [helper.make_node("Cast", ["y"], ["decoder_out"], to=TensorProto.FLOAT)],
        "decoder",
        [helper.make_tensor_value_info("y", integer_type, [batch, 2])],
        [helper.make_tensor_value_info("decoder_out", TensorProto.FLOAT, [batch, 2])],
    )
    joiner_weight = np.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    joiner_graph = helper.make_graph(
        [
            helper.make_node("Add", ["encoder_out", "decoder_out"], ["joiner_sum"]),
            helper.make_node("MatMul", ["joiner_sum", "weight"], ["logit"]),
        ],
        "joiner",
        [
            helper.make_tensor_value_info("encoder_out", TensorProto.FLOAT, [batch, 2]),
            helper.make_tensor_value_info("decoder_out", TensorProto.FLOAT, [batch, 2]),
        ],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, [batch, 3])],
        [numpy_helper.from_array(joiner_weight, "weight")],
    )
    models = {
        "encoder": helper.make_model(
            encoder_graph,
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=9,
        ),
        "decoder": helper.make_model(
            decoder_graph,
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=9,
        ),
        "joiner": helper.make_model(
            joiner_graph,
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=9,
        ),
    }
    metadata = {
        "encoder": {"T": "3", "decode_chunk_len": "2"},
        "decoder": {
            "context_size": "2",
            "blank_id": "0",
            "unk_id": "2",
            "vocab_size": "3",
        },
        "joiner": {"joiner_dim": "2", "vocab_size": "3"},
    }
    filenames = {}
    for component, model in models.items():
        for key, value in metadata[component].items():
            item = model.metadata_props.add()
            item.key = key
            item.value = value
        filename = model_dir / f"{component}.onnx"
        onnx.save_model(model, str(filename))
        filenames[component] = filename
    return filenames


@pytest.mark.parametrize(
    "case,match",
    [
        ("extra_key", "only the key"),
        ("dtype", "float32"),
        ("rank", "expected"),
        ("feature_dim", "expected"),
        ("empty", "empty"),
        ("nonfinite", "NaN/Inf"),
    ],
)
def test_feature_npz_validation(tmp_path: Path, case: str, match: str) -> None:
    filename = tmp_path / "features.npz"
    values = {"features": np.ones((3, 2), dtype=np.float32)}
    if case == "extra_key":
        values["extra"] = np.zeros((1,), dtype=np.float32)
    elif case == "dtype":
        values["features"] = values["features"].astype(np.float64)
    elif case == "rank":
        values["features"] = values["features"][0]
    elif case == "feature_dim":
        values["features"] = np.ones((3, 3), dtype=np.float32)
    elif case == "empty":
        values["features"] = np.empty((0, 2), dtype=np.float32)
    elif case == "nonfinite":
        values["features"][0, 0] = np.nan
    np.savez(filename, **values)

    with pytest.raises(ValueError, match=match):
        _load_feature_array(filename, feature_dim=2)


def test_rejects_fixed_batch_larger_than_one(tmp_path: Path) -> None:
    model_filenames = _save_streaming_models(
        tmp_path, dynamic_batch=False, fixed_batch_size=2
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez(
        feature_dir / "sample.npz",
        features=np.ones((3, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="fixed batch size 1"):
        generate_calibration_bundle(
            model_filenames,
            feature_dir,
            tmp_path / "bundle",
            max_utterances=1,
        )


@pytest.mark.parametrize("dynamic_batch", [False, True])
@pytest.mark.parametrize("integer_type", [TensorProto.INT32, TensorProto.INT64])
def test_generate_streaming_calibration_bundle(
    tmp_path: Path, dynamic_batch: bool, integer_type: int
) -> None:
    model_filenames = _save_streaming_models(
        tmp_path, dynamic_batch, integer_type=integer_type
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez(
        feature_dir / "short.npz",
        features=np.array([[0.2, 0.1]], dtype=np.float32),
    )
    np.savez(
        feature_dir / "long.npz",
        features=np.linspace(-0.5, 0.5, 10, dtype=np.float32).reshape(5, 2),
    )
    bundle_dir = tmp_path / "bundle"

    manifest = generate_calibration_bundle(
        model_filenames=model_filenames,
        feature_dir=feature_dir,
        bundle_dir=bundle_dir,
        max_utterances=2,
        max_chunks=4,
        max_samples_per_model=20,
    )

    assert manifest["complete"] is True
    assert manifest["sample_counts"]["encoder"] == 4
    assert manifest["sample_counts"]["decoder"] > 0
    assert manifest["sample_counts"]["joiner"] > 0
    assert manifest["streaming"]["blank_id"] == 0
    assert manifest["streaming"]["unk_id"] == 2
    expected_dtype = np.dtype(
        np.int32 if integer_type == TensorProto.INT32 else np.int64
    )
    assert manifest["streaming"]["decoder_input_dtype"] == str(expected_dtype)
    assert (
        load_model_io_schema(model_filenames["decoder"])["inputs"][0]["elem_type"]
        == integer_type
    )
    validate_calibration_manifest(
        bundle_dir, model_filenames, calibration_method="percentile"
    )
    for component, filename in model_filenames.items():
        reader = NpzCalibrationDataReader(filename, bundle_dir / component)
        while True:
            values = reader.get_next()
            if values is None:
                break
            if not dynamic_batch or component == "encoder":
                assert next(iter(values.values())).shape[0] == 1
    comparison = compare_streaming_model_sets(
        model_filenames,
        model_filenames,
        bundle_dir,
        max_utterances=2,
        max_chunks=4,
    )
    assert comparison["processed_chunks"] == 4
    assert comparison["token_agreement"] == 1.0
    assert comparison["first_token_divergence"] is None
    assert all(value["max_abs"] == 0 for value in comparison["outputs"].values())
    with pytest.raises(ValueError, match="limit mismatch"):
        validate_calibration_manifest(
            bundle_dir,
            model_filenames,
            calibration_method="percentile",
            max_chunks=5,
        )
    if not dynamic_batch:
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        static_filenames = {}
        for component, filename in model_filenames.items():
            output_filename = static_dir / f"{component}.onnx"
            bundle_hash = calibration_bundle_hash(bundle_dir)
            quantize_static_int8(
                filename,
                output_filename,
                bundle_dir / component,
                calibration_method="minmax",
                calibration_bundle_hash_value=bundle_hash,
                op_types_to_quantize=resolve_static_quantization_op_types(
                    component, "extended"
                ),
            )
            metadata = {
                item.key: item.value
                for item in onnx.load(str(output_filename)).metadata_props
            }
            assert (
                metadata["icefall.quantization.calibration_bundle_sha256"]
                == bundle_hash
            )
            assert metadata["icefall.quantization.op_types"] == ",".join(
                resolve_static_quantization_op_types(component, "extended")
            )
            static_filenames[component] = output_filename
        joiner_model = onnx.load(str(static_filenames["joiner"]))
        producers = {
            output: node for node in joiner_model.graph.node for output in node.output
        }
        joiner_add = next(
            node for node in joiner_model.graph.node if node.op_type == "Add"
        )
        assert all(
            producers[name].op_type == "DequantizeLinear" for name in joiner_add.input
        )
        static_comparison = compare_streaming_model_sets(
            model_filenames,
            static_filenames,
            bundle_dir,
            max_utterances=2,
            max_chunks=4,
        )
        assert static_comparison["processed_chunks"] == 4
        assert static_comparison["token_count"] > 0
    original_bundle_hash = calibration_bundle_hash(bundle_dir)
    sample_filename = bundle_dir / "encoder" / "000000.npz"
    with np.load(sample_filename, allow_pickle=False) as archive:
        sample = {name: np.asarray(archive[name]) for name in archive.files}
    sample["x"] = sample["x"] + 0.01
    np.savez_compressed(sample_filename, **sample)
    assert calibration_bundle_hash(bundle_dir) != original_bundle_hash
    with pytest.raises(ValueError, match="sample identity"):
        validate_calibration_manifest(
            bundle_dir, model_filenames, calibration_method="percentile"
        )


@pytest.mark.parametrize("token", [0, 2])
def test_blank_and_unk_do_not_update_calibration_context(
    tmp_path: Path, token: int
) -> None:
    model_filenames = _save_streaming_models(tmp_path, dynamic_batch=False)
    joiner = onnx.load(str(model_filenames["joiner"]))
    logits = np.full((1, 3), -1.0, dtype=np.float32)
    logits[0, token] = 1.0
    del joiner.graph.node[:]
    del joiner.graph.initializer[:]
    joiner.graph.initializer.append(numpy_helper.from_array(logits, "fixed_logit"))
    joiner.graph.node.append(helper.make_node("Identity", ["fixed_logit"], ["logit"]))
    onnx.save_model(joiner, str(model_filenames["joiner"]))

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez(
        feature_dir / "sample.npz",
        features=np.ones((5, 2), dtype=np.float32),
    )
    bundle_dir = tmp_path / "bundle"
    manifest = generate_calibration_bundle(
        model_filenames,
        feature_dir,
        bundle_dir,
        max_utterances=1,
        max_chunks=3,
        max_samples_per_model=20,
    )

    assert manifest["sample_counts"]["decoder"] == 1
    with np.load(bundle_dir / "decoder" / "000000.npz") as sample:
        np.testing.assert_array_equal(sample["y"], np.array([[-1, 0]], dtype=np.int64))


def test_calibration_bundle_truncation_has_valid_ranges(tmp_path: Path) -> None:
    model_filenames = _save_streaming_models(tmp_path, dynamic_batch=False)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez(
        feature_dir / "long.npz",
        features=np.ones((8, 2), dtype=np.float32),
    )
    bundle_dir = tmp_path / "bundle"

    manifest = generate_calibration_bundle(
        model_filenames,
        feature_dir,
        bundle_dir,
        max_utterances=1,
        max_chunks=10,
        max_samples_per_model=1,
    )

    assert manifest["truncated"] is True
    assert manifest["sample_counts"] == {
        "encoder": 1,
        "decoder": 1,
        "joiner": 1,
    }
    utterance = manifest["utterances"][0]
    assert utterance["encoder_sample_start"] == 0
    assert utterance["encoder_sample_end"] == 1
    comparison = compare_streaming_model_sets(
        model_filenames,
        model_filenames,
        bundle_dir,
        max_utterances=1,
        max_chunks=10,
    )
    assert comparison["processed_chunks"] == 1


def test_streaming_comparison_uses_independent_states(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    reference_filenames = _save_streaming_models(reference_dir, False)
    candidate_filenames = _save_streaming_models(candidate_dir, False)
    candidate_encoder = onnx.load(str(candidate_filenames["encoder"]))
    increment = next(
        item
        for item in candidate_encoder.graph.initializer
        if item.name == "state_increment"
    )
    increment.CopyFrom(
        numpy_helper.from_array(np.array([3], dtype=np.int64), "state_increment")
    )
    onnx.save_model(candidate_encoder, str(candidate_filenames["encoder"]))
    candidate_joiner = onnx.load(str(candidate_filenames["joiner"]))
    weight = next(
        item for item in candidate_joiner.graph.initializer if item.name == "weight"
    )
    weight.CopyFrom(
        numpy_helper.from_array(np.zeros((2, 3), dtype=np.float32), "weight")
    )
    onnx.save_model(candidate_joiner, str(candidate_filenames["joiner"]))

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez(
        feature_dir / "positive.npz",
        features=np.full((5, 2), 0.5, dtype=np.float32),
    )
    bundle_dir = tmp_path / "bundle"
    generate_calibration_bundle(
        reference_filenames,
        feature_dir,
        bundle_dir,
        max_utterances=1,
        max_chunks=3,
        max_samples_per_model=20,
    )

    comparison = compare_streaming_model_sets(
        reference_filenames,
        candidate_filenames,
        bundle_dir,
        max_utterances=1,
        max_chunks=3,
    )

    assert comparison["outputs"]["new_processed_lens"]["max_abs"] > 0
    assert comparison["token_agreement"] < 1
    assert comparison["first_token_divergence"] is not None


def _generate_test_calibration_bundle(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_filenames = _save_streaming_models(model_dir, dynamic_batch=False)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez(
        feature_dir / "b.npz",
        features=np.full((3, 2), 0.5, dtype=np.float32),
    )
    np.savez(
        feature_dir / "a.npz",
        features=np.full((3, 2), 0.25, dtype=np.float32),
    )
    bundle_dir = tmp_path / "calibration"
    generate_calibration_bundle(
        model_filenames,
        feature_dir,
        bundle_dir,
        max_utterances=2,
        max_chunks=10,
        max_samples_per_model=20,
    )
    return model_filenames, feature_dir, bundle_dir


def test_calibration_manifest_identity_and_atomic_commit(tmp_path: Path) -> None:
    model_filenames, feature_dir, bundle_dir = _generate_test_calibration_bundle(
        tmp_path
    )
    manifest = load_calibration_manifest(bundle_dir)
    assert manifest["complete"] is True
    assert [item["filename"] for item in manifest["features"]] == [
        "a.npz",
        "b.npz",
    ]
    assert not list(tmp_path.glob(".calibration.tmp-*"))
    validate_calibration_manifest(
        bundle_dir,
        model_filenames,
        "percentile",
        feature_dir=feature_dir,
        max_utterances=2,
        max_chunks=10,
        max_samples_per_model=20,
    )
    with pytest.raises(ValueError, match="method"):
        validate_calibration_manifest(bundle_dir, model_filenames, "minmax")

    model = onnx.load(str(model_filenames["encoder"]))
    item = model.metadata_props.add()
    item.key = "changed"
    item.value = "1"
    onnx.save_model(model, str(model_filenames["encoder"]))
    with pytest.raises(ValueError, match="identity"):
        validate_calibration_manifest(bundle_dir, model_filenames, "percentile")


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("missing_component", "models must contain"),
        ("extra_component", "samples must contain"),
        ("invalid_limit", "Invalid calibration limit"),
        ("sample_gap", "sample sequence mismatch"),
        ("range_overlap", "Invalid calibration sample range"),
        ("range_out_of_bounds", "Invalid calibration sample range"),
        ("total_chunks", "total_chunks"),
    ],
)
def test_calibration_manifest_rejects_internal_corruption(
    tmp_path: Path, corruption: str, match: str
) -> None:
    _, _, bundle_dir = _generate_test_calibration_bundle(tmp_path)
    manifest_filename = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_filename.read_text())
    if corruption == "missing_component":
        manifest["models"].pop("joiner")
    elif corruption == "extra_component":
        manifest["samples"]["extra"] = []
    elif corruption == "invalid_limit":
        manifest["limits"]["max_chunks"] = 0
    elif corruption == "sample_gap":
        manifest["samples"]["encoder"][0]["filename"] = "000001.npz"
    elif corruption == "range_overlap":
        manifest["utterances"][1]["encoder_sample_start"] = 0
    elif corruption == "range_out_of_bounds":
        manifest["utterances"][-1]["encoder_sample_end"] += 1
    elif corruption == "total_chunks":
        manifest["total_chunks"] += 1
    manifest_filename.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        load_calibration_manifest(bundle_dir)


def test_failed_calibration_generation_removes_temporary_bundle(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_filenames = _save_streaming_models(model_dir, dynamic_batch=False)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez(
        feature_dir / "a-valid.npz",
        features=np.ones((3, 2), dtype=np.float32),
    )
    np.savez(
        feature_dir / "b-invalid.npz",
        features=np.empty((0, 2), dtype=np.float32),
    )
    bundle_dir = tmp_path / "calibration"

    with pytest.raises(ValueError, match="empty"):
        generate_calibration_bundle(
            model_filenames,
            feature_dir,
            bundle_dir,
            max_utterances=2,
            max_chunks=10,
            max_samples_per_model=20,
        )
    assert not bundle_dir.exists()
    assert not list(tmp_path.glob(".calibration.tmp-*"))
