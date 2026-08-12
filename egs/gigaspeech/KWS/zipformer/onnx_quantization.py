import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import _get_all_tensors
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

PathLike = Union[str, Path]
STANDARD_ONNX_DOMAINS = {"", "ai.onnx"}
CALIBRATION_COMPONENTS = ("encoder", "decoder", "joiner")


def resolve_quantization_mode(
    quantization_mode: Optional[str], enable_int8_quantization: int
) -> str:
    if quantization_mode is None:
        return "dynamic" if enable_int8_quantization else "none"
    return quantization_mode


def _unique_name(base: str, used_names: Set[str]) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    candidate = name
    index = 1
    while candidate in used_names:
        candidate = f"{name}_{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def _value_info_schema(value_info: onnx.ValueInfoProto) -> Dict[str, object]:
    tensor_type = value_info.type.tensor_type
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return {
        "name": value_info.name,
        "elem_type": tensor_type.elem_type,
        "shape": shape,
    }


def model_io_schema(model: onnx.ModelProto) -> Dict[str, List[Dict[str, object]]]:
    return {
        "inputs": [_value_info_schema(value) for value in model.graph.input],
        "outputs": [_value_info_schema(value) for value in model.graph.output],
    }


def load_model_io_schema(filename: PathLike) -> Dict[str, List[Dict[str, object]]]:
    model = onnx.load(str(filename), load_external_data=False)
    return model_io_schema(model)


def set_model_metadata(model: onnx.ModelProto, values: Dict[str, object]) -> None:
    properties = {item.key: item for item in model.metadata_props}
    for key, value in values.items():
        text = str(value)
        if key in properties:
            properties[key].value = text
        else:
            item = model.metadata_props.add()
            item.key = key
            item.value = text


def _external_locations(model: onnx.ModelProto) -> List[str]:
    locations = set()
    for tensor in _get_all_tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        external_data = {item.key: item.value for item in tensor.external_data}
        location = external_data.get("location")
        if location:
            locations.add(location)
    return sorted(locations)


def _update_digest_from_file(digest, filename: Path) -> None:
    with filename.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                return
            digest.update(chunk)


def file_hash(filename: PathLike) -> str:
    digest = hashlib.sha256()
    _update_digest_from_file(digest, Path(filename))
    return digest.hexdigest()


def model_artifact_hash(filename: PathLike) -> str:
    filename = Path(filename)
    model = onnx.load(str(filename), load_external_data=False)
    digest = hashlib.sha256()
    _update_digest_from_file(digest, filename)
    for location in _external_locations(model):
        external_filename = filename.parent / location
        digest.update(location.encode("utf-8"))
        _update_digest_from_file(digest, external_filename)
    return digest.hexdigest()


def _save_model(
    model: onnx.ModelProto,
    filename: Path,
    use_external_data: bool,
) -> None:
    if use_external_data:
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
    onnx.checker.check_model(str(filename))


def _model_metadata_values(model: onnx.ModelProto) -> Dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def _custom_node_fingerprints(
    graph: onnx.GraphProto,
    graph_path: str = "graph",
) -> List[str]:
    fingerprints = []
    for node in graph.node:
        if node.domain not in STANDARD_ONNX_DOMAINS:
            digest = hashlib.sha256()
            digest.update(graph_path.encode("utf-8"))
            digest.update(node.SerializeToString(deterministic=True))
            fingerprints.append(digest.hexdigest())
        node_id = node.name or f"{node.domain}:{node.op_type}:{','.join(node.output)}"
        for attribute in node.attribute:
            attribute_path = f"{graph_path}/{node_id}:{attribute.name}"
            if attribute.type == onnx.AttributeProto.GRAPH:
                fingerprints.extend(
                    _custom_node_fingerprints(attribute.g, attribute_path)
                )
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for index, subgraph in enumerate(attribute.graphs):
                    fingerprints.extend(
                        _custom_node_fingerprints(
                            subgraph,
                            f"{attribute_path}[{index}]",
                        )
                    )
    return fingerprints


def _custom_domain_fingerprint(model: onnx.ModelProto) -> Dict[str, object]:
    return {
        "nodes": _custom_node_fingerprints(model.graph),
        "opsets": sorted(
            (item.domain, item.version)
            for item in model.opset_import
            if item.domain not in STANDARD_ONNX_DOMAINS
        ),
        "functions": sorted(
            hashlib.sha256(function.SerializeToString(deterministic=True)).hexdigest()
            for function in model.functions
        ),
    }


def validate_quantized_model_artifact(
    model_input: PathLike,
    model_output: PathLike,
    required_metadata: Dict[str, object],
) -> None:
    model_input = Path(model_input)
    model_output = Path(model_output)
    source = onnx.load(str(model_input), load_external_data=False)
    candidate = onnx.load(str(model_output), load_external_data=False)
    if model_io_schema(candidate) != model_io_schema(source):
        raise ValueError("Quantization changed the model I/O schema")

    source_metadata = _model_metadata_values(source)
    candidate_metadata = _model_metadata_values(candidate)
    for key, value in source_metadata.items():
        if candidate_metadata.get(key) != value:
            raise ValueError(f"Quantization changed model metadata: {key}")
    for key, value in required_metadata.items():
        if candidate_metadata.get(key) != str(value):
            raise ValueError(f"Missing quantization metadata: {key}")

    if _custom_domain_fingerprint(candidate) != _custom_domain_fingerprint(source):
        raise ValueError("Quantization changed non-standard ONNX content")
    for location in _external_locations(candidate):
        if not (model_output.parent / location).is_file():
            raise ValueError(f"Missing external ONNX data file: {location}")

    onnx.checker.check_model(str(model_output))
    ort.InferenceSession(str(model_output), providers=["CPUExecutionProvider"])


def _quantize_per_channel(
    weight: np.ndarray,
    axis: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduce_axes = tuple(index for index in range(weight.ndim) if index != axis)
    max_abs = np.max(np.abs(weight.astype(np.float32)), axis=reduce_axes)
    scales = max_abs / 127.0
    scales = np.where(scales == 0, 1.0, scales).astype(weight.dtype)
    reshape = [1] * weight.ndim
    reshape[axis] = weight.shape[axis]
    quantized = np.rint(weight / scales.reshape(reshape))
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    zero_points = np.zeros(scales.shape, dtype=np.int8)
    return quantized, scales, zero_points


def quantize_weight_only_int8(
    model_input: PathLike,
    model_output: PathLike,
    use_external_data: bool = False,
    metadata: Optional[Dict[str, object]] = None,
    op_types: Sequence[str] = ("MatMul", "Conv"),
) -> Dict[str, object]:
    model_input = Path(model_input)
    model_output = Path(model_output)
    model = onnx.load(str(model_input), load_external_data=True)
    default_opset = next(
        (item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        None,
    )
    if default_opset is None or default_opset < 13:
        raise ValueError(
            "Per-channel DequantizeLinear requires the default ONNX opset to be >= 13"
        )
    original_schema = model_io_schema(model)
    initializers = {item.name: item for item in model.graph.initializer}
    graph_inputs = {item.name for item in model.graph.input}
    used_names = {
        name
        for node in model.graph.node
        for name in list(node.input) + list(node.output) + [node.name]
        if name
    }
    used_names.update(initializers)
    quantized_weights = {}
    dequantize_nodes = []
    skipped = []

    for node in model.graph.node:
        if (
            node.domain not in STANDARD_ONNX_DOMAINS
            or node.op_type not in op_types
            or len(node.input) < 2
        ):
            continue
        weight_name = node.input[1]
        initializer = initializers.get(weight_name)
        if initializer is None:
            skipped.append((node.name, weight_name, "non-constant weight"))
            continue
        if weight_name in graph_inputs:
            skipped.append((node.name, weight_name, "initializer is a graph input"))
            continue
        if initializer.data_type != TensorProto.FLOAT:
            skipped.append((node.name, weight_name, "weight is not float32"))
            continue
        weight = numpy_helper.to_array(initializer)
        if not np.isfinite(weight).all():
            skipped.append((node.name, weight_name, "weight contains NaN/Inf"))
            continue
        axis = 1 if node.op_type == "MatMul" else 0
        valid_rank = weight.ndim == 2 if node.op_type == "MatMul" else weight.ndim >= 3
        if not valid_rank:
            skipped.append((node.name, weight_name, f"weight rank is {weight.ndim}"))
            continue
        cache_key = (weight_name, axis)
        if cache_key not in quantized_weights:
            quantized, scales, zero_points = _quantize_per_channel(weight, axis)
            quantized_name = _unique_name(f"{weight_name}_weight_only_int8", used_names)
            scale_name = _unique_name(f"{weight_name}_weight_only_scale", used_names)
            zero_point_name = _unique_name(
                f"{weight_name}_weight_only_zero_point", used_names
            )
            dequantized_name = _unique_name(
                f"{weight_name}_weight_only_dequantized", used_names
            )
            node_name = _unique_name(
                f"{weight_name}_weight_only_dequantize", used_names
            )
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(quantized, quantized_name),
                    numpy_helper.from_array(scales, scale_name),
                    numpy_helper.from_array(zero_points, zero_point_name),
                ]
            )
            dequantize_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [quantized_name, scale_name, zero_point_name],
                    [dequantized_name],
                    name=node_name,
                    axis=axis,
                )
            )
            quantized_weights[cache_key] = dequantized_name
        node.input[1] = quantized_weights[cache_key]

    if not quantized_weights:
        details = "; ".join(
            f"node={node_name or '<unnamed>'} weight={weight_name}: {reason}"
            for node_name, weight_name, reason in skipped
        )
        suffix = f"; skipped: {details}" if details else ""
        raise ValueError(
            f"No eligible MatMul/Conv weights found in {model_input}{suffix}"
        )

    remaining_inputs = {name for node in model.graph.node for name in node.input}
    removable_names = {
        weight_name
        for weight_name, _ in quantized_weights
        if weight_name not in remaining_inputs
    }
    kept_initializers = [
        item for item in model.graph.initializer if item.name not in removable_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(dequantize_nodes + original_nodes)

    quantization_metadata = {
        "icefall.quantization.mode": "weight_only_int8",
        "icefall.quantization.format": "standard_onnx_weight_dq",
        "icefall.quantization.weight_type": "QInt8",
        "icefall.quantization.per_channel": "1",
        "icefall.quantization.op_types": ",".join(op_types),
    }
    if metadata:
        quantization_metadata.update(metadata)
    set_model_metadata(model, quantization_metadata)

    if model_io_schema(model) != original_schema:
        raise ValueError("Weight-only quantization changed the model I/O schema")

    _save_model(model, model_output, use_external_data)
    validate_quantized_model_artifact(
        model_input,
        model_output,
        quantization_metadata,
    )
    for node_name, weight_name, reason in skipped:
        logging.warning(
            "Skip weight-only quantization for node %s weight %s: %s",
            node_name,
            weight_name,
            reason,
        )
    return {
        "model_input": str(model_input),
        "model_output": str(model_output),
        "quantized_weight_count": len(quantized_weights),
        "skipped": skipped,
        "io_schema": original_schema,
    }


CALIBRATION_SCHEMA_VERSION = 1
ORT_TYPE_TO_DTYPE = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(bool)": np.dtype(np.bool_),
}


def _normalize_ort_shape(shape: Sequence[object]) -> List[object]:
    return [value if isinstance(value, (int, str)) else None for value in shape]


def ort_session_schema(session: ort.InferenceSession) -> Dict[str, object]:
    def convert(node) -> Dict[str, object]:
        return {
            "name": node.name,
            "type": node.type,
            "shape": _normalize_ort_shape(node.shape),
        }

    return {
        "inputs": [convert(node) for node in session.get_inputs()],
        "outputs": [convert(node) for node in session.get_outputs()],
    }


def model_manifest_record(filename: PathLike) -> Dict[str, object]:
    filename = Path(filename)
    session = ort.InferenceSession(str(filename), providers=["CPUExecutionProvider"])
    return {
        "filename": filename.name,
        "sha256": model_artifact_hash(filename),
        "onnx_schema": load_model_io_schema(filename),
        "ort_schema": ort_session_schema(session),
        "external_data": _external_locations(
            onnx.load(str(filename), load_external_data=False)
        ),
    }


def feature_manifest_records(
    feature_dir: PathLike,
    max_utterances: int,
) -> List[Dict[str, object]]:
    feature_dir = Path(feature_dir)
    filenames = sorted(feature_dir.glob("*.npz"))
    if max_utterances <= 0:
        raise ValueError("calibration max utterances must be positive")
    filenames = filenames[:max_utterances]
    if not filenames:
        raise ValueError(f"No feature NPZ files found in {feature_dir}")
    return [
        {
            "filename": filename.name,
            "size": filename.stat().st_size,
            "sha256": file_hash(filename),
        }
        for filename in filenames
    ]


def _sample_manifest_records(sample_dir: Path) -> List[Dict[str, object]]:
    return [
        {
            "filename": filename.name,
            "size": filename.stat().st_size,
            "sha256": file_hash(filename),
        }
        for filename in sorted(sample_dir.glob("*.npz"))
    ]


def _validate_file_records(records: object, field: str) -> List[str]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"Calibration manifest has no {field} records")
    filenames = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid calibration {field} record")
        filename = record.get("filename")
        size = record.get("size")
        sha256 = record.get("sha256")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Invalid calibration {field} filename")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid calibration {field} file size")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"Invalid calibration {field} SHA256")
        filenames.append(filename)
    return filenames


def _validate_calibration_manifest_structure(manifest: Dict[str, object]) -> None:
    if manifest.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration manifest schema version")
    if manifest.get("complete") is not True:
        raise ValueError("Calibration bundle is incomplete")
    if manifest.get("calibration_method") not in ("percentile", "minmax"):
        raise ValueError("Invalid calibration method in bundle manifest")
    if not isinstance(manifest.get("truncated"), bool):
        raise ValueError("Calibration manifest has invalid truncation status")

    required_components = set(CALIBRATION_COMPONENTS)
    for field in ("models", "samples", "sample_counts"):
        values = manifest.get(field)
        if not isinstance(values, dict) or set(values) != required_components:
            raise ValueError(
                f"Calibration manifest {field} must contain "
                f"{sorted(required_components)}"
            )
    for component in CALIBRATION_COMPONENTS:
        record = manifest["models"][component]
        if not isinstance(record, dict):
            raise ValueError(f"Invalid calibration model record for {component}")
        required_model_fields = {
            "filename",
            "sha256",
            "onnx_schema",
            "ort_schema",
            "external_data",
        }
        if set(record) != required_model_fields:
            raise ValueError(f"Incomplete calibration model record for {component}")

    sample_counts = manifest["sample_counts"]
    for component in CALIBRATION_COMPONENTS:
        count = sample_counts[component]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Invalid calibration sample count for {component}")
        filenames = _validate_file_records(
            manifest["samples"][component], f"{component} sample"
        )
        expected_filenames = [f"{index:06d}.npz" for index in range(count)]
        if len(filenames) != count or filenames != expected_filenames:
            raise ValueError(f"Calibration sample sequence mismatch for {component}")

    limits = manifest.get("limits")
    required_limits = {
        "max_utterances",
        "max_chunks",
        "max_samples_per_model",
    }
    if not isinstance(limits, dict) or set(limits) != required_limits:
        raise ValueError("Calibration manifest has invalid limits")
    for name in required_limits:
        value = limits[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid calibration limit: {name}")

    features = manifest.get("features")
    feature_filenames = _validate_file_records(features, "feature")
    if feature_filenames != sorted(feature_filenames):
        raise ValueError("Calibration features are not sorted")
    if len(set(feature_filenames)) != len(feature_filenames):
        raise ValueError("Calibration feature filenames are not unique")
    if len(feature_filenames) > limits["max_utterances"]:
        raise ValueError("Calibration feature count exceeds max_utterances")

    utterances = manifest.get("utterances")
    if not isinstance(utterances, list) or len(utterances) != len(features):
        raise ValueError("Calibration feature and utterance records differ")
    previous_ends = {component: 0 for component in CALIBRATION_COMPONENTS}
    for feature_record, utterance in zip(features, utterances):
        if not isinstance(utterance, dict):
            raise ValueError("Invalid calibration utterance record")
        if utterance.get("filename") != feature_record["filename"]:
            raise ValueError("Calibration feature and utterance order differ")
        for component in CALIBRATION_COMPONENTS:
            start = utterance.get(f"{component}_sample_start")
            end = utterance.get(f"{component}_sample_end")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start != previous_ends[component]
                or end < start
                or end > sample_counts[component]
            ):
                raise ValueError(f"Invalid calibration sample range for {component}")
            previous_ends[component] = end
    for component in CALIBRATION_COMPONENTS:
        if previous_ends[component] != sample_counts[component]:
            raise ValueError(f"Incomplete calibration sample ranges for {component}")

    total_chunks = manifest.get("total_chunks")
    if total_chunks != sample_counts["encoder"]:
        raise ValueError("Calibration total_chunks does not match encoder samples")
    if total_chunks > limits["max_chunks"]:
        raise ValueError("Calibration chunk count exceeds max_chunks")
    for component in CALIBRATION_COMPONENTS:
        if sample_counts[component] > limits["max_samples_per_model"]:
            raise ValueError(f"Calibration sample count exceeds limit for {component}")
    if manifest["truncated"] and not (
        total_chunks >= limits["max_chunks"]
        or any(
            sample_counts[component] >= limits["max_samples_per_model"]
            for component in CALIBRATION_COMPONENTS
        )
    ):
        raise ValueError("Calibration manifest has inconsistent truncation status")

    streaming = manifest.get("streaming")
    required_streaming = {
        "segment",
        "offset",
        "feature_dim",
        "blank_id",
        "unk_id",
        "vocab_size",
        "context_size",
        "joiner_dim",
        "decoder_input_dtype",
        "sample_batch_size",
        "dynamic_batch",
    }
    if not isinstance(streaming, dict) or set(streaming) != required_streaming:
        raise ValueError("Calibration manifest has invalid streaming metadata")
    for name in (
        "segment",
        "offset",
        "feature_dim",
        "vocab_size",
        "context_size",
        "joiner_dim",
    ):
        value = streaming[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid calibration streaming value: {name}")
    if streaming["offset"] > streaming["segment"]:
        raise ValueError("Calibration offset exceeds segment length")
    for name in ("blank_id", "unk_id"):
        value = streaming[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= streaming["vocab_size"]
        ):
            raise ValueError(f"Invalid calibration streaming value: {name}")
    if not isinstance(streaming["decoder_input_dtype"], str):
        raise ValueError("Invalid calibration decoder input dtype")
    if streaming["sample_batch_size"] != 1:
        raise ValueError("Calibration sample batch size must be 1")
    dynamic_batch = streaming["dynamic_batch"]
    if (
        not isinstance(dynamic_batch, dict)
        or set(dynamic_batch) != required_components
        or not all(isinstance(value, bool) for value in dynamic_batch.values())
    ):
        raise ValueError("Invalid calibration dynamic batch metadata")


def calibration_bundle_hash(bundle_dir: PathLike) -> str:
    bundle_dir = Path(bundle_dir)
    load_calibration_manifest(bundle_dir)
    filenames = [bundle_dir / "manifest.json"]
    for component in CALIBRATION_COMPONENTS:
        filenames.extend(sorted((bundle_dir / component).glob("*.npz")))
    digest = hashlib.sha256()
    for filename in filenames:
        relative_name = filename.relative_to(bundle_dir).as_posix()
        digest.update(relative_name.encode("utf-8"))
        _update_digest_from_file(digest, filename)
    return digest.hexdigest()


def write_calibration_manifest(
    bundle_dir: PathLike,
    manifest: Dict[str, object],
) -> str:
    bundle_dir = Path(bundle_dir)
    manifest = dict(manifest)
    manifest["samples"] = {
        component: _sample_manifest_records(bundle_dir / component)
        for component in CALIBRATION_COMPONENTS
    }
    manifest["sample_counts"] = {
        component: len(records) for component, records in manifest["samples"].items()
    }
    manifest["schema_version"] = CALIBRATION_SCHEMA_VERSION
    manifest["complete"] = True
    _validate_calibration_manifest_structure(manifest)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    filename = bundle_dir / "manifest.json"
    filename.write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_calibration_manifest(bundle_dir: PathLike) -> Dict[str, object]:
    filename = Path(bundle_dir) / "manifest.json"
    if not filename.is_file():
        raise ValueError(f"Missing calibration manifest: {filename}")
    manifest = json.loads(filename.read_text())
    _validate_calibration_manifest_structure(manifest)
    return manifest


def validate_calibration_manifest(
    bundle_dir: PathLike,
    model_filenames: Dict[str, PathLike],
    calibration_method: str,
    feature_dir: Optional[PathLike] = None,
    max_utterances: Optional[int] = None,
    max_chunks: Optional[int] = None,
    max_samples_per_model: Optional[int] = None,
) -> Dict[str, object]:
    bundle_dir = Path(bundle_dir)
    if set(model_filenames) != set(CALIBRATION_COMPONENTS):
        raise ValueError(
            f"model_filenames must contain {sorted(CALIBRATION_COMPONENTS)}"
        )
    manifest = load_calibration_manifest(bundle_dir)
    if manifest.get("calibration_method") != calibration_method:
        raise ValueError("Calibration method does not match the bundle manifest")
    limits = manifest.get("limits", {})
    expected_limits = {
        "max_utterances": max_utterances,
        "max_chunks": max_chunks,
        "max_samples_per_model": max_samples_per_model,
    }
    for name, value in expected_limits.items():
        if value is not None and limits.get(name) != value:
            raise ValueError(f"Calibration limit mismatch for {name}")
    model_records = manifest.get("models")
    if not isinstance(model_records, dict):
        raise ValueError("Calibration manifest has no model records")
    sample_counts = manifest.get("sample_counts")
    sample_records = manifest.get("samples")
    if not isinstance(sample_counts, dict):
        raise ValueError("Calibration manifest has no sample counts")
    if not isinstance(sample_records, dict):
        raise ValueError("Calibration manifest has no sample hashes")
    for component, filename in model_filenames.items():
        expected = model_manifest_record(filename)
        if model_records.get(component) != expected:
            raise ValueError(f"Calibration model identity mismatch for {component}")
        sample_dir = bundle_dir / component
        actual_records = _sample_manifest_records(sample_dir)
        actual_count = len(actual_records)
        if actual_count <= 0 or sample_counts.get(component) != actual_count:
            raise ValueError(f"Calibration sample count mismatch for {component}")
        if sample_records.get(component) != actual_records:
            raise ValueError(f"Calibration sample identity mismatch for {component}")
    if feature_dir is not None:
        if max_utterances is None:
            max_utterances = len(manifest.get("features", []))
        expected_features = feature_manifest_records(feature_dir, max_utterances)
        if (
            manifest.get("features")
            != expected_features[: len(manifest.get("features", []))]
        ):
            raise ValueError("Calibration feature identity mismatch")
    return manifest


def create_calibration_temp_dir(bundle_dir: PathLike) -> Path:
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        raise FileExistsError(f"Calibration bundle already exists: {bundle_dir}")
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_dir.name}.tmp-",
            dir=str(bundle_dir.parent),
        )
    )


def commit_calibration_bundle(temp_dir: PathLike, bundle_dir: PathLike) -> None:
    temp_dir = Path(temp_dir)
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        raise FileExistsError(f"Calibration bundle already exists: {bundle_dir}")
    os.replace(str(temp_dir), str(bundle_dir))


def discard_calibration_temp_dir(temp_dir: PathLike) -> None:
    temp_dir = Path(temp_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class NpzCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        model_filename: PathLike,
        sample_dir: PathLike,
        max_samples: Optional[int] = None,
    ):
        self.model_filename = Path(model_filename)
        self.sample_dir = Path(sample_dir)
        self.session = ort.InferenceSession(
            str(self.model_filename), providers=["CPUExecutionProvider"]
        )
        self.inputs = self.session.get_inputs()
        self.files = sorted(self.sample_dir.glob("*.npz"))
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.files = self.files[:max_samples]
        if not self.files:
            raise ValueError(f"No calibration NPZ files found in {self.sample_dir}")
        for filename in self.files:
            self._load_and_validate(filename)
        self.rewind()

    def _load_and_validate(self, filename: Path) -> Dict[str, np.ndarray]:
        with np.load(filename, allow_pickle=False) as archive:
            actual_names = set(archive.files)
            expected_names = {item.name for item in self.inputs}
            if actual_names != expected_names:
                raise ValueError(
                    f"Input names in {filename} do not match {self.model_filename}: "
                    f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
                )
            values = {name: np.asarray(archive[name]) for name in archive.files}
        for item in self.inputs:
            value = values[item.name]
            expected_dtype = ORT_TYPE_TO_DTYPE.get(item.type)
            if expected_dtype is None:
                raise ValueError(f"Unsupported ONNX input type: {item.type}")
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"Input {item.name} in {filename} has dtype {value.dtype}, "
                    f"expected {expected_dtype}"
                )
            expected_shape = item.shape
            if value.ndim != len(expected_shape):
                raise ValueError(
                    f"Input {item.name} in {filename} has rank {value.ndim}, "
                    f"expected {len(expected_shape)}"
                )
            for index, expected_dim in enumerate(expected_shape):
                if isinstance(expected_dim, int) and value.shape[index] != expected_dim:
                    raise ValueError(
                        f"Input {item.name} in {filename} has shape {value.shape}, "
                        f"expected dimension {index} to be {expected_dim}"
                    )
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"Input {item.name} in {filename} contains NaN/Inf")
        return values

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.index == len(self.files):
            return None
        values = self._load_and_validate(self.files[self.index])
        self.index += 1
        return values

    def rewind(self) -> None:
        self.index = 0


def _node_arg_dtype(node) -> np.dtype:
    dtype = ORT_TYPE_TO_DTYPE.get(node.type)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX input type: {node.type}")
    return dtype


def _zero_input_from_schema(node) -> np.ndarray:
    shape = []
    for dim in node.shape:
        if isinstance(dim, int):
            if dim < 0:
                raise ValueError(f"Invalid input shape for {node.name}: {node.shape}")
            shape.append(dim)
        else:
            shape.append(1)
    return np.zeros(shape, dtype=_node_arg_dtype(node))


def _has_dynamic_leading_batch(session: ort.InferenceSession) -> bool:
    first_input = session.get_inputs()[0]
    if not first_input.shape:
        raise ValueError(f"Model input {first_input.name} has no batch dimension")
    batch_dim = first_input.shape[0]
    if isinstance(batch_dim, int):
        if batch_dim != 1:
            raise ValueError(f"Only fixed batch size 1 is supported, got {batch_dim}")
        return False
    return True


def _save_npz_sample(
    sample_dir: Path,
    index: int,
    values: Dict[str, np.ndarray],
) -> None:
    np.savez_compressed(sample_dir / f"{index:06d}.npz", **values)


def _load_feature_array(filename: Path, feature_dim: int) -> np.ndarray:
    with np.load(filename, allow_pickle=False) as archive:
        if set(archive.files) != {"features"}:
            raise ValueError(f"{filename} must contain only the key 'features'")
        features = np.asarray(archive["features"])
    if features.dtype != np.float32:
        raise ValueError(f"Features in {filename} must be float32")
    if features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError(
            f"Features in {filename} have shape {features.shape}, "
            f"expected [T, {feature_dim}]"
        )
    if features.shape[0] == 0:
        raise ValueError(f"Features in {filename} are empty")
    if not np.isfinite(features).all():
        raise ValueError(f"Features in {filename} contain NaN/Inf")
    return features


def _model_metadata(session: ort.InferenceSession) -> Dict[str, str]:
    return session.get_modelmeta().custom_metadata_map


def _run_session(
    session: ort.InferenceSession,
    values: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    output_names = [item.name for item in session.get_outputs()]
    outputs = session.run(output_names, values)
    return dict(zip(output_names, outputs))


def generate_calibration_bundle(
    model_filenames: Dict[str, PathLike],
    feature_dir: PathLike,
    bundle_dir: PathLike,
    calibration_method: str = "percentile",
    max_utterances: int = 100,
    max_chunks: int = 1000,
    max_samples_per_model: int = 1000,
) -> Dict[str, object]:
    required_components = {"encoder", "decoder", "joiner"}
    if set(model_filenames) != required_components:
        raise ValueError(f"model_filenames must contain {sorted(required_components)}")
    if calibration_method not in ("percentile", "minmax"):
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    if max_chunks <= 0 or max_samples_per_model <= 0:
        raise ValueError("Calibration chunk/sample limits must be positive")

    model_filenames = {
        component: Path(filename) for component, filename in model_filenames.items()
    }
    sessions = {
        component: ort.InferenceSession(
            str(filename), providers=["CPUExecutionProvider"]
        )
        for component, filename in model_filenames.items()
    }
    encoder_session = sessions["encoder"]
    decoder_session = sessions["decoder"]
    joiner_session = sessions["joiner"]
    encoder_inputs = {item.name: item for item in encoder_session.get_inputs()}
    if "x" not in encoder_inputs:
        raise ValueError("Encoder model has no input named 'x'")
    x_schema = encoder_inputs["x"]
    if len(x_schema.shape) != 3:
        raise ValueError(f"Unexpected encoder x shape: {x_schema.shape}")
    encoder_metadata = _model_metadata(encoder_session)
    decoder_metadata = _model_metadata(decoder_session)
    joiner_metadata = _model_metadata(joiner_session)
    segment = int(encoder_metadata["T"])
    offset = int(encoder_metadata["decode_chunk_len"])
    if segment <= 0 or offset <= 0:
        raise ValueError(f"Invalid streaming segment/offset: {segment}/{offset}")
    if isinstance(x_schema.shape[1], int) and x_schema.shape[1] != segment:
        raise ValueError("Encoder metadata T does not match the x input shape")
    feature_dim = x_schema.shape[2]
    if not isinstance(feature_dim, int):
        raise ValueError("Encoder feature dimension must be fixed")
    context_size = int(decoder_metadata["context_size"])
    blank_id = int(decoder_metadata["blank_id"])
    unk_id = int(decoder_metadata["unk_id"])
    vocab_size = int(decoder_metadata["vocab_size"])
    joiner_vocab_size = int(joiner_metadata["vocab_size"])
    if joiner_vocab_size != vocab_size:
        raise ValueError("Decoder and joiner vocab sizes differ")
    joiner_dim = int(joiner_metadata["joiner_dim"])
    joiner_output_dim = joiner_session.get_outputs()[0].shape[-1]
    if isinstance(joiner_output_dim, int) and joiner_output_dim != vocab_size:
        raise ValueError("Joiner output dimension does not match vocab size")
    decoder_input_schema = decoder_session.get_inputs()[0]
    decoder_input_dtype = _node_arg_dtype(decoder_input_schema)
    encoder_dynamic_batch = _has_dynamic_leading_batch(encoder_session)
    decoder_dynamic_batch = _has_dynamic_leading_batch(decoder_session)
    joiner_dynamic_batch = _has_dynamic_leading_batch(joiner_session)

    feature_records = feature_manifest_records(feature_dir, max_utterances)
    feature_dir = Path(feature_dir)
    bundle_dir = Path(bundle_dir)
    temp_dir = create_calibration_temp_dir(bundle_dir)
    sample_dirs = {component: temp_dir / component for component in required_components}
    for sample_dir in sample_dirs.values():
        sample_dir.mkdir()

    sample_counts = {component: 0 for component in required_components}
    total_chunks = 0
    used_feature_records = []
    utterance_records = []
    truncated = False
    try:
        for feature_record in feature_records:
            if total_chunks >= min(max_chunks, max_samples_per_model):
                truncated = True
                break
            feature_filename = feature_dir / feature_record["filename"]
            features = _load_feature_array(feature_filename, feature_dim)
            used_feature_records.append(feature_record)
            utterance_records.append(
                {
                    "filename": feature_record["filename"],
                    "encoder_sample_start": sample_counts["encoder"],
                    "decoder_sample_start": sample_counts["decoder"],
                    "joiner_sample_start": sample_counts["joiner"],
                }
            )
            encoder_states = {
                name: _zero_input_from_schema(item)
                for name, item in encoder_inputs.items()
                if name != "x"
            }
            context = np.asarray(
                [[-1] * (context_size - 1) + [blank_id]],
                dtype=decoder_input_dtype,
            )
            decoder_records = []
            decoder_values = {decoder_input_schema.name: context.copy()}
            decoder_records.append(decoder_values)
            decoder_out = _run_session(decoder_session, decoder_values)["decoder_out"]
            start = 0
            while start < features.shape[0]:
                if total_chunks >= min(max_chunks, max_samples_per_model):
                    truncated = True
                    break
                chunk = features[start : start + segment]
                if chunk.shape[0] < segment:
                    chunk = np.pad(
                        chunk,
                        ((0, segment - chunk.shape[0]), (0, 0)),
                        constant_values=np.log(1e-10),
                    ).astype(np.float32)
                encoder_values = {"x": chunk[np.newaxis, ...], **encoder_states}
                _save_npz_sample(
                    sample_dirs["encoder"],
                    sample_counts["encoder"],
                    encoder_values,
                )
                sample_counts["encoder"] += 1
                total_chunks += 1
                encoder_outputs = _run_session(encoder_session, encoder_values)
                encoder_out = encoder_outputs["encoder_out"]
                next_states = {}
                for name in encoder_states:
                    output_name = f"new_{name}"
                    if output_name not in encoder_outputs:
                        raise ValueError(f"Encoder has no output named {output_name}")
                    next_states[name] = encoder_outputs[output_name]
                encoder_states = next_states

                joiner_records = []
                for frame_index in range(encoder_out.shape[1]):
                    joiner_values = {
                        "encoder_out": encoder_out[:, frame_index, :],
                        "decoder_out": decoder_out,
                    }
                    joiner_records.append(joiner_values)
                    logits = _run_session(joiner_session, joiner_values)["logit"]
                    token = int(np.argmax(logits[0]))
                    if token not in (blank_id, unk_id):
                        context = np.concatenate(
                            [
                                context[:, 1:],
                                np.asarray([[token]], dtype=context.dtype),
                            ],
                            axis=1,
                        )
                        decoder_values = {decoder_input_schema.name: context.copy()}
                        decoder_records.append(decoder_values)
                        decoder_out = _run_session(decoder_session, decoder_values)[
                            "decoder_out"
                        ]

                if decoder_records:
                    for values in decoder_records:
                        if sample_counts["decoder"] >= max_samples_per_model:
                            truncated = True
                            break
                        _save_npz_sample(
                            sample_dirs["decoder"],
                            sample_counts["decoder"],
                            values,
                        )
                        sample_counts["decoder"] += 1
                    decoder_records = []

                if joiner_records:
                    for values in joiner_records:
                        if sample_counts["joiner"] >= max_samples_per_model:
                            truncated = True
                            break
                        _save_npz_sample(
                            sample_dirs["joiner"],
                            sample_counts["joiner"],
                            values,
                        )
                        sample_counts["joiner"] += 1
                start += offset
            if decoder_records:
                raise RuntimeError("Unflushed decoder calibration records")
            utterance_records[-1].update(
                {
                    "encoder_sample_end": sample_counts["encoder"],
                    "decoder_sample_end": sample_counts["decoder"],
                    "joiner_sample_end": sample_counts["joiner"],
                }
            )

        if any(count == 0 for count in sample_counts.values()):
            raise ValueError(f"Calibration generated empty component: {sample_counts}")
        manifest = {
            "models": {
                component: model_manifest_record(filename)
                for component, filename in model_filenames.items()
            },
            "features": used_feature_records,
            "utterances": utterance_records,
            "calibration_method": calibration_method,
            "streaming": {
                "segment": segment,
                "offset": offset,
                "feature_dim": feature_dim,
                "blank_id": blank_id,
                "unk_id": unk_id,
                "vocab_size": vocab_size,
                "context_size": context_size,
                "joiner_dim": joiner_dim,
                "decoder_input_dtype": str(decoder_input_dtype),
                "sample_batch_size": 1,
                "dynamic_batch": {
                    "encoder": encoder_dynamic_batch,
                    "decoder": decoder_dynamic_batch,
                    "joiner": joiner_dynamic_batch,
                },
            },
            "limits": {
                "max_utterances": max_utterances,
                "max_chunks": max_chunks,
                "max_samples_per_model": max_samples_per_model,
            },
            "total_chunks": total_chunks,
            "truncated": truncated,
            "sample_counts": sample_counts,
        }
        write_calibration_manifest(temp_dir, manifest)
        commit_calibration_bundle(temp_dir, bundle_dir)
    except Exception:
        discard_calibration_temp_dir(temp_dir)
        raise
    return load_calibration_manifest(bundle_dir)


def _update_model_metadata_file(
    filename: PathLike,
    metadata: Dict[str, object],
) -> None:
    filename = Path(filename)
    model = onnx.load(str(filename), load_external_data=False)
    set_model_metadata(model, metadata)
    onnx.save_model(model, str(filename))
    onnx.checker.check_model(str(filename))


def quantize_static_int8(
    model_input: PathLike,
    model_output: PathLike,
    sample_dir: PathLike,
    calibration_method: str = "percentile",
    use_external_data: bool = False,
    calibration_manifest_hash: Optional[str] = None,
    calibration_bundle_hash_value: Optional[str] = None,
) -> Dict[str, object]:
    if calibration_method == "percentile":
        method = CalibrationMethod.Percentile
    elif calibration_method == "minmax":
        method = CalibrationMethod.MinMax
    else:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    if (
        calibration_manifest_hash is not None
        and calibration_bundle_hash_value is not None
        and calibration_manifest_hash != calibration_bundle_hash_value
    ):
        raise ValueError("Conflicting calibration bundle hashes")
    bundle_hash = calibration_bundle_hash_value or calibration_manifest_hash
    model_input = Path(model_input)
    model_output = Path(model_output)
    original_schema = load_model_io_schema(model_input)
    reader = NpzCalibrationDataReader(model_input, sample_dir)
    quantize_static(
        model_input=model_input,
        model_output=model_output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["MatMul", "Conv"],
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        use_external_data_format=use_external_data,
        calibrate_method=method,
        extra_options={
            "QDQOpTypePerChannelSupportToAxis": {"MatMul": 1},
            "WeightSymmetric": True,
        },
    )
    if load_model_io_schema(model_output) != original_schema:
        raise ValueError("Static quantization changed the model I/O schema")
    metadata = {
        "icefall.quantization.mode": "static_int8",
        "icefall.quantization.format": "QDQ",
        "icefall.quantization.activation_type": "QInt8",
        "icefall.quantization.weight_type": "QInt8",
        "icefall.quantization.per_channel": "1",
        "icefall.quantization.reduce_range": "0",
        "icefall.quantization.matmul_axis": "1",
        "icefall.quantization.op_types": "MatMul,Conv",
        "icefall.quantization.calibration_method": calibration_method,
        "icefall.quantization.onnxruntime_version": ort.__version__,
    }
    if bundle_hash is not None:
        metadata["icefall.quantization.calibration_bundle_sha256"] = bundle_hash
    _update_model_metadata_file(model_output, metadata)
    validate_quantized_model_artifact(model_input, model_output, metadata)
    return {
        "model_input": str(model_input),
        "model_output": str(model_output),
        "calibration_method": calibration_method,
        "sample_count": len(reader.files),
        "io_schema": original_schema,
    }


def _new_error_stats() -> Dict[str, object]:
    return {"max_abs": 0.0, "sum_abs": 0.0, "element_count": 0}


def _update_error_stats(
    stats: Dict[str, object],
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(
            "Output shape mismatch: "
            f"reference {reference.shape}, candidate {candidate.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Model output contains NaN/Inf")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    if difference.size:
        stats["max_abs"] = max(stats["max_abs"], float(difference.max()))
        stats["sum_abs"] += float(difference.sum())
        stats["element_count"] += difference.size


def _finalize_error_stats(stats: Dict[str, object]) -> Dict[str, float]:
    count = stats["element_count"]
    return {
        "max_abs": stats["max_abs"],
        "mean_abs": stats["sum_abs"] / count if count else 0.0,
        "element_count": count,
    }


def compare_models_on_npz(
    reference_filename: PathLike,
    candidate_filename: PathLike,
    sample_dir: PathLike,
    max_samples: int = 5,
) -> Dict[str, Dict[str, float]]:
    if load_model_io_schema(reference_filename) != load_model_io_schema(
        candidate_filename
    ):
        raise ValueError("Reference and candidate I/O schemas differ")
    reference_session = ort.InferenceSession(
        str(reference_filename), providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        str(candidate_filename), providers=["CPUExecutionProvider"]
    )
    output_names = [item.name for item in reference_session.get_outputs()]
    if output_names != [item.name for item in candidate_session.get_outputs()]:
        raise ValueError("Reference and candidate output names differ")
    reader = NpzCalibrationDataReader(
        reference_filename, sample_dir, max_samples=max_samples
    )
    stats = {name: _new_error_stats() for name in output_names}
    while True:
        values = reader.get_next()
        if values is None:
            break
        reference_outputs = reference_session.run(output_names, values)
        candidate_outputs = candidate_session.run(output_names, values)
        for name, reference, candidate in zip(
            output_names, reference_outputs, candidate_outputs
        ):
            _update_error_stats(stats[name], reference, candidate)
    return {name: _finalize_error_stats(value) for name, value in stats.items()}


def compare_streaming_model_sets(
    reference_filenames: Dict[str, PathLike],
    candidate_filenames: Dict[str, PathLike],
    bundle_dir: PathLike,
    max_utterances: int = 5,
    max_chunks: int = 50,
) -> Dict[str, object]:
    components = {"encoder", "decoder", "joiner"}
    if set(reference_filenames) != components or set(candidate_filenames) != components:
        raise ValueError(f"Model sets must contain {sorted(components)}")
    for component in components:
        if load_model_io_schema(reference_filenames[component]) != load_model_io_schema(
            candidate_filenames[component]
        ):
            raise ValueError(f"I/O schema mismatch for {component}")
    reference_sessions = {
        component: ort.InferenceSession(
            str(filename), providers=["CPUExecutionProvider"]
        )
        for component, filename in reference_filenames.items()
    }
    candidate_sessions = {
        component: ort.InferenceSession(
            str(filename), providers=["CPUExecutionProvider"]
        )
        for component, filename in candidate_filenames.items()
    }
    manifest = load_calibration_manifest(bundle_dir)
    utterances = manifest.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("Calibration manifest has no utterance boundaries")
    metadata = _model_metadata(reference_sessions["decoder"])
    context_size = int(metadata["context_size"])
    blank_id = int(metadata["blank_id"])
    unk_id = int(metadata["unk_id"])
    decoder_name = reference_sessions["decoder"].get_inputs()[0].name
    decoder_dtype = _node_arg_dtype(reference_sessions["decoder"].get_inputs()[0])
    reference_encoder_inputs = {
        item.name: item for item in reference_sessions["encoder"].get_inputs()
    }
    candidate_encoder_inputs = {
        item.name: item for item in candidate_sessions["encoder"].get_inputs()
    }
    state_names = [name for name in reference_encoder_inputs if name != "x"]
    if state_names != [name for name in candidate_encoder_inputs if name != "x"]:
        raise ValueError("Encoder state inputs differ")

    stats = {
        "encoder_out": _new_error_stats(),
        "decoder_out": _new_error_stats(),
        "logit": _new_error_stats(),
    }
    for name in state_names:
        stats[f"new_{name}"] = _new_error_stats()
    token_count = 0
    matching_token_count = 0
    first_token_divergence = None
    processed_chunks = 0

    for utterance_index, utterance in enumerate(utterances[:max_utterances]):
        reference_states = {
            name: _zero_input_from_schema(reference_encoder_inputs[name])
            for name in state_names
        }
        candidate_states = {
            name: _zero_input_from_schema(candidate_encoder_inputs[name])
            for name in state_names
        }
        reference_context = np.asarray(
            [[-1] * (context_size - 1) + [blank_id]], dtype=decoder_dtype
        )
        candidate_context = reference_context.copy()
        reference_decoder_out = _run_session(
            reference_sessions["decoder"], {decoder_name: reference_context}
        )["decoder_out"]
        candidate_decoder_out = _run_session(
            candidate_sessions["decoder"], {decoder_name: candidate_context}
        )["decoder_out"]
        _update_error_stats(
            stats["decoder_out"], reference_decoder_out, candidate_decoder_out
        )

        start = int(utterance["encoder_sample_start"])
        end = int(utterance["encoder_sample_end"])
        for sample_index in range(start, end):
            if processed_chunks >= max_chunks:
                break
            sample_filename = Path(bundle_dir) / "encoder" / f"{sample_index:06d}.npz"
            with np.load(sample_filename, allow_pickle=False) as archive:
                x = np.asarray(archive["x"])
            reference_values = {"x": x, **reference_states}
            candidate_values = {"x": x, **candidate_states}
            reference_outputs = _run_session(
                reference_sessions["encoder"], reference_values
            )
            candidate_outputs = _run_session(
                candidate_sessions["encoder"], candidate_values
            )
            reference_encoder_out = reference_outputs["encoder_out"]
            candidate_encoder_out = candidate_outputs["encoder_out"]
            _update_error_stats(
                stats["encoder_out"], reference_encoder_out, candidate_encoder_out
            )
            for name in state_names:
                output_name = f"new_{name}"
                _update_error_stats(
                    stats[output_name],
                    reference_outputs[output_name],
                    candidate_outputs[output_name],
                )
                reference_states[name] = reference_outputs[output_name]
                candidate_states[name] = candidate_outputs[output_name]

            for frame_index in range(reference_encoder_out.shape[1]):
                contexts_match = np.array_equal(reference_context, candidate_context)
                reference_joiner_values = {
                    "encoder_out": reference_encoder_out[:, frame_index, :],
                    "decoder_out": reference_decoder_out,
                }
                candidate_joiner_values = {
                    "encoder_out": candidate_encoder_out[:, frame_index, :],
                    "decoder_out": candidate_decoder_out,
                }
                reference_logit = _run_session(
                    reference_sessions["joiner"], reference_joiner_values
                )["logit"]
                candidate_logit = _run_session(
                    candidate_sessions["joiner"], candidate_joiner_values
                )["logit"]
                if contexts_match:
                    _update_error_stats(
                        stats["logit"], reference_logit, candidate_logit
                    )
                reference_token = int(np.argmax(reference_logit[0]))
                candidate_token = int(np.argmax(candidate_logit[0]))
                token_count += 1
                if reference_token == candidate_token:
                    matching_token_count += 1
                elif first_token_divergence is None:
                    first_token_divergence = {
                        "utterance": utterance_index,
                        "chunk": processed_chunks,
                        "frame": frame_index,
                        "reference_token": reference_token,
                        "candidate_token": candidate_token,
                    }
                if reference_token not in (blank_id, unk_id):
                    reference_context = np.concatenate(
                        [
                            reference_context[:, 1:],
                            np.asarray([[reference_token]], dtype=decoder_dtype),
                        ],
                        axis=1,
                    )
                    reference_decoder_out = _run_session(
                        reference_sessions["decoder"],
                        {decoder_name: reference_context},
                    )["decoder_out"]
                if candidate_token not in (blank_id, unk_id):
                    candidate_context = np.concatenate(
                        [
                            candidate_context[:, 1:],
                            np.asarray([[candidate_token]], dtype=decoder_dtype),
                        ],
                        axis=1,
                    )
                    candidate_decoder_out = _run_session(
                        candidate_sessions["decoder"],
                        {decoder_name: candidate_context},
                    )["decoder_out"]
                if np.array_equal(reference_context, candidate_context):
                    _update_error_stats(
                        stats["decoder_out"],
                        reference_decoder_out,
                        candidate_decoder_out,
                    )
            processed_chunks += 1
        if processed_chunks >= max_chunks:
            break

    return {
        "outputs": {
            name: _finalize_error_stats(value) for name, value in stats.items()
        },
        "processed_chunks": processed_chunks,
        "token_count": token_count,
        "matching_token_count": matching_token_count,
        "token_agreement": matching_token_count / token_count if token_count else 1.0,
        "first_token_divergence": first_token_divergence,
    }
