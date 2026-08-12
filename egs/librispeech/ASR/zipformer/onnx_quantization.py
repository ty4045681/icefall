import importlib.util
from pathlib import Path

implementation_path = (
    Path(__file__).parents[3]
    / "gigaspeech"
    / "KWS"
    / "zipformer"
    / "onnx_quantization.py"
)
spec = importlib.util.spec_from_file_location(
    "icefall_gigaspeech_kws_onnx_quantization", implementation_path
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Cannot load ONNX quantization helpers from {implementation_path}"
    )
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)

calibration_bundle_hash = implementation.calibration_bundle_hash
compare_models_on_npz = implementation.compare_models_on_npz
compare_streaming_model_sets = implementation.compare_streaming_model_sets
generate_calibration_bundle = implementation.generate_calibration_bundle
normalize_static_quantization_op_types = (
    implementation.normalize_static_quantization_op_types
)
quantize_static_int8 = implementation.quantize_static_int8
quantize_weight_only_int8 = implementation.quantize_weight_only_int8
resolve_quantization_mode = implementation.resolve_quantization_mode
resolve_static_quantization_op_types = (
    implementation.resolve_static_quantization_op_types
)
STATIC_QUANTIZATION_PROFILES = implementation.STATIC_QUANTIZATION_PROFILES
validate_calibration_manifest = implementation.validate_calibration_manifest
