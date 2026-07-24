#!/usr/bin/env python3
#
# KWS Evaluation Script
#
# Evaluates keyword spotting models against a manifest CSV file.
# Computes TP, FP, FN, TN, Accuracy, Precision, Recall, FPR, F1 metrics.
#
# Usage:
#   python ./zipformer/eval_kws.py \
#       --manifest /path/to/manifest.csv \
#       --model-config /path/to/model_config.json \
#       --output-dir ./eval_results
#
# See README or plan for manifest.csv and model_config.json format details.

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
from beam_search import keywords_search
from train import get_model, get_params

from icefall import ContextGraph
from icefall.utils import AttributeDict

LOG_EPS = math.log(1e-10)


@dataclass
class KwMetric:
    TP: int = 0
    FN: int = 0
    FP: int = 0
    TN: int = 0
    FN_list: List[str] = field(default_factory=list)
    FP_list: List[str] = field(default_factory=list)
    TP_list: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"(TP:{self.TP}, FN:{self.FN}, FP:{self.FP}, TN:{self.TN})"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate KWS models against a manifest CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the manifest CSV file.",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model configuration JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (cpu or cuda).",
    )

    return parser


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load and parse the manifest CSV file.

    Returns a list of dicts with keys: audio_path, keyword, text_variant, label.
    audio_path is resolved to absolute path relative to CSV directory.
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = row["audio_path"].replace("\\", os.sep)
            audio_path = os.path.join(manifest_dir, audio_path)
            entries.append(
                {
                    "audio_path": audio_path,
                    "keyword": row["keyword"].strip(),
                    "text_variant": row["text_variant"].strip(),
                    "label": int(row["label"]),
                }
            )
    return entries


def build_model_params(model_args: Dict) -> AttributeDict:
    """Build params dict from model_args for model construction."""
    params = get_params()

    # Set model architecture parameters
    params.num_encoder_layers = model_args.get("num_encoder_layers", "1,1,1,1,1,1")
    params.downsampling_factor = model_args.get("downsampling_factor", "1,2,4,8,4,2")
    params.feedforward_dim = model_args.get("feedforward_dim", "192,192,192,192,192,192")
    params.num_heads = model_args.get("num_heads", "4,4,4,8,4,4")
    params.encoder_dim = model_args.get("encoder_dim", "128,128,128,128,128,128")
    params.query_head_dim = model_args.get("query_head_dim", "32")
    params.value_head_dim = model_args.get("value_head_dim", "12")
    params.pos_head_dim = model_args.get("pos_head_dim", "4")
    params.pos_dim = int(model_args.get("pos_dim", 48))
    params.encoder_unmasked_dim = model_args.get(
        "encoder_unmasked_dim", "128,128,128,128,128,128"
    )
    params.cnn_module_kernel = model_args.get("cnn_module_kernel", "31,31,15,15,15,31")
    params.decoder_dim = int(model_args.get("decoder_dim", 320))
    params.joiner_dim = int(model_args.get("joiner_dim", 320))
    params.causal = bool(int(model_args.get("causal", 1)))
    params.chunk_size = str(model_args.get("chunk_size", "16"))
    params.left_context_frames = str(model_args.get("left_context_frames", "64"))
    params.use_transducer = bool(int(model_args.get("use_transducer", 1)))
    params.use_ctc = bool(int(model_args.get("use_ctc", 0)))
    params.context_size = int(model_args.get("context_size", 2))

    return params


def load_kws_model(
    model_config: Dict, device: torch.device
) -> Tuple[nn.Module, spm.SentencePieceProcessor, ContextGraph, str]:
    """Load a single KWS model from config.

    Returns (model, sp, keywords_graph, keyword).
    """
    pt_path = model_config["pt_path"]
    keyword = model_config["keyword"]
    score = float(model_config.get("score", 1.0))
    threshold = float(model_config.get("threshold", 0.35))
    bpe_model_path = model_config["bpe_model"]
    model_args = model_config.get("model_args", {})

    # Build params and model
    params = build_model_params(model_args)

    # Load BPE model to get vocab_size and blank_id
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(f"Loading model for keyword '{keyword}' from {pt_path}")
    model = get_model(params)

    # Load pretrained weights
    checkpoint = torch.load(pt_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Build ContextGraph for this keyword
    keyword_upper = keyword.upper()
    token_ids = [sp.encode(keyword_upper)]
    phrases = [keyword_upper]
    keywords_scores = [score]
    keywords_thresholds = [threshold]

    keywords_graph = ContextGraph(context_score=score, ac_threshold=threshold)
    keywords_graph.build(
        token_ids=token_ids,
        phrases=phrases,
        scores=keywords_scores,
        ac_thresholds=keywords_thresholds,
    )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"  Model params: {num_param}, vocab_size: {params.vocab_size}")

    return model, sp, keywords_graph, keyword, params


def compute_fbank(
    audio_path: str, device: torch.device, sample_rate: int = 16000
) -> torch.Tensor:
    """Compute 80-dim fbank features for a single audio file.

    Tries kaldifeat first, then torchaudio as fallback.

    Returns a tensor of shape (1, T, 80).
    """
    try:
        from kaldifeat import Fbank, FbankOptions
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32")
        if sr != sample_rate:
            raise ValueError(
                f"Expected sample rate {sample_rate}, got {sr} for {audio_path}"
            )
        if len(data.shape) > 1:
            data = data[:, 0]

        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, num_samples)

        opts = FbankOptions()
        opts.device = torch.device("cpu")
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sample_rate
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400

        fbank = Fbank(opts)
        features = fbank(waveform)

        if isinstance(features, list):
            features = features[0]  # (T, 80)
            features = features.unsqueeze(0)  # (1, T, 80)

        return features.to(device)

    except ImportError:
        pass

    # Fallback: torchaudio
    try:
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        waveform, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        # Use first channel
        if waveform.size(0) > 1:
            waveform = waveform[0:1]

        features = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=sample_rate,
            dither=0.0,
            snip_edges=False,
            high_freq=-400,
        )  # (T, 80)

        return features.unsqueeze(0).to(device)  # (1, T, 80)

    except ImportError:
        raise ImportError(
            "Either kaldifeat or torchaudio is required for feature extraction. "
            "Install one of them:\n"
            "  pip install kaldifeat\n"
            "  pip install torchaudio"
        )


def run_kws_inference(
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    keywords_graph: ContextGraph,
    params: AttributeDict,
    features: torch.Tensor,
    device: torch.device,
) -> List[str]:
    """Run KWS inference on features.

    Args:
        features: (1, T, 80) fbank features.

    Returns:
        List of detected keyword phrases.
    """
    feature_lens = torch.tensor([features.size(1)], dtype=torch.int64, device=device)

    if params.causal:
        pad_len = 30
        feature_lens = feature_lens + pad_len
        features = torch.nn.functional.pad(
            features,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPS,
        )

    encoder_out, encoder_out_lens = model.forward_encoder(features, feature_lens)

    ans_dict = keywords_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        keywords_graph=keywords_graph,
        beam=4,
        num_tailing_blanks=1,
        blank_penalty=0.0,
    )

    detected = []
    for ans in ans_dict:
        for hit in ans:
            detected.append(hit.phrase)

    return detected


def evaluate(
    manifest_entries: List[Dict],
    models: Dict[str, Tuple],
    device: torch.device,
) -> Dict[str, KwMetric]:
    """Run evaluation over all manifest entries.

    Args:
        manifest_entries: List of manifest entries.
        models: Dict mapping keyword -> (model, sp, keywords_graph, params).

    Returns:
        Dict mapping keyword (and "all") -> KwMetric.
    """
    metrics = {"all": KwMetric()}
    for kw in models:
        metrics[kw] = KwMetric()

    total = len(manifest_entries)
    for idx, entry in enumerate(manifest_entries):
        if idx % 100 == 0:
            logging.info(f"Processing {idx}/{total}...")

        audio_path = entry["audio_path"]
        target_keyword = entry["keyword"]
        text_variant = entry["text_variant"]
        label = entry["label"]

        if target_keyword not in models:
            logging.warning(
                f"Keyword '{target_keyword}' not found in model config, skipping: {audio_path}"
            )
            continue

        model, sp, keywords_graph, params = models[target_keyword]

        # Compute features
        try:
            features = compute_fbank(audio_path, device)
        except Exception as e:
            logging.error(f"Failed to process {audio_path}: {e}")
            continue

        # Run inference
        detected = run_kws_inference(
            model=model,
            sp=sp,
            keywords_graph=keywords_graph,
            params=params,
            features=features,
            device=device,
        )

        # Check if the target keyword was triggered
        triggered = any(
            d.upper() == target_keyword.upper() for d in detected
        )

        # Update metrics
        if label == 1:
            # Positive sample
            if triggered:
                metrics["all"].TP += 1
                metrics[target_keyword].TP += 1
                metrics[target_keyword].TP_list.append(
                    f"({text_variant} -> {target_keyword})"
                )
            else:
                metrics["all"].FN += 1
                metrics[target_keyword].FN += 1
                metrics[target_keyword].FN_list.append(
                    f"({text_variant} -> MISS)"
                )
        else:
            # Negative sample
            if triggered:
                metrics["all"].FP += 1
                metrics[target_keyword].FP += 1
                metrics[target_keyword].FP_list.append(
                    f"({text_variant} -> {target_keyword})"
                )
            else:
                metrics["all"].TN += 1
                metrics[target_keyword].TN += 1

    return metrics


def format_results(metrics: Dict[str, KwMetric]) -> str:
    """Format evaluation results as a string."""
    lines = []
    width = 10

    for key in sorted(metrics.keys(), key=lambda x: (x != "all", x)):
        item = metrics[key]
        total = item.TP + item.TN + item.FP + item.FN
        if total == 0:
            continue

        acc = (item.TP + item.TN) / total
        precision = 0.0 if (item.TP + item.FP) == 0 else item.TP / (item.TP + item.FP)
        recall = 0.0 if (item.TP + item.FN) == 0 else item.TP / (item.TP + item.FN)
        fpr = 0.0 if (item.FP + item.TN) == 0 else item.FP / (item.FP + item.TN)
        f1 = (
            0.0
            if precision * recall == 0
            else 2 * precision * recall / (precision + recall)
        )

        s = f"{key}:\n"
        s += f"\t{'TP':{width}}{'FP':{width}}{'FN':{width}}{'TN':{width}}\n"
        s += f"\t{str(item.TP):{width}}{str(item.FP):{width}}{str(item.FN):{width}}{str(item.TN):{width}}\n"
        s += f"\tAccuracy:  {acc:.4f}\n"
        s += f"\tPrecision: {precision:.4f}\n"
        s += f"\tRecall:    {recall:.4f}\n"
        s += f"\tFPR:       {fpr:.4f}\n"
        s += f"\tF1:        {f1:.4f}\n"
        if key != "all":
            if item.TP_list:
                s += f"\tTP list: {' # '.join(item.TP_list[:20])}\n"
            if item.FP_list:
                s += f"\tFP list: {' # '.join(item.FP_list[:20])}\n"
            if item.FN_list:
                s += f"\tFN list: {' # '.join(item.FN_list[:20])}\n"
        lines.append(s)

    return "\n".join(lines)


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.info(f"Arguments: {vars(args)}")

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    # Load manifest
    logging.info(f"Loading manifest from {args.manifest}")
    manifest_entries = load_manifest(args.manifest)
    logging.info(f"Loaded {len(manifest_entries)} entries from manifest.")

    # Load model config
    logging.info(f"Loading model config from {args.model_config}")
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    # Load all models
    models = {}  # keyword -> (model, sp, keywords_graph, params)
    for cfg in model_configs:
        model, sp, keywords_graph, keyword, params = load_kws_model(cfg, device)
        models[keyword] = (model, sp, keywords_graph, params)

    logging.info(f"Loaded {len(models)} model(s) for keywords: {list(models.keys())}")

    # Run evaluation
    logging.info("Starting evaluation...")
    metrics = evaluate(manifest_entries, models, device)

    # Format and output results
    results_str = format_results(metrics)
    logging.info("\n" + "=" * 60 + "\nEvaluation Results\n" + "=" * 60)
    print("\n" + results_str)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "kws_eval_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(results_str)
    logging.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
