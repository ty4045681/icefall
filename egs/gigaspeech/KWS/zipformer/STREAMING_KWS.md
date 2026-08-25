# Streaming KWS and wake-word clip export

`streaming_kws.py` runs the causal Zipformer encoder and keyword beam search
chunk by chunk. It accepts either one audio file or a dma-kws-style CSV.

## Inputs

An Icefall training checkpoint normally contains both `model` and the model
arguments. An exported weight-only `pretrained.pt` does not contain the model
arguments; for that file, pass `--model-config` using the same structure as
`example_model_config.json`. In both cases, the original SentencePiece
`bpe.model` is required.

The CSV requires:

```csv
audio_path,keyword,label
audio/positive.wav,hey eva,1
audio/negative.wav,hey eva,0
audio/id_tokens.wav,bpe_ids:123 456,1
audio/piece_tokens.wav,bpe_pieces:▁HEY ▁EVA,1
audio/json_ids.wav,"[123,456]",1
```

`label` is optional. Relative audio paths are resolved against the CSV's
directory. Extra columns such as `text_variant` and `keyword_phonemes` are
retained in the output.

The `keyword` cell accepts either text or tokens from the same `bpe.model`:

- `hey eva`: ordinary text; it is upper-cased and then SentencePiece-encoded,
  preserving the existing behavior.
- `text:[alarm]`: explicit text. Use `text:` when real text begins with `[` or
  one of the reserved prefixes.
- `bpe_ids:123 456`, `bpe_ids:[123,456]`, or `[123,456]`: token IDs. The bare
  JSON form is a compatibility shorthand; the prefixed form is clearer.
- `bpe_pieces:▁HEY ▁EVA` or
  `bpe_pieces:["▁HEY","▁EVA"]`: exact SentencePiece pieces. A bare JSON
  string array is also accepted.

Direct BPE input is never encoded a second time or upper-cased. IDs are checked
against the vocabulary and pieces are matched exactly; out-of-range IDs,
`<blk>`, `<unk>`, the recipe's reserved `<sos/eos>`, and unknown pieces fail
before inference. A malformed field beginning with `[` also fails instead of
silently falling back to text. JSON arrays containing commas or quotes must
follow normal CSV quoting rules; for example, `"bpe_ids:[123,456]"` and
`"bpe_pieces:[""▁HEY"",""▁EVA""]"`.

## Batch decode and export

The script can be launched from the repository root (it adds that root to its
module search path for dma-kws subprocess compatibility):

```bash
python3 egs/gigaspeech/KWS/zipformer/streaming_kws.py \
  --checkpoint /path/to/epoch-30.pt \
  --bpe-model /path/to/bpe.model \
  --manifest /path/to/manifest.csv \
  --output-dir /path/to/wake_clips \
  --device cuda:0 \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 64 \
  --keywords-score 1.5 \
  --keywords-threshold 0.35 \
  --pre-roll-sec 0.15 \
  --post-roll-sec 0.15
```

`chunk-size` and `left-context-frames` are measured at the 50 Hz
encoder-embed rate. A standard training checkpoint often records multi-latency
lists such as `16,32,64,-1`; the script requires an explicit single deployment
point rather than selecting one silently.

The output layout is:

```text
/path/to/wake_clips/
├── manifest.csv
└── wavs/
    ├── row_00000000_...wav
    └── ...
```

Exported WAVs are mono, 16 kHz, PCM16. `manifest.csv` remains compatible with
dma-kws: `audio_path` is relative to the output manifest, the original
`keyword` representation is retained, and the original optional `label` is
preserved. `detected_keyword` contains the readable text decoded from direct
BPE input. A hit on a negative source row therefore remains `label=0`. Added
audit columns include source path, raw and padded times, exact half-open sample
indexes, score, token-frame timestamps, and the streaming operating point.

The current dma-kws two-stage evaluator assumes `keyword` is human-readable
text when it runs Stage II G2P. Direct-BPE rows can be used by this Icefall
runner (and its Stage I locator), but a two-stage dma-kws integration should
keep a separate `keyword_text` column and update Stage II to use it.

Existing output files are not replaced unless `--overwrite` is given. A WAV
read/decode failure is logged and processing continues; the process returns
status 2 after writing successful rows. Use `--fail-fast` to stop immediately
instead. CSV schema errors (missing fields, empty keyword, or invalid label)
always fail before inference so a malformed dataset is not processed partially.

## Single WAV / dma-kws locator mode

```bash
python3 egs/gigaspeech/KWS/zipformer/streaming_kws.py \
  --checkpoint /path/to/epoch-30.pt \
  --bpe-model /path/to/bpe.model \
  --wav /path/to/input.wav \
  --keywords "hey eva" \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 64
```

Each hit is one JSON object on stdout:

```json
{"keyword":"HEY EVA","start_time":1.0,"end_time":1.64,"score":0.83,"timestamp_frames":[25,40],"decode_mode":"streaming"}
```

`start_time` and `end_time` are seconds, so this mode can be used directly as
`locator.decode_script` for dma-kws's `icefall_pt_kws` locator. The raw token
frames are deliberately named `timestamp_frames`, not `timestamps`, because
dma-kws interprets a `timestamps` array as seconds.

The dma-kws locator does not add the BPE path or device automatically. A typical
locator override therefore includes:

```yaml
locator:
  type: icefall_pt
  root: /path/to/icefall
  decode_script: egs/gigaspeech/KWS/zipformer/streaming_kws.py
  checkpoint: /path/to/epoch-30.pt
  decode_args:
    - --bpe-model
    - /path/to/bpe.model
    - --device
    - cuda:0
```

The explicit BPE argument can be omitted only when a full training checkpoint's
saved `bpe_model` path still resolves. For a weight-only PT, also pass
`--model-config` through `decode_args`.

## Boundary semantics

This recipe has a 10 ms fbank shift and fixed 4x output subsampling. Token frame
`t` maps to a 40 ms interval. A hit from frames `[first, ..., last]` uses the
half-open raw span:

```text
raw_start = first * 0.04 seconds
raw_end   = (last + 1) * 0.04 seconds
```

The batch exporter then adds configurable pre/post roll, clamps the span to the
16 kHz model waveform, and writes integer `start_sample` / `end_sample` values.
This makes the waveform operation sample-exact and avoids truncating the last
40 ms frame.

If the causal model emits every token of a terminal keyword only during the
synthetic flush silence, the token span is anchored to the real audio EOF and
the output row has `tail_anchored=1`. Artificial silence is never exported.

RNN-T token emission times are not forced-alignment boundaries. The default
150 ms margins are intended to avoid clipping weak word onsets or endings. If
tighter linguistic boundaries are required, calibrate the offsets on annotated
audio or add a CTC/forced-alignment stage; changing chunk size alone is not a
valid fixed latency correction.
