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

## 多阈值精确扫描

批量 manifest 模式可用 `--keywords-thresholds` 一次扫描多个声学阈值：

```bash
python3 egs/gigaspeech/KWS/zipformer/streaming_kws.py \
  --checkpoint /path/to/epoch-30.pt \
  --bpe-model /path/to/bpe.model \
  --manifest /path/to/negative_manifest.csv \
  --output-dir /path/to/threshold_scan \
  --device cuda:0 \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 64 \
  --keywords-score 1.5 \
  --keywords-thresholds "0.10,0.15 0.20,0.25"
```

该参数接受逗号或空格分隔的 `[0, 1]` 有限数，会按数值去重并排序；
它不能与显式 `--keywords-threshold` 同时使用，也不支持单 WAV
`--wav` 模式。单 WAV 请使用单值 `--keywords-threshold`。

对每个 manifest trial（即输入 CSV 行），程序只执行一次音频加载、feature
计算和流式 encoder/cache 推进，并共享 joiner encoder projection。但每个
阈值都有独立的 context graph、beam hypotheses、尾随 blank 计数、context
state 和命中后的 reset。因此它等价于在同一份 encoder 输出上分别运行各
阈值的解码器，而不是近似扫描。

不能用“低阈值解码一次，再按命中 `score` 过滤”替代。低阈值命中会立即
reset 它自己的 decoder/context state，后续 beam 路径和命中时间已与高阈值
解码器不同。所以各阈值的命中集不保证是包含关系，命中数也不保证
随阈值单调。

所有阈值的命中写入同一个 `manifest.csv`，每行都有
`keywords_threshold`。多阈值模式的 WAV 文件名包含 `_thr_<threshold>`，
因此即使不同阈值得到相同的 sample 边界，也不会互相覆盖。

## 纯负样本阈值评估

`negative_kws_eval.py` 将负样本准备、精确阈值扫描和报告分成
`prepare` / `sweep` / `report` 三个子命令。

### 1. 准备 MUSAN manifest

MUSAN 根目录通常以 `music` / `noise` / `speech` 为第一层。下例会
递归扫描 WAV，并为每个“音频 × 关键词”组合生成一行 `label=0`：

```bash
python3 egs/gigaspeech/KWS/zipformer/negative_kws_eval.py prepare \
  --input-dir /datasets/musan \
  --output-manifest /work/kws_eval/musan_negative.csv \
  --keyword "hey eva" \
  --extensions wav \
  --category-depth 0 \
  --overwrite
```

`--keyword` 可重复以评估多个关键词。`--category-depth 0` 取输入根目录下的
第一个目录分量，对 MUSAN 即上述三类；负数表示直接父目录。
`--extensions` 接受空格或逗号分隔的后缀，但当前时长探测仅支持
WAV/WAVE；MUSAN 使用默认 `wav` 即可。

多个 keyword 会把同一音频展开为多个独立 trial；当前实现按行推理，
因此 encoder 只在“同一 trial 的多个阈值”之间共享，不跨 keyword trial 共享。

输出 CSV 包含 `audio_path,keyword,label,category,source_duration_sec`。
`audio_path` 默认相对于输出 manifest；`--absolute-paths` 可改为绝对路径。

### 2. 扫描阈值

```bash
python3 egs/gigaspeech/KWS/zipformer/negative_kws_eval.py sweep \
  --manifest /work/kws_eval/musan_negative.csv \
  --output-dir /work/kws_eval/musan_sweep \
  --thresholds 0.10:0.50:0.05 \
  --category-field category \
  --duration-field source_duration_sec \
  --overwrite \
  -- \
  --checkpoint /path/to/epoch-30.pt \
  --bpe-model /path/to/bpe.model \
  --device cuda:0 \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 64 \
  --keywords-score 1.5
```

`--thresholds` 支持三种可混合的写法：空格列表、逗号列表，以及
`start:stop:step`，例如 `--thresholds 0.1 0.2,0.3 0.4:0.8:0.05`。
范围不越过 `stop`；当步长恰好到达时包含终点。所有值必须在 `[0, 1]`
内，并会按数值去重、排序。

`--` 之后是透传给 `streaming_kws.py` 的模型和流式参数。
`--decode-script` 可替换默认的同目录脚本，`--python` 可选择解释器。
manifest 路径、输出路径、阈值、`--fail-fast` 和 `--overwrite` 由 sweep
管理，不能在透传参数中重复指定。

sweep 强制启用 `--fail-fast`：任一源音频失败都会使本次 inference
失败，`run.json` 记录 `failed` 状态，详情见 `runner.log`。这样不会把
“未评估的文件”错当成“零误触发”。修复输入后应用 `--overwrite`
重新运行。`--resume` 不是失败任务的断点续跑；它只在 manifest、decode
script 及参数指纹完全一致，且 inference 已成功完成时，复用命中结果
并重建报告。为完整跟踪模型资产，使用 `--resume` 的 sweep 必须在透传
参数中显式给出可从当前目录解析的 `--checkpoint` 和 `--bpe-model`
（建议绝对路径）；命中 manifest 也会按 SHA256 校验。`--resume` 与
`--overwrite` 互斥。

如果输入 manifest 有 `label`，sweep 要求每行都为 `0`。只有在已独立
确认整份无标签 manifest 都是负样本时，才应使用 `--assume-negative`
允许空标签。默认要求 `category` 列存在，字段名可通过 `--category-field`
指定；若明确不做分类，可传 `--category-field ''`，所有行将归入
`uncategorized`。缺少指定的时长字段时会直接探测 WAV 时长。

### 3. 产物和指标

`sweep` 成功后目录如下：

```text
/work/kws_eval/musan_sweep/
├── run.json
├── exposure.csv
├── runner.log
├── inference/
│   ├── manifest.csv
│   └── wavs/
└── report/
    ├── source_metrics.csv
    ├── threshold_metrics.csv
    ├── summary.json
    ├── report.html
    ├── fa-per-hour-<keyword-slug>.svg
    └── source-trial-trigger-rate-<keyword-slug>.svg
```

- `run.json` 保存命令、输入/脚本指纹、阈值、状态和产物路径；
  `runner.log` 是 inference 子进程日志。
- `exposure.csv` 是标准化后的负样本试验及时长分母；
  `inference/manifest.csv` 是含 `keywords_threshold` 的合并命中表。
- `source_metrics.csv` 每行对应“阈值 × 源 trial”，记录命中事件数、
  是否触发及最高分；`threshold_metrics.csv` 按阈值、关键词和 category
  汇总，并附带 `ALL` 总计。
- `summary.json` 是机器可读指标与非单调警告；`report.html` 和 SVG
  是可视化报告。

这里的一个 **source trial** 定义为输入 manifest 的一行，即一个
“音频 × 关键词”对。同一音频如果对多个关键词建行，会计为多个
source trials；报告同时给出 `unique_audio_files` 以便区分。

- `false_alarm_events` 是命中事件总数；同一 trial 的多次命中全部计数。
- `triggered_source_trials` 是至少命中一次的 trial 数；每个 trial
  最多计一次。
- `FA/h = false_alarm_events / exposure_hours`，其中 exposure 是当前
  “阈值 × 关键词 × category”切片的 source-trial 时长之和。
- `source_trial_trigger_rate = triggered_source_trials / source_trials`。

零命中是正常成功结果：`inference/manifest.csv` 只有表头，报告仍会生成，
`false_alarm_events`、`FA/h` 和触发率均为 0。若更高阈值的
FA/h 或触发率反而升高，报告会保留实测值并生成非单调警告，
不做强制单调修正。

如果只需从已完成的 inference 重建 CSV/JSON/HTML/SVG，无需再加载模型：

```bash
python3 egs/gigaspeech/KWS/zipformer/negative_kws_eval.py report \
  --run-dir /work/kws_eval/musan_sweep \
  --overwrite
```

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
