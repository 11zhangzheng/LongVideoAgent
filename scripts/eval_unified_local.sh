#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/env.sh"
fi

# =====================================
# Unified Local Eval Config (Edit Here)
# =====================================
# Required
DATASET="${DATASET:-tvqa_plus}"             # tvqa | tvqa_plus
DATA_ROOT="${DATA_ROOT:-/mnt/disk1/zhangzheng/Tvqa_data}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/disk1/zhangzheng/models}"
LLM_PATH="${LLM_PATH:-$MODEL_ROOT/longvideoagent-qwen2.5-3b}"

# Core eval args
MAX_TURN="${MAX_TURN:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

# Quick eval: set EVAL_LIMIT=0 to run the full validation set.
EVAL_LIMIT="${EVAL_LIMIT:-3}"
DEFAULT_PYTHON_BIN="/home/zhangzheng/disk1/zhangzheng/conda_envs/videotemp_o3/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON_BIN"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"

# vLLM config. The upstream evaluator uses max_model_len=60000 while the
# released Qwen2.5-3B config declares 32768, so vLLM requires this override.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-30000}"
export VLLM_DTYPE="${VLLM_DTYPE:-half}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VISION_POLICY="${VISION_POLICY:-dynamic}"
export VISION_MIN_FRAMES="${VISION_MIN_FRAMES:-4}"
export VISION_MAX_FRAMES="${VISION_MAX_FRAMES:-20}"
export VISION_MAX_CLIPS="${VISION_MAX_CLIPS:-2}"
LOCAL_EVAL_LOG="${LOCAL_EVAL_LOG:-./results/eval_unified_local_last.log}"

# API agents used by local pipeline
GROUNDING_MODEL="${GROUNDING_MODEL:-grok-4-fast-reasoning}"
VISION_MODEL="${VISION_MODEL:-gpt-4o}"
GROUNDING_BASE_URL="${GROUNDING_BASE_URL:-https://api2.aigcbest.top/v1}"
VISION_BASE_URL="${VISION_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
GROUNDING_API_KEY="${GROUNDING_API_KEY:-${qdd_api:-}}"
VISION_API_KEY="${VISION_API_KEY:-${aliyun_api:-}}"

# Optional grounding cache
GROUNDING_CACHE_JSON_PATH="${GROUNDING_CACHE_JSON_PATH:-$DATA_ROOT/grounding_pairs_6000_samples.json}"

# Optional explicit path overrides (leave empty to use dataset defaults)
QUESTIONS_PATH="${QUESTIONS_PATH:-}"
SUBS_PATH="${SUBS_PATH:-}"
BASE_FRAME_DIR="${BASE_FRAME_DIR:-}"
BBOX_JSON_PATH="${BBOX_JSON_PATH:-}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-}"
DETAILED_OUTPUT_FILENAME="${DETAILED_OUTPUT_FILENAME:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RESUME_EVAL="${RESUME_EVAL:-0}"

# =========================
# Dataset-specific defaults
# =========================
if [[ "$DATASET" == "tvqa" ]]; then
  DEFAULT_QUESTIONS_PATH="$DATA_ROOT/LongTVQA_val_normalized.jsonl"
  DEFAULT_SUBS_PATH="$DATA_ROOT/hf_datasets/LongTVQA/LongTVQA_subtitles_clip_level.jsonl"
  DEFAULT_BASE_FRAME_DIR="$DATA_ROOT/frames/bbt_frames"
  DEFAULT_BBOX_JSON_PATH=""
  DEFAULT_OUTPUT_FILENAME="./results/eval_tvqa_local_quick.json"
  DEFAULT_DETAILED_OUTPUT_FILENAME="./results/data_log_eval_tvqa_local_quick.json"
elif [[ "$DATASET" == "tvqa_plus" ]]; then
  DEFAULT_QUESTIONS_PATH="$DATA_ROOT/LongTVQA_plus_val_normalized.json"
  DEFAULT_SUBS_PATH="$DATA_ROOT/hf_datasets/LongTVQA/LongTVQA_subtitles_clip_level.jsonl"
  DEFAULT_BASE_FRAME_DIR="$DATA_ROOT/frames/bbt_frames"
  DEFAULT_BBOX_JSON_PATH="$DATA_ROOT/clip_bbox_mapping.json"
  DEFAULT_OUTPUT_FILENAME="./results/eval_tvqa_plus_local_quick.json"
  DEFAULT_DETAILED_OUTPUT_FILENAME="./results/data_log_eval_tvqa_plus_local_quick.json"
else
  echo "[ERROR] DATASET must be 'tvqa' or 'tvqa_plus', got: $DATASET" >&2
  exit 1
fi

if [[ -z "$LLM_PATH" ]]; then
  echo "[ERROR] Please set LLM_PATH, for example: /mnt/disk1/zhangzheng/models/longvideoagent-qwen2.5-3b" >&2
  exit 1
fi

if [[ ! -d "$LLM_PATH" || ! -f "$LLM_PATH/config.json" ]]; then
  echo "[ERROR] LLM_PATH does not look like a local HuggingFace model directory: $LLM_PATH" >&2
  echo "        Download first, for example:" >&2
  echo "        HF_ENDPOINT=https://hf-mirror.com HF_HOME=$DATA_ROOT/.hf_home hf download longvideoagent/longvideoagent-qwen2.5-3b --local-dir $MODEL_ROOT/longvideoagent-qwen2.5-3b --local-dir-use-symlinks False --resume-download" >&2
  exit 1
fi

if [[ -z "$GROUNDING_API_KEY" || -z "$VISION_API_KEY" ]]; then
  echo "[ERROR] Please set GROUNDING_API_KEY and VISION_API_KEY at the top of scripts/eval_unified_local.sh" >&2
  exit 1
fi

if [[ "$SKIP_GPU_CHECK" != "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"[ERROR] Cannot import torch for GPU preflight: {exc}")

if not torch.cuda.is_available():
    raise SystemExit(
        "[ERROR] CUDA is not available. Local vLLM evaluation requires a GPU node "
        "with a working NVIDIA driver. Set SKIP_GPU_CHECK=1 only if you know vLLM "
        "can still infer the device."
    )

names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
print(f"CUDA OK: {torch.cuda.device_count()} visible device(s): {names}")
PY
fi

# Apply optional overrides
if [[ -z "$QUESTIONS_PATH" ]]; then QUESTIONS_PATH="$DEFAULT_QUESTIONS_PATH"; fi
if [[ -z "$SUBS_PATH" ]]; then SUBS_PATH="$DEFAULT_SUBS_PATH"; fi
if [[ -z "$BASE_FRAME_DIR" ]]; then BASE_FRAME_DIR="$DEFAULT_BASE_FRAME_DIR"; fi
if [[ -z "$BBOX_JSON_PATH" ]]; then BBOX_JSON_PATH="$DEFAULT_BBOX_JSON_PATH"; fi
if [[ -z "$OUTPUT_FILENAME" ]]; then OUTPUT_FILENAME="$DEFAULT_OUTPUT_FILENAME"; fi
if [[ -z "$DETAILED_OUTPUT_FILENAME" ]]; then DETAILED_OUTPUT_FILENAME="$DEFAULT_DETAILED_OUTPUT_FILENAME"; fi
if [[ -z "$CHECKPOINT_PATH" ]]; then CHECKPOINT_PATH="${OUTPUT_FILENAME}.checkpoint.jsonl"; fi

mkdir -p \
  "$(dirname "$OUTPUT_FILENAME")" \
  "$(dirname "$DETAILED_OUTPUT_FILENAME")" \
  "$(dirname "$CHECKPOINT_PATH")"

if [[ "$EVAL_LIMIT" =~ ^[0-9]+$ ]] && (( EVAL_LIMIT > 0 )); then
  QUICK_DIR="$ROOT_DIR/results/.quick_eval"
  mkdir -p "$QUICK_DIR"
  QUICK_QUESTIONS_PATH="$QUICK_DIR/${DATASET}_first_${EVAL_LIMIT}_questions.json"
  "$PYTHON_BIN" - "$QUESTIONS_PATH" "$QUICK_QUESTIONS_PATH" "$EVAL_LIMIT" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])

with src.open(encoding="utf-8") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError:
        f.seek(0)
        data = [json.loads(line) for line in f if line.strip()]

if not isinstance(data, list):
    raise SystemExit(f"Questions file must contain a list: {src}")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(data[:limit], f, ensure_ascii=False, indent=2)
PY
  QUESTIONS_PATH="$QUICK_QUESTIONS_PATH"
fi

echo "=== Unified Local Evaluation ==="
echo "dataset: $DATASET"
echo "python_bin: $PYTHON_BIN"
echo "llm_path: $LLM_PATH"
echo "max_turn: $MAX_TURN"
echo "gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
echo "vllm_max_model_len: $VLLM_MAX_MODEL_LEN"
echo "vllm_dtype: $VLLM_DTYPE"
echo "vision_policy: $VISION_POLICY"
echo "vision_frame_range: $VISION_MIN_FRAMES-$VISION_MAX_FRAMES"
echo "vision_max_clips: $VISION_MAX_CLIPS"
echo "vision_max_calls: ${VISION_MAX_CALLS:-auto}"
echo "eval_limit: $EVAL_LIMIT"
echo "questions: $QUESTIONS_PATH"
echo "subtitles: $SUBS_PATH"
echo "frames: $BASE_FRAME_DIR"
echo "bbox: ${BBOX_JSON_PATH:-<none>}"
echo "summary_output: $OUTPUT_FILENAME"
echo "detail_output: $DETAILED_OUTPUT_FILENAME"
echo "checkpoint: $CHECKPOINT_PATH"
echo "resume: $RESUME_EVAL"
echo "log: $LOCAL_EVAL_LOG"
echo "grounding_model: $GROUNDING_MODEL"
echo "vision_model: $VISION_MODEL"

cmd=(
  "$PYTHON_BIN" src/evaluation/lvagent/evaluate_local_unified.py
  --dataset "$DATASET"
  --llm-path "$LLM_PATH"
  --max_turn "$MAX_TURN"
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
  --questions-path "$QUESTIONS_PATH"
  --subs-path "$SUBS_PATH"
  --base-frame-dir "$BASE_FRAME_DIR"
  --output-filename "$OUTPUT_FILENAME"
  --detailed-output-filename "$DETAILED_OUTPUT_FILENAME"
  --checkpoint-path "$CHECKPOINT_PATH"
  --grounding-model "$GROUNDING_MODEL"
  --vision-model "$VISION_MODEL"
  --grounding-base-url "$GROUNDING_BASE_URL"
  --vision-base-url "$VISION_BASE_URL"
  --grounding-api-key "$GROUNDING_API_KEY"
  --vision-api-key "$VISION_API_KEY"
  --grounding-cache-json-path "$GROUNDING_CACHE_JSON_PATH"
)

if [[ -n "$BBOX_JSON_PATH" ]]; then
  cmd+=(--bbox-json-path "$BBOX_JSON_PATH")
fi
if [[ "$RESUME_EVAL" == "1" ]]; then
  cmd+=(--resume)
fi

if [[ "$RESUME_EVAL" == "1" && -s "$CHECKPOINT_PATH" ]]; then
  "${cmd[@]}" 2>&1 | tee -a "$LOCAL_EVAL_LOG"
else
  "${cmd[@]}" 2>&1 | tee "$LOCAL_EVAL_LOG"
fi

echo "=== Done ==="
echo "Summary JSON: $OUTPUT_FILENAME"
echo "Detailed JSON: $DETAILED_OUTPUT_FILENAME"
