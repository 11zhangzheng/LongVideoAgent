#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/env.sh"
fi

# ===================================
# Unified API Eval Config (Edit Here)
# ===================================
# Required
DATASET="${DATASET:-tvqa_plus}"             # tvqa | tvqa_plus
DATA_ROOT="${DATA_ROOT:-/mnt/disk1/zhangzheng/Tvqa_data}"

# Core eval args
CHECKPOINT_STEP="${CHECKPOINT_STEP:-api}"   # metadata only
MAX_TURN="${MAX_TURN:-5}"
THREADS="${THREADS:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.1}" # metadata only

# Quick eval: set EVAL_LIMIT=0 to run the full validation set.
EVAL_LIMIT="${EVAL_LIMIT:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# API models
GROUNDING_MODEL="${GROUNDING_MODEL:-grok-4-fast-reasoning}"
VISION_MODEL="${VISION_MODEL:-gpt-4o}"
MAIN_MODEL="${MAIN_MODEL:-grok-4-fast-reasoning}"

# API endpoints
GROUNDING_BASE_URL="${GROUNDING_BASE_URL:-https://api2.aigcbest.top/v1}"
VISION_BASE_URL="${VISION_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MAIN_BASE_URL="${MAIN_BASE_URL:-https://api2.aigcbest.top/v1}"

# API keys are loaded from env.sh. Keep these empty in the script.
GROUNDING_API_KEY="${GROUNDING_API_KEY:-${qdd_api:-}}"
VISION_API_KEY="${VISION_API_KEY:-${aliyun_api:-}}"
MAIN_API_KEY="${MAIN_API_KEY:-${qdd_api:-}}"

# Optional grounding cache
GROUNDING_CACHE_JSON_PATH="${GROUNDING_CACHE_JSON_PATH:-$DATA_ROOT/grounding_pairs_6000_samples.json}"

# Optional explicit path overrides (leave empty to use dataset defaults)
QUESTIONS_PATH="${QUESTIONS_PATH:-}"
SUBS_PATH="${SUBS_PATH:-}"
BASE_FRAME_DIR="${BASE_FRAME_DIR:-}"
BBOX_JSON_PATH="${BBOX_JSON_PATH:-}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-}"
DETAILED_OUTPUT_FILENAME="${DETAILED_OUTPUT_FILENAME:-}"

# =========================
# Dataset-specific defaults
# =========================
if [[ "$DATASET" == "tvqa" ]]; then
  DEFAULT_QUESTIONS_PATH="$DATA_ROOT/LongTVQA_val_normalized.jsonl"
  DEFAULT_SUBS_PATH="$DATA_ROOT/hf_datasets/LongTVQA/LongTVQA_subtitles_clip_level.jsonl"
  DEFAULT_BASE_FRAME_DIR="$DATA_ROOT/frames/bbt_frames"
  DEFAULT_BBOX_JSON_PATH=""
  DEFAULT_OUTPUT_FILENAME="./results/eval_tvqa_api_quick.json"
  DEFAULT_DETAILED_OUTPUT_FILENAME="./results/data_log_eval_tvqa_api_quick.json"
elif [[ "$DATASET" == "tvqa_plus" ]]; then
  DEFAULT_QUESTIONS_PATH="$DATA_ROOT/LongTVQA_plus_val_normalized.json"
  DEFAULT_SUBS_PATH="$DATA_ROOT/hf_datasets/LongTVQA/LongTVQA_subtitles_clip_level.jsonl"
  DEFAULT_BASE_FRAME_DIR="$DATA_ROOT/frames/bbt_frames"
  DEFAULT_BBOX_JSON_PATH="$DATA_ROOT/clip_bbox_mapping.json"
  DEFAULT_OUTPUT_FILENAME="./results/eval_tvqa_plus_api_quick.json"
  DEFAULT_DETAILED_OUTPUT_FILENAME="./results/data_log_eval_tvqa_plus_api_quick.json"
else
  echo "[ERROR] DATASET must be 'tvqa' or 'tvqa_plus', got: $DATASET" >&2
  exit 1
fi

if [[ -z "$GROUNDING_API_KEY" || -z "$VISION_API_KEY" || -z "$MAIN_API_KEY" ]]; then
  echo "[ERROR] Please set GROUNDING_API_KEY, VISION_API_KEY, and MAIN_API_KEY at the top of scripts/eval_unified_api.sh" >&2
  exit 1
fi

# Apply optional overrides
if [[ -z "$QUESTIONS_PATH" ]]; then QUESTIONS_PATH="$DEFAULT_QUESTIONS_PATH"; fi
if [[ -z "$SUBS_PATH" ]]; then SUBS_PATH="$DEFAULT_SUBS_PATH"; fi
if [[ -z "$BASE_FRAME_DIR" ]]; then BASE_FRAME_DIR="$DEFAULT_BASE_FRAME_DIR"; fi
if [[ -z "$BBOX_JSON_PATH" ]]; then BBOX_JSON_PATH="$DEFAULT_BBOX_JSON_PATH"; fi
if [[ -z "$OUTPUT_FILENAME" ]]; then OUTPUT_FILENAME="$DEFAULT_OUTPUT_FILENAME"; fi
if [[ -z "$DETAILED_OUTPUT_FILENAME" ]]; then DETAILED_OUTPUT_FILENAME="$DEFAULT_DETAILED_OUTPUT_FILENAME"; fi

mkdir -p "$(dirname "$OUTPUT_FILENAME")" "$(dirname "$DETAILED_OUTPUT_FILENAME")"

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

echo "=== Unified API Evaluation ==="
echo "dataset: $DATASET"
echo "checkpoint_step: $CHECKPOINT_STEP"
echo "max_turn: $MAX_TURN"
echo "threads: $THREADS"
echo "eval_limit: $EVAL_LIMIT"
echo "questions: $QUESTIONS_PATH"
echo "subtitles: $SUBS_PATH"
echo "frames: $BASE_FRAME_DIR"
echo "bbox: ${BBOX_JSON_PATH:-<none>}"
echo "summary_output: $OUTPUT_FILENAME"
echo "detail_output: $DETAILED_OUTPUT_FILENAME"
echo "grounding_model: $GROUNDING_MODEL"
echo "vision_model: $VISION_MODEL"
echo "main_model: $MAIN_MODEL"

cmd=(
  "$PYTHON_BIN" src/evaluation/lvagent/evaluate_api_unified.py
  --dataset "$DATASET"
  --checkpoint_step "$CHECKPOINT_STEP"
  --max_turn "$MAX_TURN"
  --threads "$THREADS"
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
  --questions-path "$QUESTIONS_PATH"
  --subs-path "$SUBS_PATH"
  --base-frame-dir "$BASE_FRAME_DIR"
  --output-filename "$OUTPUT_FILENAME"
  --detailed-output-filename "$DETAILED_OUTPUT_FILENAME"
  --grounding-model "$GROUNDING_MODEL"
  --vision-model "$VISION_MODEL"
  --main-model "$MAIN_MODEL"
  --grounding-base-url "$GROUNDING_BASE_URL"
  --vision-base-url "$VISION_BASE_URL"
  --main-base-url "$MAIN_BASE_URL"
  --grounding-api-key "$GROUNDING_API_KEY"
  --vision-api-key "$VISION_API_KEY"
  --main-api-key "$MAIN_API_KEY"
  --grounding-cache-json-path "$GROUNDING_CACHE_JSON_PATH"
)

if [[ -n "$BBOX_JSON_PATH" ]]; then
  cmd+=(--bbox-json-path "$BBOX_JSON_PATH")
fi

"${cmd[@]}"

echo "=== Done ==="
echo "Summary JSON: $OUTPUT_FILENAME"
echo "Detailed JSON: $DETAILED_OUTPUT_FILENAME"
