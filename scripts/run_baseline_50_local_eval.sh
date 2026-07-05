#!/usr/bin/env bash
set -euo pipefail

# Run the local MasterAgent baseline on the first N TVQA+ samples.
# Memory, verifier, and clip refiner are disabled by default.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-/home/zhangzheng/disk1/zhangzheng/conda_envs/videotemp_o3/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

DATA_ROOT="${DATA_ROOT:-/home/zhangzheng/disk1/Tvqa_data}"
MODEL_ROOT="${MODEL_ROOT:-/home/zhangzheng/disk1/models}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-50}"

DATASET="${DATASET:-tvqa_plus}"
QUESTIONS_PATH="${QUESTIONS_PATH:-$DATA_ROOT/LongTVQA_plus_val_normalized.json}"
SUBS_PATH="${SUBS_PATH:-$DATA_ROOT/hf_datasets/LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json}"
BASE_FRAME_DIR="${BASE_FRAME_DIR:-$DATA_ROOT/frames/bbt_frames}"
BBOX_JSON_PATH="${BBOX_JSON_PATH:-$DATA_ROOT/clip_bbox_mapping.json}"
LLM_PATH="${LLM_PATH:-$MODEL_ROOT/longvideoagent-qwen2.5-3b}"

RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/results/local_baseline_50}"
SUBSET_DIR="${SUBSET_DIR:-$RESULT_DIR/subsets}"
SUBSET_PATH="${SUBSET_PATH:-$SUBSET_DIR/${DATASET}_first${SAMPLE_LIMIT}.json}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-$RESULT_DIR/baseline_${SAMPLE_LIMIT}_summary.json}"
DETAILED_OUTPUT_FILENAME="${DETAILED_OUTPUT_FILENAME:-$RESULT_DIR/baseline_${SAMPLE_LIMIT}_detail.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RESULT_DIR/baseline_${SAMPLE_LIMIT}.checkpoint.jsonl}"
LOG_FILE="${LOG_FILE:-$RESULT_DIR/baseline_${SAMPLE_LIMIT}.log}"

CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_TURN="${MAX_TURN:-5}"

GROUNDING_MODEL="${GROUNDING_MODEL:-grok-4-fast-reasoning}"
VISION_MODEL="${VISION_MODEL:-gpt-4o}"
GROUNDING_BASE_URL="${GROUNDING_BASE_URL:-https://api2.aigcbest.top/v1}"
VISION_BASE_URL="${VISION_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
GROUNDING_API_KEY="${GROUNDING_API_KEY:-${qdd_api:-${DASHSCOPE_API_KEY:-}}}"
VISION_API_KEY="${VISION_API_KEY:-${aliyun_api:-${DASHSCOPE_API_KEY:-}}}"

VISION_POLICY="${VISION_POLICY:-fixed}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16000}"
VLLM_DTYPE="${VLLM_DTYPE:-half}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DRY_RUN="${DRY_RUN:-0}"

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export VLLM_ALLOW_LONG_MAX_MODEL_LEN
export VLLM_MAX_MODEL_LEN
export VLLM_DTYPE
export PYTORCH_CUDA_ALLOC_CONF

if [ ! -f "$QUESTIONS_PATH" ]; then
  echo "[ERROR] Questions file not found: $QUESTIONS_PATH"
  exit 1
fi
if [ ! -f "$SUBS_PATH" ]; then
  echo "[ERROR] Subtitles file not found: $SUBS_PATH"
  exit 1
fi
if [ ! -d "$BASE_FRAME_DIR" ]; then
  echo "[ERROR] Frame directory not found: $BASE_FRAME_DIR"
  exit 1
fi
if [ "$DATASET" = "tvqa_plus" ] && [ ! -f "$BBOX_JSON_PATH" ]; then
  echo "[ERROR] BBox JSON file not found: $BBOX_JSON_PATH"
  exit 1
fi
if [ ! -d "$LLM_PATH" ]; then
  echo "[ERROR] LLM path not found: $LLM_PATH"
  exit 1
fi
if [[ -z "$GROUNDING_API_KEY" || -z "$VISION_API_KEY" ]]; then
  echo "[ERROR] Please set GROUNDING_API_KEY and VISION_API_KEY, or source env.sh."
  exit 1
fi

mkdir -p "$RESULT_DIR" "$SUBSET_DIR"

"$PYTHON_BIN" - "$QUESTIONS_PATH" "$SUBSET_PATH" "$SAMPLE_LIMIT" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise TypeError(f"Expected a JSON list, got {type(data).__name__}: {src}")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(data[:limit], f, ensure_ascii=False, indent=2)

print(f"[INFO] Wrote first {min(limit, len(data))} samples to {dst}")
PY

echo "============================================================"
echo " LongVideoAgent Local Baseline Evaluation"
echo "============================================================"
echo "DATASET:                  $DATASET"
echo "SAMPLE_LIMIT:             $SAMPLE_LIMIT"
echo "QUESTIONS_PATH:           $SUBSET_PATH"
echo "SUBS_PATH:                $SUBS_PATH"
echo "BASE_FRAME_DIR:           $BASE_FRAME_DIR"
echo "BBOX_JSON_PATH:           $BBOX_JSON_PATH"
echo "LLM_PATH:                 $LLM_PATH"
echo "PYTHON_BIN:               $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES:     $CUDA_VISIBLE_DEVICES"
echo "GPU_MEMORY_UTILIZATION:   $GPU_MEMORY_UTILIZATION"
echo "VISION_POLICY:            $VISION_POLICY"
echo "OUTPUT_FILENAME:          $OUTPUT_FILENAME"
echo "DETAILED_OUTPUT_FILENAME: $DETAILED_OUTPUT_FILENAME"
echo "CHECKPOINT_PATH:          $CHECKPOINT_PATH"
echo "LOG_FILE:                 $LOG_FILE"
echo "DRY_RUN:                  $DRY_RUN"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY_RUN] Basic checks passed. No evaluation started."
  exit 0
fi

USE_VIDEO_MEMORY=0 \
USE_VERIFIER=0 \
USE_CLIP_REFINER=0 \
VISION_POLICY="$VISION_POLICY" \
"$PYTHON_BIN" src/evaluation/lvagent/evaluate_local_unified.py \
  --dataset "$DATASET" \
  --llm-path "$LLM_PATH" \
  --max_turn "$MAX_TURN" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --questions-path "$SUBSET_PATH" \
  --subs-path "$SUBS_PATH" \
  --base-frame-dir "$BASE_FRAME_DIR" \
  --bbox-json-path "$BBOX_JSON_PATH" \
  --output-filename "$OUTPUT_FILENAME" \
  --detailed-output-filename "$DETAILED_OUTPUT_FILENAME" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --grounding-model "$GROUNDING_MODEL" \
  --vision-model "$VISION_MODEL" \
  --grounding-base-url "$GROUNDING_BASE_URL" \
  --vision-base-url "$VISION_BASE_URL" \
  --grounding-api-key "$GROUNDING_API_KEY" \
  --vision-api-key "$VISION_API_KEY" \
  2>&1 | tee "$LOG_FILE"
