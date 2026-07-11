#!/usr/bin/env bash
set -euo pipefail

# Run the API memory-only evaluation on the first N TVQA+ samples.
# Defaults to N=50, enables VideoMemory, and disables verifier/refiner.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/zhangzheng/disk1/Tvqa_data}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"
MEMORY_MAX_ITEMS="${MEMORY_MAX_ITEMS:-8}"

QUESTIONS_PATH="${QUESTIONS_PATH:-$DATA_ROOT/results/subsets/tvqa_plus_first${SAMPLE_LIMIT}.json}"
SUBS_PATH="${SUBS_PATH:-$DATA_ROOT/hf_datasets/LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json}"
BASE_FRAME_DIR="${BASE_FRAME_DIR:-$DATA_ROOT/frames/bbt_frames}"
BBOX_JSON_PATH="${BBOX_JSON_PATH:-$DATA_ROOT/clip_bbox_mapping.json}"

RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/results}"
SUBSET_DIR="${SUBSET_DIR:-$RESULT_DIR/subsets}"
SUBSET_PATH="${SUBSET_PATH:-$SUBSET_DIR/tvqa_plus_first${SAMPLE_LIMIT}.json}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-$RESULT_DIR/memory_${SAMPLE_LIMIT}_summary.json}"
DETAILED_OUTPUT_FILENAME="${DETAILED_OUTPUT_FILENAME:-$RESULT_DIR/memory_${SAMPLE_LIMIT}_detail.json}"

THREADS="${THREADS:-5}"
MAX_TURN="${MAX_TURN:-5}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-api}"

GROUNDING_MODEL="${GROUNDING_MODEL:-grok-4-fast-reasoning}"
VISION_MODEL="${VISION_MODEL:-gpt-4o}"
MAIN_MODEL="${MAIN_MODEL:-grok-4-fast-reasoning}"
GROUNDING_BASE_URL="${GROUNDING_BASE_URL:-https://api2.aigcbest.top/v1}"
VISION_BASE_URL="${VISION_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MAIN_BASE_URL="${MAIN_BASE_URL:-https://api2.aigcbest.top/v1}"

GROUNDING_API_KEY="${GROUNDING_API_KEY:-${qdd_api:-}}"
VISION_API_KEY="${VISION_API_KEY:-${aliyun_api:-}}"
MAIN_API_KEY="${MAIN_API_KEY:-${qdd_api:-}}"

mkdir -p "$RESULT_DIR" "$SUBSET_DIR"

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
if [ ! -f "$BBOX_JSON_PATH" ]; then
  echo "[ERROR] BBox JSON file not found: $BBOX_JSON_PATH"
  exit 1
fi

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
echo " LongVideoAgent API Memory-Only Evaluation"
echo "============================================================"
echo "SAMPLE_LIMIT:             $SAMPLE_LIMIT"
echo "MEMORY_MAX_ITEMS:         $MEMORY_MAX_ITEMS"
echo "QUESTIONS_PATH:           $SUBSET_PATH"
echo "SUBS_PATH:                $SUBS_PATH"
echo "BASE_FRAME_DIR:           $BASE_FRAME_DIR"
echo "BBOX_JSON_PATH:           $BBOX_JSON_PATH"
echo "OUTPUT_FILENAME:          $OUTPUT_FILENAME"
echo "DETAILED_OUTPUT_FILENAME: $DETAILED_OUTPUT_FILENAME"
echo "THREADS:                  $THREADS"
echo "MAX_TURN:                 $MAX_TURN"
echo "============================================================"

USE_VIDEO_MEMORY=1 USE_VERIFIER=0 USE_CLIP_REFINER=0 MEMORY_MAX_ITEMS="$MEMORY_MAX_ITEMS" \
"$PYTHON_BIN" src/evaluation/lvagent/evaluate_api_unified.py \
  --dataset tvqa_plus \
  --checkpoint_step "$CHECKPOINT_STEP" \
  --max_turn "$MAX_TURN" \
  --threads "$THREADS" \
  --questions-path "$SUBSET_PATH" \
  --subs-path "$SUBS_PATH" \
  --base-frame-dir "$BASE_FRAME_DIR" \
  --bbox-json-path "$BBOX_JSON_PATH" \
  --output-filename "$OUTPUT_FILENAME" \
  --detailed-output-filename "$DETAILED_OUTPUT_FILENAME" \
  --grounding-model "$GROUNDING_MODEL" \
  --vision-model "$VISION_MODEL" \
  --main-model "$MAIN_MODEL" \
  --grounding-base-url "$GROUNDING_BASE_URL" \
  --vision-base-url "$VISION_BASE_URL" \
  --main-base-url "$MAIN_BASE_URL" \
  --grounding-api-key "$GROUNDING_API_KEY" \
  --vision-api-key "$VISION_API_KEY" \
  --main-api-key "$MAIN_API_KEY"
