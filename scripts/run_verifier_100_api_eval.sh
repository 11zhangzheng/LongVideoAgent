#!/usr/bin/env bash
set -euo pipefail

# Run the API verifier-only evaluation on the fixed first-100 TVQA+ subset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/zhangzheng/disk1/Tvqa_data}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"

SUBSET_PATH="${SUBSET_PATH:-$ROOT_DIR/results/subsets/tvqa_plus_first${SAMPLE_LIMIT}.json}"
SUBS_PATH="${SUBS_PATH:-$DATA_ROOT/hf_datasets/LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json}"
BASE_FRAME_DIR="${BASE_FRAME_DIR:-$DATA_ROOT/frames/bbt_frames}"
BBOX_JSON_PATH="${BBOX_JSON_PATH:-$DATA_ROOT/clip_bbox_mapping.json}"

RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/results/api_verifier_${SAMPLE_LIMIT}}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-$RESULT_DIR/verifier_${SAMPLE_LIMIT}_summary.json}"
DETAILED_OUTPUT_FILENAME="${DETAILED_OUTPUT_FILENAME:-$RESULT_DIR/verifier_${SAMPLE_LIMIT}_detail.json}"

THREADS="${THREADS:-5}"
MAX_TURN="${MAX_TURN:-6}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-api}"
VERIFIER_MAX_ROUNDS="${VERIFIER_MAX_ROUNDS:-1}"

GROUNDING_MODEL="${GROUNDING_MODEL:-grok-4-fast-reasoning}"
MAIN_MODEL="${MAIN_MODEL:-grok-4-fast-reasoning}"
VISION_MODEL="${VISION_MODEL:-qwen-vl-max}"
GROUNDING_BASE_URL="${GROUNDING_BASE_URL:-https://api2.aigcbest.top/v1}"
MAIN_BASE_URL="${MAIN_BASE_URL:-https://api2.aigcbest.top/v1}"
VISION_BASE_URL="${VISION_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

GROUNDING_API_KEY="${GROUNDING_API_KEY:-${qdd_api:-}}"
MAIN_API_KEY="${MAIN_API_KEY:-${qdd_api:-}}"
VISION_API_KEY="${VISION_API_KEY:-${aliyun_api:-}}"

mkdir -p "$RESULT_DIR"

if [ ! -f "$SUBSET_PATH" ]; then
  echo "[ERROR] Fixed subset file not found: $SUBSET_PATH"
  echo "        Create it first, or pass SUBSET_PATH=/path/to/fixed_subset.json"
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

echo "============================================================"
echo " LongVideoAgent API Verifier-Only Evaluation"
echo "============================================================"
echo "SAMPLE_LIMIT:             $SAMPLE_LIMIT"
echo "SUBSET_PATH:              $SUBSET_PATH"
echo "SUBS_PATH:                $SUBS_PATH"
echo "BASE_FRAME_DIR:           $BASE_FRAME_DIR"
echo "BBOX_JSON_PATH:           $BBOX_JSON_PATH"
echo "OUTPUT_FILENAME:          $OUTPUT_FILENAME"
echo "DETAILED_OUTPUT_FILENAME: $DETAILED_OUTPUT_FILENAME"
echo "THREADS:                  $THREADS"
echo "MAX_TURN:                 $MAX_TURN"
echo "VERIFIER_MAX_ROUNDS:      $VERIFIER_MAX_ROUNDS"
echo "GROUNDING_MODEL:          $GROUNDING_MODEL"
echo "MAIN_MODEL:               $MAIN_MODEL"
echo "VISION_MODEL:             $VISION_MODEL"
echo "============================================================"

USE_VIDEO_MEMORY=0 \
USE_VERIFIER=1 \
USE_CLIP_REFINER=0 \
VERIFIER_MAX_ROUNDS="$VERIFIER_MAX_ROUNDS" \
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
  --main-model "$MAIN_MODEL" \
  --vision-model "$VISION_MODEL" \
  --grounding-base-url "$GROUNDING_BASE_URL" \
  --main-base-url "$MAIN_BASE_URL" \
  --vision-base-url "$VISION_BASE_URL" \
  --grounding-api-key "$GROUNDING_API_KEY" \
  --main-api-key "$MAIN_API_KEY" \
  --vision-api-key "$VISION_API_KEY"
