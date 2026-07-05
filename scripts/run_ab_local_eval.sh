#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LongVideoAgent Local Evaluation A/B Test Script
# fixed vs dynamic vision policy on the SAME samples
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# -----------------------------
# User-configurable variables
# -----------------------------
if [ -f "$ROOT_DIR/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/env.sh"
fi

GROUNDING_API_KEY="${GROUNDING_API_KEY:-${qdd_api:-${DASHSCOPE_API_KEY:-}}}"
VISION_API_KEY="${VISION_API_KEY:-${aliyun_api:-${DASHSCOPE_API_KEY:-}}}"

GROUNDING_BASE_URL="${GROUNDING_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
VISION_BASE_URL="${VISION_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

GROUNDING_MODEL="${GROUNDING_MODEL:-qwen-plus}"
VISION_MODEL="${VISION_MODEL:-qwen-vl-plus}"

DATASET="${DATASET:-tvqa_plus}"

# 数据路径
DATA_ROOT="${DATA_ROOT:-/home/zhangzheng/disk1/Tvqa_data}"
QUESTIONS_PATH="${QUESTIONS_PATH:-$DATA_ROOT/LongTVQA_plus_val_normalized.json}"
SUBS_PATH="${SUBS_PATH:-$DATA_ROOT/hf_datasets/LongTVQA/LongTVQA_subtitles_clip_level.jsonl}"
BASE_FRAME_DIR="${BASE_FRAME_DIR:-$DATA_ROOT/frames/bbt_frames}"
BBOX_JSON_PATH="${BBOX_JSON_PATH:-$DATA_ROOT/clip_bbox_mapping.json}"

# 模型路径
MODEL_ROOT="${MODEL_ROOT:-/home/zhangzheng/disk1/models}"
LLM_PATH="${LLM_PATH:-$MODEL_ROOT/longvideoagent-qwen2.5-3b}"

# Python 环境
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzheng/disk1/zhangzheng/conda_envs/videotemp_o3/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

# 评测阶段：默认跑前 10/20/50 个样本
STAGES="${STAGES:-10 20 50}"

# GPU 设置
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# vLLM 设置
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_TURN="${MAX_TURN:-5}"

# 结果目录
RESULT_ROOT="${RESULT_ROOT:-results/ab_local_eval}"

# 只检查配置，不启动评测
DRY_RUN="${DRY_RUN:-0}"

# 是否强制重跑。1 表示删除已有同名结果重新跑
FORCE_RERUN="${FORCE_RERUN:-1}"

# dynamic 策略可调参数
VISION_MIN_FRAMES="${VISION_MIN_FRAMES:-4}"
VISION_MAX_FRAMES="${VISION_MAX_FRAMES:-20}"
VISION_MAX_CLIPS="${VISION_MAX_CLIPS:-2}"
VISION_MAX_CALLS="${VISION_MAX_CALLS:-2}"

# 是否允许 vLLM 超过 config max length
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16000}"
VLLM_DTYPE="${VLLM_DTYPE:-half}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export VLLM_ALLOW_LONG_MAX_MODEL_LEN
export VLLM_MAX_MODEL_LEN
export VLLM_DTYPE
export PYTORCH_CUDA_ALLOC_CONF

# -----------------------------
# Basic checks
# -----------------------------
if [ ! -f "$QUESTIONS_PATH" ]; then
  echo "[ERROR] Questions file not found: $QUESTIONS_PATH"
  exit 1
fi

if [ ! -d "$LLM_PATH" ]; then
  echo "[ERROR] LLM path not found: $LLM_PATH"
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

if [[ "$DATASET" == "tvqa_plus" && ! -f "$BBOX_JSON_PATH" ]]; then
  echo "[ERROR] BBox JSON file not found: $BBOX_JSON_PATH"
  exit 1
fi

if [[ -z "$GROUNDING_API_KEY" || -z "$VISION_API_KEY" ]]; then
  echo "[ERROR] Please set GROUNDING_API_KEY and VISION_API_KEY, or source env.sh."
  exit 1
fi

if [ ! -f "src/evaluation/lvagent/evaluate_local_unified.py" ]; then
  echo "[ERROR] Please run this script from the LongVideoAgent project root."
  exit 1
fi

mkdir -p "$RESULT_ROOT"
mkdir -p "$RESULT_ROOT/subsets"
mkdir -p "$RESULT_ROOT/fixed"
mkdir -p "$RESULT_ROOT/dynamic"
mkdir -p "$RESULT_ROOT/reports"

echo "============================================================"
echo " LongVideoAgent A/B Local Evaluation"
echo "============================================================"
echo "DATASET:               $DATASET"
echo "QUESTIONS_PATH:        $QUESTIONS_PATH"
echo "SUBS_PATH:             $SUBS_PATH"
echo "BASE_FRAME_DIR:        $BASE_FRAME_DIR"
echo "BBOX_JSON_PATH:        $BBOX_JSON_PATH"
echo "LLM_PATH:              $LLM_PATH"
echo "PYTHON_BIN:            $PYTHON_BIN"
echo "STAGES:                $STAGES"
echo "CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "MAX_TURN:              $MAX_TURN"
echo "GPU_MEMORY_UTILIZATION:$GPU_MEMORY_UTILIZATION"
echo "VLLM_MAX_MODEL_LEN:    $VLLM_MAX_MODEL_LEN"
echo "PYTORCH_CUDA_ALLOC_CONF:$PYTORCH_CUDA_ALLOC_CONF"
echo "RESULT_ROOT:           $RESULT_ROOT"
echo "DRY_RUN:               $DRY_RUN"
echo "FORCE_RERUN:           $FORCE_RERUN"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY_RUN] Basic checks passed. No evaluation started."
  exit 0
fi

# -----------------------------
# Helper: create first-N subset
# -----------------------------
make_subset() {
  local stage="$1"
  local subset_path="$RESULT_ROOT/subsets/${DATASET}_first${stage}.json"

  "$PYTHON_BIN" - "$QUESTIONS_PATH" "$subset_path" "$stage" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
n = int(sys.argv[3])

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    subset = data[:n]
elif isinstance(data, dict):
    # 常见情况：某个 key 下是 list，例如 {"data": [...]}
    subset = {}
    cut_done = False
    for k, v in data.items():
        if isinstance(v, list):
            subset[k] = v[:n]
            cut_done = True
        else:
            subset[k] = v

    # 如果 dict 不是 {"data": [...]} 结构，而是 qid -> item 结构
    # 则取前 n 个 key
    if not cut_done:
        items = list(data.items())[:n]
        subset = dict(items)
else:
    raise TypeError(f"Unsupported JSON type: {type(data)}")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(dst)
PY
}

# -----------------------------
# Helper: run one policy on one stage
# -----------------------------
run_one() {
  local policy="$1"
  local stage="$2"
  local subset_path="$3"

  local policy_dir="$RESULT_ROOT/$policy"
  local summary_file="$policy_dir/summary_${policy}_${stage}.json"
  local detail_file="$policy_dir/detail_${policy}_${stage}.json"
  local log_file="$policy_dir/run_${policy}_${stage}.log"

  mkdir -p "$policy_dir"

  if [ "$FORCE_RERUN" = "1" ]; then
    rm -f "$summary_file" "$detail_file" "$log_file"
  fi

  if [ -f "$summary_file" ] && [ -f "$detail_file" ]; then
    echo "[SKIP] $policy stage=$stage already exists."
    return 0
  fi

  echo "------------------------------------------------------------"
  echo "[RUN] policy=$policy stage=$stage"
  echo "[RUN] subset=$subset_path"
  echo "[RUN] summary=$summary_file"
  echo "[RUN] detail=$detail_file"
  echo "------------------------------------------------------------"

  if [ "$policy" = "dynamic" ]; then
    VISION_POLICY="dynamic" \
    VISION_MIN_FRAMES="$VISION_MIN_FRAMES" \
    VISION_MAX_FRAMES="$VISION_MAX_FRAMES" \
    VISION_MAX_CLIPS="$VISION_MAX_CLIPS" \
    VISION_MAX_CALLS="$VISION_MAX_CALLS" \
    "$PYTHON_BIN" src/evaluation/lvagent/evaluate_local_unified.py \
      --dataset "$DATASET" \
      --questions-path "$subset_path" \
      --llm-path "$LLM_PATH" \
      --max_turn "$MAX_TURN" \
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
      --subs-path "$SUBS_PATH" \
      --base-frame-dir "$BASE_FRAME_DIR" \
      --bbox-json-path "$BBOX_JSON_PATH" \
      --output-filename "$summary_file" \
      --detailed-output-filename "$detail_file" \
      --grounding-model "$GROUNDING_MODEL" \
      --vision-model "$VISION_MODEL" \
      --grounding-base-url "$GROUNDING_BASE_URL" \
      --vision-base-url "$VISION_BASE_URL" \
      --grounding-api-key "$GROUNDING_API_KEY" \
      --vision-api-key "$VISION_API_KEY" \
      2>&1 | tee "$log_file"
  else
    VISION_POLICY="fixed" "$PYTHON_BIN" src/evaluation/lvagent/evaluate_local_unified.py \
  --dataset "$DATASET" \
  --questions-path "$subset_path" \
  --llm-path "$LLM_PATH" \
  --max_turn "$MAX_TURN" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --subs-path "$SUBS_PATH" \
  --base-frame-dir "$BASE_FRAME_DIR" \
  --bbox-json-path "$BBOX_JSON_PATH" \
  --output-filename "$summary_file" \
  --detailed-output-filename "$detail_file" \
  --grounding-model "$GROUNDING_MODEL" \
  --vision-model "$VISION_MODEL" \
  --grounding-base-url "$GROUNDING_BASE_URL" \
  --vision-base-url "$VISION_BASE_URL" \
  --grounding-api-key "$GROUNDING_API_KEY" \
  --vision-api-key "$VISION_API_KEY" 2>&1 | tee "$log_file"
  fi
}

# -----------------------------
# Helper: compare fixed/dynamic sample identity
# -----------------------------
compare_ids() {
  local stage="$1"
  local fixed_detail="$RESULT_ROOT/fixed/detail_fixed_${stage}.json"
  local dynamic_detail="$RESULT_ROOT/dynamic/detail_dynamic_${stage}.json"

  if [ ! -f "$fixed_detail" ] || [ ! -f "$dynamic_detail" ]; then
    echo "[WARN] Cannot compare ids for stage=$stage because detail file missing."
    return 0
  fi

  "$PYTHON_BIN" - "$fixed_detail" "$dynamic_detail" "$stage" <<'PY'
import json
import sys
from pathlib import Path

fixed_path = Path(sys.argv[1])
dynamic_path = Path(sys.argv[2])
stage = sys.argv[3]

def load_items(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 如果有明显 results/data 字段
        for key in ["results", "data", "items", "details"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        # 否则当成 qid -> item
        return list(data.values())
    return []

def get_id(item, idx):
    if not isinstance(item, dict):
        return str(idx)
    for key in [
        "qid", "question_id", "id", "sample_id",
        "vid_name", "video_id", "clip_id", "question"
    ]:
        if key in item and item[key] is not None:
            return str(item[key])
    return str(idx)

fixed_items = load_items(fixed_path)
dynamic_items = load_items(dynamic_path)

fixed_ids = [get_id(x, i) for i, x in enumerate(fixed_items)]
dynamic_ids = [get_id(x, i) for i, x in enumerate(dynamic_items)]

same_len = len(fixed_ids) == len(dynamic_ids)
same_order = fixed_ids == dynamic_ids

print(f"[CHECK stage={stage}] fixed={len(fixed_ids)} dynamic={len(dynamic_ids)}")
print(f"[CHECK stage={stage}] same_length={same_len}")
print(f"[CHECK stage={stage}] same_order={same_order}")

if not same_order:
    for i, (a, b) in enumerate(zip(fixed_ids, dynamic_ids)):
        if a != b:
            print(f"[CHECK stage={stage}] first_mismatch_index={i}")
            print(f"[CHECK stage={stage}] fixed_id={a}")
            print(f"[CHECK stage={stage}] dynamic_id={b}")
            break
    sys.exit(2)
PY
}

# -----------------------------
# Main loop
# -----------------------------
for STAGE in $STAGES; do
  echo
  echo "==================== Stage: first $STAGE samples ===================="

  SUBSET_PATH="$(make_subset "$STAGE")"

  # 关键：fixed 和 dynamic 使用同一个 SUBSET_PATH
  run_one "fixed" "$STAGE" "$SUBSET_PATH"
  run_one "dynamic" "$STAGE" "$SUBSET_PATH"

  # 检查 detailed JSON 中的样本是否一致
  compare_ids "$STAGE"

  echo "[DONE] Stage $STAGE completed."
done

# -----------------------------
# Make a lightweight markdown report
# -----------------------------
REPORT_FILE="$RESULT_ROOT/reports/ab_report.md"

"$PYTHON_BIN" - "$RESULT_ROOT" "$REPORT_FILE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = Path(sys.argv[2])

def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick(d, keys, default=""):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default

rows = []
for fixed_summary in sorted((root / "fixed").glob("summary_fixed_*.json")):
    stage = fixed_summary.stem.split("_")[-1]
    dyn_summary = root / "dynamic" / f"summary_dynamic_{stage}.json"

    f = load_json(fixed_summary)
    d = load_json(dyn_summary)

    if f is None or d is None:
        continue

    rows.append({
        "stage": stage,
        "fixed_acc": pick(f, ["accuracy", "Accuracy", "acc"]),
        "dynamic_acc": pick(d, ["accuracy", "Accuracy", "acc"]),
        "fixed_completion": pick(f, ["completion_rate", "Completion rate", "completion"]),
        "dynamic_completion": pick(d, ["completion_rate", "Completion rate", "completion"]),
        "fixed_turns": pick(f, ["average_turns", "avg_turns", "Average turns per question"]),
        "dynamic_turns": pick(d, ["average_turns", "avg_turns", "Average turns per question"]),
        "fixed_vision": pick(f, ["vision_calls_per_question", "avg_vision_calls", "Vision calls per question"]),
        "dynamic_vision": pick(d, ["vision_calls_per_question", "avg_vision_calls", "Vision calls per question"]),
    })

lines = []
lines.append("# Fixed vs Dynamic Vision Policy A/B Report")
lines.append("")
lines.append("| Stage | Fixed Acc | Dynamic Acc | Fixed Completion | Dynamic Completion | Fixed Turns | Dynamic Turns | Fixed Vision/Q | Dynamic Vision/Q |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    lines.append(
        f"| {r['stage']} | {r['fixed_acc']} | {r['dynamic_acc']} | "
        f"{r['fixed_completion']} | {r['dynamic_completion']} | "
        f"{r['fixed_turns']} | {r['dynamic_turns']} | "
        f"{r['fixed_vision']} | {r['dynamic_vision']} |"
    )

report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(lines), encoding="utf-8")
print(f"[REPORT] saved to {report}")
PY

echo
echo "============================================================"
echo "[ALL DONE]"
echo "Results:"
echo "  fixed:   $RESULT_ROOT/fixed"
echo "  dynamic: $RESULT_ROOT/dynamic"
echo "Report:"
echo "  $REPORT_FILE"
echo "============================================================"
