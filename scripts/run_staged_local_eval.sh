#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGES="${STAGES:-10 20 50 full}"
STABILITY_DIR="${STABILITY_DIR:-./results/stability_local}"
RESET_REPORT="${RESET_REPORT:-0}"
POLICY_LABEL="${VISION_POLICY:-dynamic}"
POLICY_LABEL="${POLICY_LABEL//[^a-zA-Z0-9_-]/_}"
REPORT_PATH="${REPORT_PATH:-$STABILITY_DIR/report_${POLICY_LABEL}.md}"
DEFAULT_PYTHON_BIN="/home/zhangzheng/disk1/zhangzheng/conda_envs/videotemp_o3/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$DEFAULT_PYTHON_BIN" ]]; then
  export PYTHON_BIN="$DEFAULT_PYTHON_BIN"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p "$STABILITY_DIR"

if [[ "$RESET_REPORT" == "1" || ! -f "$REPORT_PATH" ]]; then
  cat > "$REPORT_PATH" <<'EOF'
# Local Evaluation Stability Report

| Stage | Total | Accuracy | Completion | Avg turns | Vision calls/q | Frames/q | Clips/q | Grounding/q | Time/q | Failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
EOF
fi

for stage in $STAGES; do
  if [[ "$stage" == "full" ]]; then
    eval_limit=0
    label="full"
  else
    eval_limit="$stage"
    label="${stage}"
  fi

  result_label="${label}_${POLICY_LABEL}"
  summary_path="$STABILITY_DIR/local_tvqa_plus_${result_label}_summary.json"
  detail_path="$STABILITY_DIR/local_tvqa_plus_${result_label}_detail.json"
  log_path="$STABILITY_DIR/local_tvqa_plus_${result_label}.log"
  checkpoint_path="$STABILITY_DIR/local_tvqa_plus_${result_label}_checkpoint.jsonl"

  echo "=== Stage: $label ==="
  EVAL_LIMIT="$eval_limit" \
  OUTPUT_FILENAME="$summary_path" \
  DETAILED_OUTPUT_FILENAME="$detail_path" \
  LOCAL_EVAL_LOG="$log_path" \
  CHECKPOINT_PATH="$checkpoint_path" \
  RESUME_EVAL=1 \
    bash scripts/eval_unified_local.sh

  "$PYTHON_BIN" - "$REPORT_PATH" "$summary_path" "$label-$POLICY_LABEL" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
summary = Path(sys.argv[2])
label = sys.argv[3]

with summary.open(encoding="utf-8") as f:
    data = json.load(f)

meta = data.get("metadata", {})
total = data.get("total", 0)
result_count = max(len(data.get("results", [])), 1)
vision_per_q = meta.get("vision_calls_total", 0) / result_count
grounding_per_q = meta.get("grounding_calls_total", 0) / result_count
frames_per_q = meta.get("vision_frames_total", 0) / result_count
clips_per_q = meta.get("vision_clips_total", 0) / result_count

row = (
    f"| {label} | {total} | {meta.get('accuracy', 0):.2%} | "
    f"{meta.get('completion_rate', 0):.2%} | {meta.get('avg_turns', 0):.2f} | "
    f"{vision_per_q:.2f} | {frames_per_q:.2f} | {clips_per_q:.2f} | "
    f"{grounding_per_q:.2f} | "
    f"{meta.get('avg_time_per_question', 0):.2f}s | {meta.get('failed_cases_count', 0)} |\n"
)

lines = report.read_text(encoding="utf-8").splitlines(keepends=True)
prefix = f"| {label} |"
for i, line in enumerate(lines):
    if line.startswith(prefix):
        lines[i] = row
        break
else:
    lines.append(row)
report.write_text("".join(lines), encoding="utf-8")
PY

  echo "Stage $label done."
  echo "Summary: $summary_path"
  echo "Detail: $detail_path"
  echo "Log: $log_path"
  echo "Checkpoint: $checkpoint_path"
done

echo "=== Stability evaluation done ==="
echo "Report: $REPORT_PATH"
