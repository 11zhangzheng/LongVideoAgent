"""Aggregate memory/verifier/refiner metrics from evaluation logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: str | None) -> Any:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_summary_path(detail_path: str) -> str | None:
    path = Path(detail_path)
    candidates = [
        path.with_name(path.name.replace("_detail", "_summary")),
        path.with_name(path.name.replace("detail_", "summary_")),
    ]
    for candidate in candidates:
        if candidate != path and candidate.is_file():
            return str(candidate)
    return None


def as_records(detail_data: Any) -> List[Dict[str, Any]]:
    if isinstance(detail_data, list):
        return [item for item in detail_data if isinstance(item, dict)]
    if isinstance(detail_data, dict):
        data = detail_data.get("results") or detail_data.get("data") or []
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def summary_results(summary_data: Any) -> List[Dict[str, Any]]:
    if isinstance(summary_data, dict) and isinstance(summary_data.get("results"), list):
        return [item for item in summary_data["results"] if isinstance(item, dict)]
    return []


def compute_metrics(detail_data: Any, summary_data: Any = None) -> Dict[str, Any]:
    details = as_records(detail_data)
    summaries = summary_results(summary_data)
    total = max(len(details), len(summaries))

    turns_by_question = [record.get("turns", []) for record in details]
    summary_by_question = summaries if summaries else [{} for _ in details]
    accuracy_records = summaries if summaries else details

    accuracy = summary_accuracy(summary_data)
    if accuracy is None:
        accuracy = compute_accuracy(accuracy_records)
    avg_turns = avg(
        [
            len(turns) if turns else int(summary.get("num_turns", 0) or 0)
            for turns, summary in zip_longest_records(turns_by_question, summary_by_question)
        ]
    )
    avg_vision_calls = avg([count_actions(turns, "search") for turns in turns_by_question])
    avg_grounding_calls = avg([count_actions(turns, "request_grounding") for turns in turns_by_question])

    verifier_results = [vr for record in details for vr in get_verifier_results(record)]
    avg_verifier_calls = avg([len(get_verifier_results(record)) for record in details])
    verifier_pass_rate = safe_div(
        sum(1 for result in verifier_results if result.get("verdict") == "PASS"),
        len(verifier_results),
    )

    observed_clip_counts: List[int] = []
    repeated_rates: List[float] = []
    selected_frame_counts: List[int] = []
    for record in details:
        observed_clips = observed_clips_for_record(record)
        observed_clip_counts.append(len(set(observed_clips)))
        repeated_rates.append(repeated_rate(observed_clips))
        selected_frame_counts.extend(selected_frame_lengths(record))

    return {
        "total_questions": total,
        "accuracy": accuracy,
        "avg_turns": avg_turns,
        "avg_vision_calls": avg_vision_calls,
        "avg_grounding_calls": avg_grounding_calls,
        "avg_verifier_calls": avg_verifier_calls,
        "avg_observed_clips": avg(observed_clip_counts),
        "repeated_clip_observation_rate": avg(repeated_rates),
        "avg_selected_frames": avg(selected_frame_counts),
        "avg_latency_per_question": avg([float(record.get("elapsed_seconds", 0.0) or 0.0) for record in details]),
        "verifier_pass_rate": verifier_pass_rate,
        "verifier_fail_to_correct_rate": verifier_fail_to_correct_rate(details, summaries or details),
    }


def zip_longest_records(turns_list: List[List[Dict[str, Any]]], summaries: List[Dict[str, Any]]):
    total = max(len(turns_list), len(summaries))
    for idx in range(total):
        turns = turns_list[idx] if idx < len(turns_list) else []
        summary = summaries[idx] if idx < len(summaries) else {}
        yield turns, summary


def compute_accuracy(summaries: List[Dict[str, Any]]) -> float:
    if not summaries:
        return 0.0
    correct = 0
    total = 0
    for record in summaries:
        gt_answer_idx = record.get("gt_answer_idx")
        final_answer = str(record.get("final_answer", "")).strip().lower()
        if gt_answer_idx is None:
            continue
        total += 1
        if final_answer == f"a{gt_answer_idx}".lower():
            correct += 1
    return safe_div(correct, total)


def summary_accuracy(summary_data: Any) -> float | None:
    if not isinstance(summary_data, dict):
        return None
    metadata = summary_data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("accuracy") is not None:
        return float(metadata["accuracy"])
    if summary_data.get("accuracy") is not None:
        return float(summary_data["accuracy"])
    return None


def count_actions(turns: List[Dict[str, Any]], action_type: str) -> int:
    return sum(1 for turn in turns if turn.get("action_type") == action_type)


def get_verifier_results(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(record.get("verifier_results"), list):
        return [item for item in record["verifier_results"] if isinstance(item, dict)]
    results = []
    for turn in record.get("turns", []):
        result = turn.get("verifier_result")
        if isinstance(result, dict):
            results.append(result)
    result = record.get("verifier_result")
    if isinstance(result, dict) and result not in results:
        results.append(result)
    return results


def observed_clips_for_record(record: Dict[str, Any]) -> List[str]:
    clips: List[str] = []
    memory = record.get("memory")
    if isinstance(memory, dict):
        search_memory = memory.get("search_memory", [])
        if isinstance(search_memory, list):
            for item in search_memory:
                if isinstance(item, dict):
                    clips.extend(str(clip_id) for clip_id in item.get("clip_ids", []))
        clip_memory = memory.get("clip_memory", {})
        if isinstance(clip_memory, dict):
            clips.extend(str(clip_id) for clip_id in clip_memory.keys())

    for turn in record.get("turns", []):
        decision = turn.get("vision_decision", {})
        if isinstance(decision, dict):
            clips.extend(str(clip_id) for clip_id in decision.get("clip_ids", []))
    return clips


def selected_frame_lengths(record: Dict[str, Any]) -> List[int]:
    lengths: List[int] = []
    memory = record.get("memory")
    if isinstance(memory, dict):
        search_memory = memory.get("search_memory", [])
        if isinstance(search_memory, list):
            for item in search_memory:
                if not isinstance(item, dict):
                    continue
                result = item.get("result", {})
                if isinstance(result, dict):
                    lengths.extend(frame_lengths_from_mapping(result.get("selected_frames", {})))

    for turn in record.get("turns", []):
        decision = turn.get("vision_decision", {})
        if isinstance(decision, dict):
            lengths.extend(frame_lengths_from_mapping(decision.get("selected_frames", {})))
    return lengths


def frame_lengths_from_mapping(value: Any) -> List[int]:
    if not isinstance(value, dict):
        return []
    return [len(frames) for frames in value.values() if isinstance(frames, list)]


def repeated_rate(values: List[str]) -> float:
    if not values:
        return 0.0
    return safe_div(len(values) - len(set(values)), len(values))


def verifier_fail_to_correct_rate(details: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> float:
    if not summaries:
        return 0.0
    corrected = 0
    candidates = 0
    for record, summary in zip_longest_records(details, summaries):
        verifier_results = get_verifier_results(record)
        if not any(result.get("verdict") in ("FAIL", "UNCERTAIN") for result in verifier_results):
            continue
        gt_answer_idx = summary.get("gt_answer_idx")
        if gt_answer_idx is None:
            continue
        candidates += 1
        final_answer = str(summary.get("final_answer", "")).strip().lower()
        if final_answer == f"a{gt_answer_idx}".lower():
            corrected += 1
    return safe_div(corrected, candidates)


def avg(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute memory/verifier metrics from detailed logs.")
    parser.add_argument("--detail", required=True, help="Detailed log JSON path")
    parser.add_argument("--summary", default=None, help="Summary JSON path; inferred when omitted")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    summary_path = args.summary or infer_summary_path(args.detail)
    metrics = compute_metrics(load_json(args.detail), load_json(summary_path))

    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
