"""Lightweight clip refinement helpers for evaluation-time search."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List


SUPPORTED_ACTIONS = {
    "expand_previous",
    "expand_next",
    "expand_both",
    "dense_resample_current_clip",
}


def use_clip_refiner() -> bool:
    value = os.getenv("USE_CLIP_REFINER", "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def infer_question_type(question: str, feedback_text: str = "") -> str:
    text = f"{question} {feedback_text}".lower()
    if re.search(r"\b(before|after|then|next|previous|earlier|later|while|during)\b", text):
        return "temporal_action"
    if re.search(r"\b(wear|wearing|hold|holding|color|visible|object|gesture|look|scene)\b", text):
        return "visual_detail"
    if re.search(r"\b(why|feel|feeling|emotion|reaction|upset|angry|happy|sad)\b", text):
        return "causal_emotion"
    if re.search(r"\b(say|said|tell|told|ask|asked|talk|reply|conversation)\b", text):
        return "dialogue"
    return "general"


def plan_clip_refinement(
    verifier_result: Dict[str, Any],
    question: str,
    question_type: str | None = None,
) -> Dict[str, Any]:
    """Pick one supported refinement action from verifier feedback."""
    suggested = verifier_result.get("suggested_action")
    if isinstance(suggested, dict):
        action = suggested.get("action") or suggested.get("type")
        if action in SUPPORTED_ACTIONS:
            return {
                "action": action,
                "reason": str(suggested.get("reason", verifier_result.get("reason", ""))),
                "source": "verifier_suggested_action",
            }

    feedback_text = " ".join(
        [
            str(verifier_result.get("reason", "")),
            " ".join(str(x) for x in verifier_result.get("missing_evidence", [])),
            " ".join(str(x) for x in verifier_result.get("contradictions", [])),
        ]
    )
    qtype = question_type or infer_question_type(question, feedback_text)
    lowered = f"{question} {feedback_text}".lower()

    if re.search(r"\b(before|previous|earlier)\b", lowered):
        action = "expand_previous"
    elif re.search(r"\b(after|next|later)\b", lowered):
        action = "expand_next"
    elif qtype in ("temporal_action", "causal_emotion"):
        action = "expand_both"
    elif qtype == "visual_detail":
        action = "dense_resample_current_clip"
    elif "wrong clip" in lowered or "different clip" in lowered:
        action = "expand_both"
    else:
        action = "dense_resample_current_clip"

    return {
        "action": action,
        "reason": verifier_result.get("reason", ""),
        "source": "heuristic",
        "question_type": qtype,
    }


def apply_refinement_to_decision(
    decision: Dict[str, Any],
    current_clip: str,
    base_frame_dir: str,
    adjacent_clip_ids_fn: Callable[[str, str, str, int], List[str]],
    refinement_action: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Apply a planned refinement to an existing vision decision."""
    if not refinement_action:
        return decision

    action = refinement_action.get("action")
    if action not in SUPPORTED_ACTIONS:
        return decision

    refined = dict(decision)
    original_clip_ids = list(refined.get("clip_ids", [current_clip]))
    original_frame_budget = int(refined.get("frame_budget", 0))
    max_frames = int(os.getenv("CLIP_REFINER_MAX_FRAMES", os.getenv("VISION_MAX_FRAMES", "24")))
    min_per_clip = int(os.getenv("CLIP_REFINER_MIN_FRAMES_PER_CLIP", "4"))

    if action == "dense_resample_current_clip":
        bonus = int(os.getenv("CLIP_REFINER_DENSE_FRAME_BONUS", "8"))
        refined["clip_ids"] = [current_clip]
        refined["frame_budget"] = max(original_frame_budget, min(max_frames, original_frame_budget + bonus))
    else:
        direction = {
            "expand_previous": "previous",
            "expand_next": "next",
            "expand_both": "both",
        }[action]
        limit = int(os.getenv("CLIP_REFINER_EXPAND_LIMIT", "2"))
        expanded = adjacent_clip_ids_fn(base_frame_dir, current_clip, direction, limit)
        clip_ids = _unique([current_clip] + original_clip_ids + expanded)
        refined["clip_ids"] = clip_ids
        refined["direction"] = direction
        refined["frame_budget"] = max(
            original_frame_budget,
            min(max_frames, max(len(clip_ids), 1) * min_per_clip),
        )

    refined["clip_refinement_action"] = {
        **refinement_action,
        "applied": True,
        "original_clip_ids": original_clip_ids,
        "refined_clip_ids": refined.get("clip_ids", []),
        "original_frame_budget": original_frame_budget,
        "refined_frame_budget": refined.get("frame_budget", original_frame_budget),
    }
    return refined


def _unique(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _self_test() -> None:
    result = {"verdict": "UNCERTAIN", "reason": "need evidence after this clip"}
    plan = plan_clip_refinement(result, "What happens next?")
    assert plan["action"] == "expand_next"
    decision = {"clip_ids": ["show_clip_0001"], "frame_budget": 8}
    refined = apply_refinement_to_decision(
        decision,
        "show_clip_0001",
        "/tmp",
        lambda _base, _vid, _direction, _limit: ["show_clip_0002"],
        plan,
    )
    assert "show_clip_0002" in refined["clip_ids"]
    assert refined["clip_refinement_action"]["applied"] is True


if __name__ == "__main__":
    _self_test()
    print("ClipRefiner self-test passed.")
