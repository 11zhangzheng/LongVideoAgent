"""VerifierAgent for evaluation-time answer checking."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List


VERDICTS = {"PASS", "FAIL", "UNCERTAIN"}


def use_verifier() -> bool:
    return _env_flag("USE_VERIFIER")


def verifier_max_rounds(default: int = 1) -> int:
    try:
        return max(0, int(os.getenv("VERIFIER_MAX_ROUNDS", str(default))))
    except ValueError:
        return default


def format_options(question_data: Dict[str, Any]) -> Dict[str, str]:
    return {f"a{i}": str(question_data.get(f"a{i}", "")) for i in range(5)}


def format_recent_evidence(turns: List[Dict[str, Any]], max_items: int = 4) -> str:
    evidence: List[str] = []
    for turn in reversed(turns):
        if turn.get("action_type") not in ("search", "request_grounding"):
            continue
        content = _truncate(str(turn.get("result_content", "")), 900)
        if content:
            evidence.append(f"- turn {turn.get('turn')}: {content}")
        if len(evidence) >= max_items:
            break
    return "\n".join(reversed(evidence))


def build_verifier_feedback(result: Dict[str, Any]) -> str:
    payload = {
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "missing_evidence": result["missing_evidence"],
        "contradictions": result["contradictions"],
        "suggested_action": result["suggested_action"],
        "reason": result["reason"],
    }
    return (
        "\n<verifier_feedback>\n"
        "The proposed answer was not accepted by the verifier. Use this feedback "
        "as evidence guidance, then continue with exactly one allowed action.\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "</verifier_feedback>\n"
    )


class VerifierAgent:
    """Small verifier wrapper that asks an LLM for a strict JSON verdict."""

    def verify(
        self,
        generate_fn: Callable[[str], str],
        question: str,
        options: Dict[str, str],
        candidate_answer: str,
        memory_context: str,
        recent_evidence: str,
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(question, options, candidate_answer, memory_context, recent_evidence)
        try:
            raw_response = generate_fn(prompt)
        except Exception as exc:
            return normalize_verifier_result(
                {
                    "verdict": "PASS",
                    "confidence": 0.0,
                    "reason": f"Verifier call failed; fallback to original answer. {exc}",
                }
            )

        if raw_response.startswith("Error:"):
            result = normalize_verifier_result(
                {
                    "verdict": "PASS",
                    "confidence": 0.0,
                    "reason": "Verifier call returned an error; fallback to original answer.",
                }
            )
            result["raw_response"] = raw_response
            return result

        result = parse_verifier_json(raw_response)
        result["raw_response"] = raw_response
        return result

    @staticmethod
    def build_prompt(
        question: str,
        options: Dict[str, str],
        candidate_answer: str,
        memory_context: str,
        recent_evidence: str,
    ) -> str:
        return f"""You are a verifier for a long-video QA agent.
Use only the provided question, options, Current Video Memory, and recent evidence.
Do not use outside knowledge. If evidence is insufficient, return UNCERTAIN.

Question:
{question}

Options:
{json.dumps(options, ensure_ascii=False)}

Candidate answer:
{candidate_answer}

Current Video Memory:
{memory_context or "(empty)"}

Recent evidence:
{recent_evidence or "(empty)"}

Return exactly one JSON object and no markdown. The JSON object must have:
{{
  "verdict": "PASS" | "FAIL" | "UNCERTAIN",
  "confidence": 0.0,
  "missing_evidence": [],
  "contradictions": [],
  "suggested_action": null,
  "reason": ""
}}
"""


def parse_verifier_json(text: str) -> Dict[str, Any]:
    try:
        return normalize_verifier_result(json.loads(text))
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            return normalize_verifier_result(parsed)
        except Exception:
            continue

    return normalize_verifier_result(
        {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "missing_evidence": ["verifier_json_parse_failed"],
            "contradictions": [],
            "suggested_action": None,
            "reason": "Verifier did not return parseable JSON.",
        }
    )


def normalize_verifier_result(value: Any) -> Dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    verdict = str(data.get("verdict", "UNCERTAIN")).upper()
    if verdict not in VERDICTS:
        verdict = "UNCERTAIN"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    suggested_action = data.get("suggested_action")
    if suggested_action is not None and not isinstance(suggested_action, dict):
        suggested_action = None

    return {
        "verdict": verdict,
        "confidence": confidence,
        "missing_evidence": _as_list(data.get("missing_evidence", [])),
        "contradictions": _as_list(data.get("contradictions", [])),
        "suggested_action": suggested_action,
        "reason": str(data.get("reason", "")),
    }


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _self_test() -> None:
    result = parse_verifier_json('prefix {"verdict":"PASS","confidence":0.8,"reason":"ok"}')
    assert result["verdict"] == "PASS"
    failed = parse_verifier_json("not json")
    assert failed["verdict"] == "UNCERTAIN"
    assert "verifier_json_parse_failed" in failed["missing_evidence"]


if __name__ == "__main__":
    _self_test()
    print("VerifierAgent self-test passed.")
