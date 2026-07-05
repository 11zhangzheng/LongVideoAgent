"""Lightweight video memory for LongVideoAgent evaluation experiments.

This module is intentionally standalone: evaluators do not import it unless a
caller explicitly opts in.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

__all__ = ["VideoMemory"]


class VideoMemory:
    """Small JSON-friendly memory store for clip observations and searches."""

    def __init__(self, max_prompt_clips: int = 8, max_observations_per_clip: int = 4):
        self.max_prompt_clips = max_prompt_clips
        self.max_observations_per_clip = max_observations_per_clip
        self.clip_memory: Dict[str, Dict[str, Any]] = {}
        self.search_memory: List[Dict[str, Any]] = []
        self.grounding_history: List[Dict[str, Any]] = []

    def init_clip(
        self,
        clip_id: Any,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Create or return a clip memory record."""
        key = str(clip_id)
        if key not in self.clip_memory:
            self.clip_memory[key] = {
                "clip_id": key,
                "metadata": self._json_safe(metadata or {}),
                "observations": [],
                "useful": False,
                "failed": False,
                "use_count": 0,
                "fail_count": 0,
                "tags": set(),
            }
        elif metadata:
            self.clip_memory[key]["metadata"].update(self._json_safe(metadata))
        return self.clip_memory[key]

    def update_clip_observation(
        self,
        clip_id: Any,
        observation: Any,
        source: str | None = None,
        score: float | None = None,
    ) -> Dict[str, Any]:
        """Append an observation to a clip record."""
        record = self.init_clip(clip_id)
        item: Dict[str, Any] = {"observation": self._json_safe(observation)}
        if source is not None:
            item["source"] = source
        if score is not None:
            item["score"] = score
        record["observations"].append(item)
        return record

    def add_search(
        self,
        query: str,
        result: Any,
        clip_ids: List[Any] | None = None,
        source: str | None = None,
    ) -> Dict[str, Any]:
        """Record a search/query result without changing evaluator behavior."""
        item: Dict[str, Any] = {
            "query": query,
            "result": self._json_safe(result),
            "clip_ids": [str(clip_id) for clip_id in (clip_ids or [])],
        }
        if source is not None:
            item["source"] = source
        self.search_memory.append(item)
        return item

    def add_grounding(
        self,
        query: str,
        source_clip: Any,
        result: Any,
        predicted_clip: Any | None = None,
    ) -> Dict[str, Any]:
        """Record a grounding request/result pair."""
        item: Dict[str, Any] = {
            "query": query,
            "source_clip": str(source_clip),
            "result": self._json_safe(result),
        }
        if predicted_clip is not None:
            item["predicted_clip"] = str(predicted_clip)
        self.grounding_history.append(item)
        return item

    def mark_useful(self, clip_id: Any, tag: str | None = None) -> Dict[str, Any]:
        """Mark a clip as useful for answering or grounding."""
        record = self.init_clip(clip_id)
        record["useful"] = True
        record["failed"] = False
        record["use_count"] += 1
        if tag:
            record["tags"].add(tag)
        return record

    def mark_failed(self, clip_id: Any, reason: str | None = None) -> Dict[str, Any]:
        """Mark a clip lookup as failed while preserving observations."""
        record = self.init_clip(clip_id)
        record["failed"] = True
        record["fail_count"] += 1
        if reason:
            record.setdefault("failure_reasons", []).append(reason)
        return record

    def to_prompt_context(self, max_items: int | None = None) -> str:
        """Render compact memory text suitable for a model prompt."""
        max_items = max(1, max_items or self.max_prompt_clips)
        lines: List[str] = []

        visited_clips = list(self.clip_memory.keys())
        useful_clips = [
            clip_id for clip_id, record in self.clip_memory.items() if record.get("useful")
        ]
        failed_clips = [
            clip_id for clip_id, record in self.clip_memory.items() if record.get("failed")
        ]

        if visited_clips and len(lines) < max_items:
            lines.append(f"- visited clips: {', '.join(visited_clips[-max_items:])}")
        if useful_clips and len(lines) < max_items:
            lines.append(f"- useful clips: {', '.join(useful_clips[-max_items:])}")
        if failed_clips and len(lines) < max_items:
            lines.append(f"- failed clips: {', '.join(failed_clips[-max_items:])}")

        for clip_id, record in reversed(list(self.clip_memory.items())):
            for item in reversed(record.get("observations", [])):
                if len(lines) >= max_items:
                    break
                source = item.get("source")
                observation = item.get("observation", {})
                if not isinstance(observation, dict):
                    continue
                if source == "verifier":
                    feedback = self._truncate_text(observation.get("verifier_feedback", ""))
                    candidate = self._truncate_text(observation.get("candidate_answer", ""), 120)
                    text = f"- verifier feedback: clip {clip_id}"
                    if candidate:
                        text += f"; candidate {candidate}"
                    if feedback:
                        text += f"; feedback {feedback}"
                    lines.append(text)
                    continue
                if source != "visual_query":
                    continue
                frame_text = self._format_selected_frames(observation.get("selected_frames"))
                response = self._truncate_text(observation.get("vision_response", ""))
                feedback = self._truncate_text(observation.get("verifier_feedback", ""))
                text = f"- recent visual observation: clip {clip_id}"
                if frame_text:
                    text += f"; frames {frame_text}"
                if response:
                    text += f"; observation {response}"
                if feedback:
                    text += f"; verifier feedback {feedback}"
                lines.append(text)
            if len(lines) >= max_items:
                break

        if not lines:
            return ""
        return "Current Video Memory:\n" + "\n".join(lines)

    @staticmethod
    def _truncate_text(value: Any, max_chars: int = 500) -> str:
        if value is None:
            return ""
        text = str(value).replace("\n", " ").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @staticmethod
    def _format_selected_frames(value: Any) -> str:
        if not isinstance(value, list):
            return ""
        if len(value) <= 12:
            return json.dumps(value)
        return json.dumps(value[:12])[:-1] + ", ...]"

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the memory."""
        return self._json_safe(
            {
                "clip_memory": self.clip_memory,
                "search_memory": self.search_memory,
                "grounding_history": self.grounding_history,
                "max_prompt_clips": self.max_prompt_clips,
                "max_observations_per_clip": self.max_observations_per_clip,
            }
        )

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, set):
            return [cls._json_safe(v) for v in sorted(value, key=str)]
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


def _self_test() -> None:
    memory = VideoMemory(max_prompt_clips=2)
    memory.init_clip("s01e01_0001", {"span": (0, 4)})
    memory.update_clip_observation(
        "s01e01_0001",
        {"event": "She enters the room", "people": {"Penny"}},
    )
    memory.add_search("who enters the room?", "matched entrance scene", ["s01e01_0001"])
    memory.add_grounding(
        "find the entrance",
        "s01e01_0000",
        {"predicted_clip": "s01e01_0001"},
        "s01e01_0001",
    )
    memory.update_clip_observation(
        "s01e01_0001",
        {"candidate_answer": "a0", "verifier_feedback": {"verdict": "UNCERTAIN"}},
        source="verifier",
    )
    memory.mark_useful("s01e01_0001", tag="answer")
    memory.mark_failed("s01e01_0002", reason="no frames found")

    payload = memory.to_json()
    json.dumps(payload, ensure_ascii=False)
    assert payload["clip_memory"]["s01e01_0001"]["metadata"]["span"] == [0, 4]
    assert payload["clip_memory"]["s01e01_0001"]["observations"][0]["observation"]["people"] == [
        "Penny"
    ]
    assert payload["grounding_history"][0]["predicted_clip"] == "s01e01_0001"
    assert payload["clip_memory"]["s01e01_0001"]["tags"] == ["answer"]
    prompt_context = memory.to_prompt_context(max_items=8)
    assert "Current Video Memory:" in prompt_context
    assert "verifier feedback" in prompt_context


if __name__ == "__main__":
    _self_test()
    print("VideoMemory self-test passed.")
