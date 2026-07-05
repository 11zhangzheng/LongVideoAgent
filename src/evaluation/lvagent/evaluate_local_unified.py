"""Unified local evaluator for TVQA and TVQA+.

This script keeps one evaluation pipeline and switches dataset-specific
input handling via `--dataset`, while using a local vLLM model as the
master agent.
"""

import argparse
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from PIL import Image
from vllm import LLM, SamplingParams

try:
    from clip_refiner import (
        apply_refinement_to_decision,
        plan_clip_refinement,
        use_clip_refiner,
    )
    from memory import VideoMemory
    from verifier import (
        VerifierAgent,
        build_verifier_feedback,
        format_options,
        format_recent_evidence,
        use_verifier,
        verifier_max_rounds,
    )
except ImportError:
    from .clip_refiner import (
        apply_refinement_to_decision,
        plan_clip_refinement,
        use_clip_refiner,
    )
    from .memory import VideoMemory
    from .verifier import (
        VerifierAgent,
        build_verifier_feedback,
        format_options,
        format_recent_evidence,
        use_verifier,
        verifier_max_rounds,
    )

grounding_client = None
vision_client = None
main_llm = None
sampling_params = None
_bbox_json_cache: Dict[str, Any] = {}


@dataclass
class EvalConfig:
    dataset: str
    questions_path: str
    subs_path: str
    base_frame_dir: str
    bbox_json_path: str | None
    output_filename: str
    detailed_output_filename: str
    llm_path: str
    gpu_memory_utilization: float
    grounding_model: str
    vision_model: str
    grounding_base_url: str
    vision_base_url: str
    grounding_api_key: str | None
    vision_api_key: str | None
    grounding_cache_json_path: str
    checkpoint_path: str
    resume: bool
    verbose: bool
    debug: bool


config: EvalConfig | None = None


def use_video_memory() -> bool:
    value = os.getenv("USE_VIDEO_MEMORY", "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


DATASET_DEFAULTS: Dict[str, Dict[str, str | None]] = {
    "tvqa": {
        "questions_path": "../Tvqa/house_met/tvqa_val_house_met.jsonl",
        "subs_path": "../Tvqa/tvqa_subtitles.json",
        "base_frame_dir": "../Tvqa/house_met_frames",
        "bbox_json_path": None,
        "output_filename": "./eval_tvqa_grok-4-fast-reasoning.json",
        "detailed_output_filename": "./data_log_eval_tvqa_grok-4-fast-reasoning.json",
    },
    "tvqa_plus": {
        "questions_path": "../Tvqa_data/tvqa_plus_val.json",
        "subs_path": "../Tvqa_data/all_episodes_subtitles_by_clips.json",
        "base_frame_dir": "../Tvqa_data/bbt_frames",
        "bbox_json_path": "../Tvqa_data/clip_bbox_mapping.json",
        "output_filename": "./eval_tvqa_plus_grok-4-fast-reasoning.json",
        "detailed_output_filename": "./data_log_eval_tvqa_plus_grok-4-fast-reasoning.json",
    },
}


DEFAULT_GROUNDING_MODEL = "grok-4-fast-reasoning"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_GROUNDING_BASE_URL = "https://api2.aigcbest.top/v1"
DEFAULT_VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_GROUNDING_CACHE_JSON = "/home/rliuay/runtao/proj_videoqa/Tvqa_data/grounding_pairs_6000_samples.json"


def vision_policy_signature() -> str:
    return "|".join(
        [
            os.getenv("VISION_POLICY", "dynamic").lower(),
            os.getenv("VISION_MIN_FRAMES", "4"),
            os.getenv("VISION_MAX_FRAMES", "20"),
            os.getenv("VISION_MAX_CLIPS", "2"),
            os.getenv("VISION_MAX_CALLS", "auto"),
        ]
    )


def load_checkpoint_records(path: str, resume: bool) -> List[Dict[str, Any]]:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        checkpoint.write_text("", encoding="utf-8")
        return []
    if not checkpoint.exists():
        return []

    records: List[Dict[str, Any]] = []
    valid_bytes = 0
    with checkpoint.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_end = f.tell()
            if not line.strip():
                valid_bytes = line_end
                continue
            try:
                item = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            expected_idx = len(records) + 1
            if item.get("idx") != expected_idx or not isinstance(item.get("record"), dict):
                break
            records.append(item["record"])
            valid_bytes = line_end

    if checkpoint.stat().st_size != valid_bytes:
        with checkpoint.open("r+b") as f:
            f.truncate(valid_bytes)
        log_warn(f"Trimmed incomplete checkpoint tail: {checkpoint}")
    return records


def append_checkpoint_record(path: str, idx: int, record: Dict[str, Any]) -> None:
    payload = json.dumps({"idx": idx, "record": record}, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(payload + "\n")
        f.flush()
        os.fsync(f.fileno())


def log_info(msg: str) -> None:
    if config is None or config.verbose:
        print(msg)


def log_debug(msg: str) -> None:
    if config is not None and config.debug:
        print(msg)


def log_warn(msg: str) -> None:
    print(msg)


def make_progress_bar(total: int, desc: str):
    if config is not None and not config.verbose:
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="q", dynamic_ncols=True, leave=True)
    except Exception:
        class _SimpleProgress:
            def __init__(self, total_count: int, title: str):
                self.total = max(total_count, 1)
                self.n = 0
                self.title = title

            def update(self, n: int = 1):
                self.n += n
                pct = min(100.0, self.n * 100.0 / self.total)
                print(f"\r{self.title}: {self.n}/{self.total} ({pct:.1f}%)", end="", flush=True)

            def close(self):
                print()

        return _SimpleProgress(total, desc)


def initialize_clients() -> None:
    global grounding_client, vision_client, config
    if config is None:
        raise RuntimeError("Config not initialized.")

    grounding_client = OpenAI(
        api_key=config.grounding_api_key,
        base_url=config.grounding_base_url,
    )
    vision_client = OpenAI(
        api_key=config.vision_api_key,
        base_url=config.vision_base_url,
    )


def initialize_main_model() -> None:
    global main_llm, sampling_params, config
    if config is None:
        raise RuntimeError("Config not initialized.")

    log_info(f"  -> Loading local main model from: {config.llm_path}")
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "60000"))
    dtype = os.getenv("VLLM_DTYPE", "half")
    main_llm = LLM(
        model=config.llm_path,
        tokenizer=config.llm_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=4096,
        skip_special_tokens=True,
    )
    log_info("Main local vLLM initialized.")


def _extract_episode_prefix_for_subtitles(vid_name: str) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset == "tvqa":
        if not vid_name:
            return ""
        parts = vid_name.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return parts[0]
    return vid_name.split("_")[0] if vid_name else ""


def _extract_episode_prefix_for_grounding(vid_name: str) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset == "tvqa":
        return vid_name
    return vid_name[:6]


def _initial_answer_format_hint() -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset == "tvqa_plus":
        return (
            "The answer must be concise and direct, in the format <answer>ax</answer>, "
            "where 'x' is the index of the selected option (e.g., <answer>a1</answer> for option a1)."
        )
    return "The answer must be concise and direct."


def _load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Questions file must contain a JSON list: {path}")
            return [normalize_question_entry(q) for q in data]
        except json.JSONDecodeError:
            pass

    questions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(normalize_question_entry(json.loads(line)))
    if not questions:
        raise ValueError(f"Questions file is empty or invalid: {path}")
    return questions


def parse_answer_to_idx(answer_value: Any) -> int:
    if isinstance(answer_value, int):
        if 0 <= answer_value <= 4:
            return answer_value
        raise ValueError(f"Invalid integer answer index: {answer_value}")
    if isinstance(answer_value, str):
        s = answer_value.strip().lower()
        m = re.fullmatch(r"a([0-4])", s)
        if m:
            return int(m.group(1))
        if s.isdigit():
            idx = int(s)
            if 0 <= idx <= 4:
                return idx
    raise ValueError(f"Unsupported answer format: {answer_value}")


def normalize_question_entry(raw_q: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(raw_q)
    if "vid_name" not in q:
        q["vid_name"] = q.get("occur_clip", "")
    if not q["vid_name"]:
        raise ValueError("Missing clip id: expected `vid_name` or `occur_clip`.")
    if "answer_idx" not in q and "answer" in q:
        q["answer_idx"] = parse_answer_to_idx(q["answer"])
    return q


def load_subtitles_file(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            raise ValueError(f"Subtitles file must contain a JSON object: {path}")
        except json.JSONDecodeError:
            pass

    subtitles: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                if len(item) == 1:
                    key, val = next(iter(item.items()))
                    subtitles[str(key)] = str(val)
                elif "clip" in item and "subtitle" in item:
                    subtitles[str(item["clip"])] = str(item["subtitle"])
    if not subtitles:
        raise ValueError(f"Subtitles file is empty or invalid: {path}")
    return subtitles


def bbox_to_string_simplified(
    file_path: str | None = None,
    key: str | None = None,
    frame_ids: List[int] | None = None,
) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset != "tvqa_plus":
        return ""

    file_path = file_path or config.bbox_json_path
    if not file_path:
        return ""

    try:
        if file_path not in _bbox_json_cache:
            with open(file_path, "r", encoding="utf-8") as f:
                _bbox_json_cache[file_path] = json.load(f)
        json_data = _bbox_json_cache[file_path]

        if key not in json_data:
            return f"Error: Key '{key}' not found in JSON data."

        bbox_data = json_data[key]
        result: List[str] = []
        sorted_frame_ids = sorted(bbox_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
        selected_ids = {str(x) for x in frame_ids} if frame_ids else None
        for frame_id in sorted_frame_ids:
            if selected_ids is not None and str(frame_id) not in selected_ids:
                continue
            result.append(f"Frame {frame_id}:")
            for bbox in bbox_data[frame_id]:
                x, y, width, height, name = bbox
                result.append(f"  - {name}: ({x}, {y}, {width}, {height})")
        return "\n".join(result) if result else "No bounding boxes for selected frames."
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in file '{file_path}'."
    except Exception as e:
        return f"Error processing JSON data: {str(e)}"


def convert_image_to_base64_data_url(path: str) -> str | None:
    try:
        with Image.open(path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

# 添加问题分类器
def classify_vision_question(question: str, search_query: str) -> str:
    text = f"{question} {search_query}".lower()
    if re.search(r"\b(before|after|then|next|first|last|while|during|starts?|ends?|finished?)\b", text):
        return "temporal_action"
    if re.search(
        r"\b(wear(?:ing)?|hold(?:ing)?|carry(?:ing)?|color|look(?:ing)? like|"
        r"where|visible|object|gesture|doing|do after|do before)\b",
        text,
    ):
        return "visual_detail"
    if re.search(r"\b(why|feel|feeling|emotion|reaction|react|upset|angry|happy|sad|surprised)\b", text):
        return "causal_emotion"
    if re.search(r"\b(say|said|tell|told|ask|asked|mention|talk|reply|answer|conversation)\b", text):
        return "dialogue"
    return "general"


def subtitle_evidence_score(question: str, subtitle: str) -> float:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "did", "does", "do",
        "what", "why", "how", "when", "where", "who", "after", "before", "then",
        "he", "she", "they", "him", "her", "his", "their", "to", "of", "in",
        "on", "at", "and", "or", "that", "this", "it",
    }
    question_tokens = {
        token for token in re.findall(r"[a-z0-9']+", question.lower())
        if len(token) > 2 and token not in stopwords
    }
    if not question_tokens:
        return 0.0
    subtitle_tokens = set(re.findall(r"[a-z0-9']+", subtitle.lower()))
    return len(question_tokens & subtitle_tokens) / len(question_tokens)


def temporal_direction(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(before|previous|earlier|first|starts?)\b", lowered):
        return "previous"
    if re.search(r"\b(after|next|later|then|ends?|finished?)\b", lowered):
        return "next"
    return "both"


def adjacent_clip_ids(base_frame_dir: str, vid: str, direction: str, limit: int) -> List[str]:
    match = re.fullmatch(r"(.*_clip_)(\d+)", vid)
    if not match or limit <= 0:
        return []
    prefix, raw_idx = match.groups()
    current_idx = int(raw_idx)
    width = len(raw_idx)
    offsets = {
        "previous": [-1, -2],
        "next": [1, 2],
        "both": [-1, 1, -2, 2],
    }.get(direction, [-1, 1])
    result: List[str] = []
    for offset in offsets:
        candidate = f"{prefix}{current_idx + offset:0{width}d}"
        if current_idx + offset >= 0 and Path(base_frame_dir, candidate).is_dir():
            result.append(candidate)
        if len(result) >= limit:
            break
    return result


def available_frame_numbers(base_frame_dir: str, vid: str) -> List[int]:
    clip_dir = Path(base_frame_dir, vid)
    if not clip_dir.is_dir():
        return []
    return sorted(
        int(path.stem)
        for path in clip_dir.glob("*.jpg")
        if path.stem.isdigit()
    )


def sample_frame_numbers(frames: List[int], count: int, direction: str) -> List[int]:
    if not frames or count <= 0:
        return []
    if direction == "previous":
        frames = frames[: max(count, (len(frames) * 2 + 2) // 3)]
    elif direction == "next":
        frames = frames[min(len(frames) // 3, max(len(frames) - count, 0)) :]
    if len(frames) <= count:
        return frames
    if count == 1:
        return [frames[len(frames) // 2]]
    positions = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    return [frames[pos] for pos in positions]


def build_vision_decision(
    vid: str,
    question_data: Dict[str, Any],
    search_query: str,
    clip_subtitles: Dict[str, str],
    policy_state: Dict[str, Any],
) -> Dict[str, Any]:
    if config is None:
        raise RuntimeError("Config not initialized.")

    policy = os.getenv("VISION_POLICY", "dynamic").lower()
    call_index = int(policy_state.get("vision_calls", 0))
    if policy == "fixed":
        return {
            "policy": "fixed",
            "question_type": "fixed",
            "subtitle_evidence": 0.0,
            "call_index": call_index + 1,
            "max_vision_calls": max(1, int(os.getenv("VISION_MAX_CALLS", "5"))),
            "frame_budget": 15,
            "clip_ids": [vid],
            "direction": "both",
            "reason": "Original fixed sampling baseline.",
        }

    question = str(question_data.get("q", ""))
    question_type = classify_vision_question(question, search_query)
    evidence_score = subtitle_evidence_score(question, clip_subtitles.get(vid, ""))
    settings = {
        "dialogue": (6, 1, 1),
        "visual_detail": (12, 2, 2),
        "temporal_action": (16, 2, 2),
        "causal_emotion": (10, 2, 2),
        "general": (8, 2, 1),
    }
    frame_budget, max_calls, max_clips = settings[question_type]

    if evidence_score >= 0.45:
        frame_budget = max(4, frame_budget - 4)
    elif evidence_score < 0.15:
        frame_budget += 2
    if call_index > 0:
        frame_budget += 4

    max_calls = int(os.getenv("VISION_MAX_CALLS", str(max_calls)))
    frame_budget = min(
        int(os.getenv("VISION_MAX_FRAMES", "20")),
        max(int(os.getenv("VISION_MIN_FRAMES", "4")), frame_budget),
    )
    max_clips = min(max_clips, int(os.getenv("VISION_MAX_CLIPS", "2")))
    direction = temporal_direction(f"{question} {search_query}")
    clip_ids = [vid]
    if call_index > 0 and max_clips > 1:
        clip_ids.extend(adjacent_clip_ids(config.base_frame_dir, vid, direction, max_clips - 1))

    return {
        "policy": "dynamic",
        "question_type": question_type,
        "subtitle_evidence": round(evidence_score, 4),
        "call_index": call_index + 1,
        "max_vision_calls": max_calls,
        "frame_budget": frame_budget,
        "clip_ids": clip_ids,
        "direction": direction,
        "reason": (
            f"type={question_type}, subtitle_evidence={evidence_score:.2f}, "
            f"previous_vision_calls={call_index}"
        ),
    }


def process_and_query_seg(
    seg: dict,
    vid: str,
    text_client: OpenAI,
    vision_client_instance: OpenAI,
    question_data: Dict[str, Any],
    clip_subtitles: Dict[str, str],
    policy_state: Dict[str, Any],
    base_frame_dir: str | None = None,
    model: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    if config is None:
        raise RuntimeError("Config not initialized.")
    base_frame_dir = base_frame_dir or config.base_frame_dir
    model = model or config.vision_model

    decision = build_vision_decision(vid, question_data, seg["description"], clip_subtitles, policy_state)
    refinement_action = policy_state.pop("pending_clip_refinement_action", None)
    if use_clip_refiner() and refinement_action:
        decision = apply_refinement_to_decision(
            decision,
            vid,
            base_frame_dir,
            adjacent_clip_ids,
            refinement_action,
        )
    if policy_state.get("vision_calls", 0) >= decision["max_vision_calls"]:
        decision["budget_exhausted"] = True
        return (
            "Vision budget exhausted. Use the available subtitle and visual evidence, "
            "request grounding if the clip is wrong, or provide the final answer.",
            decision,
        )

    policy_state["vision_calls"] = policy_state.get("vision_calls", 0) + 1
    clip_ids = decision["clip_ids"]
    per_clip_budget = max(2, decision["frame_budget"] // max(len(clip_ids), 1))
    messages_content = []
    selected_frames: Dict[str, List[int]] = {}
    total_images = 0
    for clip_id in clip_ids:
        available = available_frame_numbers(base_frame_dir, clip_id)
        if decision["policy"] == "fixed":
            available_set = set(available)
            frame_nums = [frame for frame in range(1, 181, 12) if frame in available_set]
        else:
            frame_nums = sample_frame_numbers(available, per_clip_budget, decision["direction"])
        selected_frames[clip_id] = frame_nums
        messages_content.append(
            {
                "type": "text",
                "text": f"Clip {clip_id}; sampled frame ids: {frame_nums}.",
            }
        )
        for fn in frame_nums:
            img_path = Path(base_frame_dir, clip_id, f"{fn:05d}.jpg")
            url = convert_image_to_base64_data_url(str(img_path))
            if url:
                messages_content.append({"type": "image_url", "image_url": {"url": url}})
                total_images += 1

    decision["selected_frames"] = selected_frames
    decision["actual_frame_count"] = total_images
    if total_images == 0:
        raise FileNotFoundError(f"No readable frames found for clips: {clip_ids}")

    if config.dataset == "tvqa_plus":
        bbox_sections = [
            f"Clip {clip_id}:\n{bbox_to_string_simplified(key=clip_id, frame_ids=selected_frames[clip_id])}"
            for clip_id in clip_ids
        ]
        bbox_info = "\n".join(bbox_sections)
        prompt = (
            f"The images above contain {total_images} sampled frames from {len(clip_ids)} clip(s). "
            f"Bounding box information for the sampled frames is:\n{bbox_info}\n"
            "You can focus on the key objects and actions within these bounding boxes in the frames.\n"
            "And here is a description of what I want to know:\n"
            f"{seg['description']}"
        )
    else:
        prompt = (
            f"The images above contain {total_images} sampled frames from {len(clip_ids)} clip(s).\n"
            "And here is a description of what I want to know:\n"
            f"{seg['description']}"
        )

    messages_content.append({"type": "text", "text": prompt})
    resp = vision_client_instance.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages_content}],
    )
    return resp.choices[0].message.content, decision


def parse_action_from_response(response: str) -> Tuple[str, str]:
    search_match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
    if search_match:
        return "search", search_match.group(1).strip()

    grounding_match = re.search(r"<request_grounding>(.*?)</request_grounding>", response, re.DOTALL)
    if grounding_match:
        return "request_grounding", grounding_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    return "invalid", ""


def execute_action(
    action_type: str,
    content: str,
    vid: str,
    text_client: OpenAI,
    vision_client_instance: OpenAI,
    question_data: Dict,
    episode_sub_block: str,
    clip_subtitles: Dict[str, str],
    policy_state: Dict[str, Any],
    memory: VideoMemory | None = None,
) -> Tuple[str, bool, Dict[str, Any] | None]:
    if action_type == "answer":
        return f"\n<answer>{content}</answer>", True, None

    if action_type == "search":
        try:
            seg = {"description": content}
            information_parts: List[str] = []
            vision_decision: Dict[str, Any] | None = None
            try:
                vision_response, vision_decision = process_and_query_seg(
                    seg,
                    vid,
                    text_client,
                    vision_client_instance,
                    question_data,
                    clip_subtitles,
                    policy_state,
                )
                information_parts.append(f"Visual Description:\n{vision_response.strip()}")
                if memory is not None:
                    clip_ids = vision_decision.get("clip_ids", [vid]) if vision_decision else [vid]
                    selected_frames = (vision_decision or {}).get("selected_frames", {})
                    for clip_id in clip_ids:
                        memory.init_clip(clip_id)
                        memory.update_clip_observation(
                            clip_id,
                            {
                                "search_query": content,
                                "selected_frames": selected_frames.get(clip_id, []),
                                "vision_response": vision_response,
                                "vision_decision": vision_decision,
                            },
                            source="visual_query",
                        )
                    memory.add_search(
                        content,
                        {
                            "observed_clip": vid,
                            "selected_frames": selected_frames,
                            "vision_response": vision_response,
                            "vision_decision": vision_decision,
                        },
                        clip_ids,
                        source="visual_query",
                    )
            except Exception as e:
                log_warn(f"Vision LLM call failed: {e}")
                information_parts.append(f"Visual Description: Error - {str(e)}")
                if memory is not None:
                    memory.mark_failed(vid, reason=f"vision_query_failed: {e}")

            combined_info = "\n".join(information_parts)
            if config and config.dataset == "tvqa_plus":
                selected_frames = (vision_decision or {}).get("selected_frames", {})
                bbox_sections = [
                    f"Clip {clip_id}:\n{bbox_to_string_simplified(key=clip_id, frame_ids=frame_ids)}"
                    for clip_id, frame_ids in selected_frames.items()
                ]
                bbox_info = "\n".join(bbox_sections)
                return (
                    f"\n<information>Bounding Box:{bbox_info}\n{combined_info}</information>\n",
                    False,
                    vision_decision,
                )
            return f"\n<information>\n{combined_info}</information>\n", False, vision_decision
        except Exception as e:
            log_warn(f"Search action failed: {e}")
            return (
                f"\n<information>Error: Failed to get information - {str(e)}</information>\n",
                False,
                None,
            )

    if action_type == "request_grounding":
        try:
            grounding_result = re_analyze_single_question_api(question_data, episode_sub_block, vid, attempt_round=1)
            if "error" in grounding_result:
                log_warn(f"Grounding failed: {grounding_result['error']}")
                result_content = f"Grounding failed for query: {content}. Error: {grounding_result['error']}"
                if memory is not None:
                    memory.add_grounding(content, vid, grounding_result)
                    memory.mark_failed(vid, reason=f"grounding_failed: {grounding_result['error']}")
                return f"\n<grounding_info>{result_content}</grounding_info>\n", False, None
            predicted_clip = grounding_result.get("predicted_clip", vid)
            new_sub = get_clip_subtitle(clip_subtitles, predicted_clip)
            if memory is not None:
                memory.add_grounding(content, vid, grounding_result, predicted_clip)
                memory.update_clip_observation(
                    predicted_clip,
                    {"subtitle": new_sub, "grounding_query": content},
                    source="request_grounding",
                )
                memory.mark_useful(predicted_clip, tag="grounding")
            result_content = f"<New_clip>{predicted_clip} + {new_sub}</New_clip>"
            return f"\n{result_content}\n", False, None
        except Exception as e:
            log_warn(f"Grounding action failed: {e}")
            if memory is not None:
                memory.add_grounding(content, vid, {"error": str(e)})
                memory.mark_failed(vid, reason=f"grounding_exception: {e}")
            return (
                f"\n<grounding_info>Error: Failed to perform grounding - {str(e)}</grounding_info>\n",
                False,
                None,
            )

    return "\nMy action is not correct. I need to search, request grounding, or answer.\n", False, None


def get_clip_subtitle(clip_subtitles: Dict[str, str], clip_name: str) -> str:
    subtitle_text = clip_subtitles.get(clip_name, "")
    if subtitle_text:
        return f"<{clip_name}>{subtitle_text}</{clip_name}>"
    log_warn(f"Warning: No subtitle found for clip {clip_name}")
    return ""


def build_subtitles_for_episode(clip_subtitles: Dict[str, str], episode_prefix: str) -> str:
    matching_clips = {k: v for k, v in clip_subtitles.items() if k.startswith(episode_prefix)}
    sorted_clips = sorted(matching_clips.items())
    formatted_subtitles = [f"<{clip_key}>{subtitle_text}</{clip_key}>" for clip_key, subtitle_text in sorted_clips]
    return "\n".join(formatted_subtitles)


def get_subtitles_for_video(clip_subtitles: Dict[str, str], vid_name: str) -> str:
    episode_prefix = _extract_episode_prefix_for_subtitles(vid_name)
    return build_subtitles_for_episode(clip_subtitles, episode_prefix)


def grounding_llm_generate(user_content: str, model: str | None = None) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    model = model or config.grounding_model
    try:
        response = grounding_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.6,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        log_warn(f"Grounding API call failed: {e}")
        return "Error: Failed to generate response"


def main_llm_generate(conversation_history: str) -> str:
    if main_llm is None or sampling_params is None:
        raise RuntimeError("Main local vLLM is not initialized.")
    try:
        outputs = main_llm.generate([conversation_history], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        log_warn(f"Main local vLLM inference failed: {e}")
        return "Error: Failed to generate response"


def postprocess_response(response: str) -> str:
    type_match = re.search(r"<type>.*?</type>", response, re.DOTALL)
    if type_match:
        return response[: type_match.end()]

    time_match = re.search(r"<time>.*?</time>", response, re.DOTALL)
    if time_match:
        return response[: time_match.end()]
    return response


def extract_clip_content(text: str) -> str:
    match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def re_analyze_single_question_api(question_data: Dict, sub_block: str, vid: str, attempt_round: int) -> Dict:
    original_idx = question_data.get("original_idx", "?")
    log_debug(f"      Question {original_idx}: API Analysis round {attempt_round}")

    prompt_content = f"""
Question: {question_data['q']}
Options:
a0: {question_data.get('a0', '')}
a1: {question_data.get('a1', '')}
a2: {question_data.get('a2', '')}
a3: {question_data.get('a3', '')}
a4: {question_data.get('a4', '')}

Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>, where each < > pair contains a clip label followed by its corresponding subtitle content.

Based on the question and subtitles, determine:
1. The specific clip label where the answer to this question occurs or is mentioned (output in <clip>label</clip> format)
{vid} may not contain the scene or context related to the question. Please determine a different specific clip label.
Please analyze the given question and provide the following information:
<clip>
clip_label (the specific clip where the question's answer can be found in the video)
</clip>
"""
    try:
        raw_response = grounding_llm_generate(prompt_content)
        log_debug(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        log_debug(f"        Predicted clip: {predicted_clip}")
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response,
        }
    except Exception as e:
        log_warn(f"        Analysis error: {e}")
        return {"error": str(e)}


def analyze_single_question_api(
    question_data: Dict,
    sub_block: str,
    attempt_round: int,
    json_path: str | None = None,
) -> Dict:
    if config is None:
        raise RuntimeError("Config not initialized.")
    json_path = json_path or config.grounding_cache_json_path

    original_idx = question_data.get("original_idx", "?")
    log_debug(f"      Question {original_idx}: API Analysis round {attempt_round}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            grounding_pairs = json.load(f)
        target_question = question_data["q"].strip()
        for _, entry in grounding_pairs.items():
            if "question" in entry and entry["question"].strip() == target_question:
                predicted_clip = entry.get("clip", "")
                log_debug(f"        Found match in JSON: Predicted clip: {predicted_clip}")
                return {
                    "predicted_clip": predicted_clip,
                    "raw_response": "From JSON cache",
                    "processed_response": "From JSON cache",
                }
        log_debug("        No match found in JSON, falling back to API...")
    except Exception as e:
        log_debug(f"        JSON loading error: {e}, falling back to API...")

    prompt_content = f"""
Question: {question_data['q']}
Options:
a0: {question_data.get('a0', '')}
a1: {question_data.get('a1', '')}
a2: {question_data.get('a2', '')}
a3: {question_data.get('a3', '')}
a4: {question_data.get('a4', '')}

Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>, where each < > pair contains a clip label followed by its corresponding subtitle content.

Based on the question and subtitles, determine:
1. The specific clip label where the answer to this question occurs or is mentioned (output in <clip>label</clip> format)
Please analyze the given question and provide the following information:
<clip>
clip_label (the specific clip where the question's answer can be found in the video)
</clip>
"""
    try:
        raw_response = grounding_llm_generate(prompt_content)
        log_debug(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        log_debug(f"        Predicted clip: {predicted_clip}")
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response,
        }
    except Exception as e:
        log_warn(f"        Analysis error: {e}")
        return {"error": str(e)}


def process_single_question(
    prompt: str,
    vid: str,
    question: str,
    question_data: Dict,
    episode_sub_block: str,
    clip_subtitles: Dict[str, str],
    max_turn: int = 5,
) -> Dict[str, Any]:
    record = {
        "vid": vid,
        "question": question,
        "turns": [],
        "final_answer": "",
        "prompt": prompt,
        "vision_policy": os.getenv("VISION_POLICY", "dynamic"),
        "vision_policy_signature": vision_policy_signature(),
    }
    verifier_enabled = use_verifier()
    memory = VideoMemory() if use_video_memory() or verifier_enabled else None
    verifier_agent = VerifierAgent() if verifier_enabled else None
    verifier_retry_count = 0
    max_verifier_retries = verifier_max_rounds()
    conversation_history = prompt
    final_answer = ""
    policy_state: Dict[str, Any] = {"vision_calls": 0}

    for turn in range(max_turn):
        log_debug(f"\n{'=' * 60}")
        log_debug(f"  Turn {turn + 1}/{max_turn}")
        log_debug(f"{'=' * 60}")

        if turn == max_turn - 1:
            conversation_history += (
                "\nThis is the final turn. Please directly perform Action C and "
                "provide the final answer in <answer>...</answer> format.\n"
            )

        llm_input = conversation_history
        memory_context = ""
        memory_context_token_estimate = 0
        if use_video_memory() and memory is not None:
            memory_context = memory.to_prompt_context(
                max_items=int(os.getenv("MEMORY_MAX_ITEMS", "8"))
            )
            memory_context_token_estimate = estimate_text_tokens(memory_context)
            if memory_context:
                llm_input = f"{conversation_history}\n\n{memory_context}\n"

        raw_response = main_llm_generate(llm_input)
        response = postprocess_response(raw_response) if "postprocess_response" in globals() else raw_response
        log_debug(f"LLM Response:\n{response}")

        action_type, content = parse_action_from_response(response)
        preview = f"{content[:100]}...{'...' if len(content) > 100 else ''}"
        log_debug(f"Parsed action - Type: {action_type}, Content: {preview}")

        turn_record = {
            "turn": turn + 1,
            "response": response,
            "action_type": action_type,
            "content": content,
            "is_done": False,
        }
        if use_video_memory() and memory is not None:
            turn_record["memory_context_token_estimate"] = memory_context_token_estimate

        log_debug(f"Executing action: {action_type}")
        result_content, is_done, vision_decision = execute_action(
            action_type,
            content,
            vid,
            text_client=grounding_client,
            vision_client_instance=vision_client,
            question_data=question_data,
            episode_sub_block=episode_sub_block,
            clip_subtitles=clip_subtitles,
            policy_state=policy_state,
            memory=memory,
        )

        turn_record["result_content"] = result_content
        turn_record["is_done"] = is_done
        if vision_decision is not None:
            turn_record["vision_decision"] = vision_decision
        log_debug(f"Action result:\n{result_content}")

        if is_done and action_type == "answer":
            if verifier_agent is not None and verifier_retry_count < max_verifier_retries:
                memory_context_for_verifier = memory.to_prompt_context() if memory is not None else ""
                recent_evidence = format_recent_evidence(record["turns"])
                verifier_result = verifier_agent.verify(
                    main_llm_generate,
                    question,
                    format_options(question_data),
                    content,
                    memory_context_for_verifier,
                    recent_evidence,
                )
                turn_record["verifier_result"] = verifier_result
                log_debug(f"Verifier result: {verifier_result}")
                if verifier_result.get("verdict") == "PASS":
                    final_answer = content
                    log_debug(f"Verifier passed answer in turn {turn + 1}: {final_answer}")
                    record["turns"].append(turn_record)
                    break

                verifier_retry_count += 1
                if use_clip_refiner():
                    refinement_action = plan_clip_refinement(
                        verifier_result,
                        question,
                        classify_vision_question(question, ""),
                    )
                    policy_state["pending_clip_refinement_action"] = refinement_action
                    turn_record["clip_refinement_action"] = refinement_action
                feedback = build_verifier_feedback(verifier_result)
                turn_record["result_content"] = feedback
                turn_record["is_done"] = False
                if memory is not None:
                    memory.update_clip_observation(
                        vid,
                        {
                            "candidate_answer": content,
                            "verifier_feedback": verifier_result,
                        },
                        source="verifier",
                    )
                conversation_history += feedback
                record["turns"].append(turn_record)
                continue

            final_answer = content
            log_debug(f"Found answer in turn {turn + 1}: {final_answer}")
            record["turns"].append(turn_record)
            break

        conversation_history += result_content
        log_debug("Updated conversation history with result content")

        if action_type == "request_grounding" and "<New_clip>" in result_content:
            match = re.search(r"<New_clip>(.*?) \+", result_content, re.DOTALL)
            if match:
                vid = match.group(1).strip()
                log_debug(f"Updated vid to: {vid}")

        record["turns"].append(turn_record)
        if turn == max_turn - 1:
            log_debug(f"  Reached maximum turns ({max_turn})")

    record["final_answer"] = final_answer
    record["conversation_history"] = conversation_history
    record["vision_calls_used"] = policy_state["vision_calls"]
    if use_video_memory() and memory is not None:
        record["memory"] = memory.to_json()
    return record


def process_question_wrapper(q: Dict, total: int, clip_subtitles: Dict[str, str], max_turn: int) -> Dict[str, Any]:
    started_at = time.perf_counter()
    try:
        q["original_idx"] = total
        log_info(f"Processing question {total}")

        episode_prefix = _extract_episode_prefix_for_grounding(q["vid_name"])
        episode_sub_block = build_subtitles_for_episode(clip_subtitles, episode_prefix)
        grounding_result = analyze_single_question_api(q, episode_sub_block, attempt_round=1)
        predicted_clip = grounding_result.get("predicted_clip", q["vid_name"])
        log_debug(f"Predicted clip for question {total}: {predicted_clip}")

        sub_block = get_clip_subtitle(clip_subtitles, predicted_clip)
        initial_prompt = f"""You must follow these rules in every turn:
Reasoning First: Start by conducting your reasoning inside <reasoning>...</reasoning>. This is where you analyze the current information and decide your next step.
Choose One Action: After reasoning, you must choose exactly one of the following three actions:

Action A: If the current information is insufficient or you are somewhat uncertain, and you need to search for visual information on the current located <clipX>, then search for visual information. If your reasoning indicates that you lack necessary visual knowledge, you can call a visual engine. To do this, use the following format: <search>query</search>
Vision calls use a limited adaptive budget. Make each search query specific, avoid repeating the same request, and use grounding instead when the current clip appears incorrect.

Action B: If the current information is insufficient or you are somewhat uncertain, and you cannot obtain the final answer from the previous location and its possible visual information, then you need to call the grounding agent again for relocation, and output in the <request_grounding> format.

Action C: Provide the Final Answer
If your reasoning indicates that you have enough information to answer, provide the final answer inside <answer>...</answer>.
{_initial_answer_format_hint()}

question: {q['q']}
a0: {q['a0']}
a1: {q['a1']}
a2: {q['a2']}
a3: {q['a3']}
a4: {q['a4']}
<information>subtitles: {sub_block}</information>
"""

        record = process_single_question(
            prompt=initial_prompt,
            vid=predicted_clip,
            question=q["q"],
            question_data=q,
            episode_sub_block=episode_sub_block,
            clip_subtitles=clip_subtitles,
            max_turn=max_turn,
        )
        record["predicted_clip"] = predicted_clip
        record["gt_answer_idx"] = q.get("answer_idx")
        record["failed"] = False
        record["error"] = ""
        record["elapsed_seconds"] = time.perf_counter() - started_at
        log_info(f"  Result: {len(record['turns'])} turns, Answer: {record['final_answer']}")
        return record
    except Exception as e:
        log_warn(f"Error processing question {total}: {e}")
        return {
            "vid": q.get("vid_name", q.get("occur_clip", "")),
            "question": q.get("q", ""),
            "turns": [],
            "final_answer": "",
            "conversation_history": "",
            "predicted_clip": "",
            "gt_answer_idx": q.get("answer_idx"),
            "failed": True,
            "error": str(e),
            "elapsed_seconds": time.perf_counter() - started_at,
            "vision_policy": os.getenv("VISION_POLICY", "dynamic"),
            "vision_policy_signature": vision_policy_signature(),
            "vision_calls_used": 0,
        }


def run_enhanced_pipeline(max_turn: int = 5) -> None:
    if config is None:
        raise RuntimeError("Config not initialized.")

    run_started_at = time.perf_counter()
    initialize_main_model()
    model_ready_at = time.perf_counter()

    clip_subtitles = load_subtitles_file(config.subs_path)
    questions = _load_questions(config.questions_path)

    total = len(questions)
    results = load_checkpoint_records(config.checkpoint_path, config.resume)
    if len(results) > total:
        raise ValueError(
            f"Checkpoint contains {len(results)} records, but questions file has only {total}: "
            f"{config.checkpoint_path}"
        )
    for idx, record in enumerate(results):
        expected_question = questions[idx].get("q", "")
        if record.get("question", "") != expected_question:
            raise ValueError(
                f"Checkpoint mismatch at question {idx + 1}. Use a different checkpoint path "
                f"or set RESUME_EVAL=0 to start over: {config.checkpoint_path}"
            )
        if record.get("vision_policy_signature") != vision_policy_signature():
            raise ValueError(
                f"Checkpoint Vision policy differs from the current configuration at question {idx + 1}. "
                f"Use a different CHECKPOINT_PATH or set RESUME_EVAL=0: {config.checkpoint_path}"
            )
    start_idx = len(results)
    if start_idx:
        log_info(f"Resuming from checkpoint: {start_idx}/{total} questions already completed")
    else:
        log_info(f"Starting a new checkpoint: {config.checkpoint_path}")

    eval_started_at = time.perf_counter()
    pbar = make_progress_bar(total, "Evaluating")
    if pbar is not None and start_idx:
        pbar.update(start_idx)
    for idx, q in enumerate(questions[start_idx:], start=start_idx + 1):
        record = process_question_wrapper(q, idx, clip_subtitles, max_turn)
        results.append(record)
        append_checkpoint_record(config.checkpoint_path, idx, record)
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    log_info(f"\nTotal processed: {total}")

    simplified_results = [
        {
            "vid": result["vid"],
            "question": result["question"],
            "num_turns": len(result["turns"]),
            "final_answer": result["final_answer"],
            "predicted_clip": result.get("predicted_clip", ""),
            "gt_answer_idx": result.get("gt_answer_idx"),
            "failed": result.get("failed", False),
            "error": result.get("error", ""),
            "elapsed_seconds": result.get("elapsed_seconds", 0.0),
            "vision_calls_used": result.get("vision_calls_used", 0),
            "vision_frames_used": sum(
                turn.get("vision_decision", {}).get("actual_frame_count", 0)
                for turn in result["turns"]
            ),
            "vision_clips_used": sum(
                len(turn.get("vision_decision", {}).get("clip_ids", []))
                for turn in result["turns"]
            ),
        }
        for result in results
    ]

    detailed_results = []
    for result in results:
        last_response = result["turns"][-1]["response"] if result["turns"] else ""
        detailed_record = {
            "vid": result["vid"],
            "question": result["question"],
            "final_answer": result.get("final_answer", ""),
            "gt_answer_idx": result.get("gt_answer_idx"),
            "conversation_history": result["conversation_history"],
            "last_llm_response": last_response,
            "turns": result["turns"],
            "vision_policy": result.get("vision_policy", ""),
            "vision_policy_signature": result.get("vision_policy_signature", ""),
            "failed": result.get("failed", False),
            "error": result.get("error", ""),
            "elapsed_seconds": result.get("elapsed_seconds", 0.0),
        }
        if use_video_memory() and "memory" in result:
            detailed_record["memory"] = result["memory"]
            detailed_record["memory_context_token_estimates"] = [
                turn.get("memory_context_token_estimate", 0) for turn in result["turns"]
            ]
        if use_verifier():
            verifier_results = [
                turn["verifier_result"] for turn in result["turns"] if "verifier_result" in turn
            ]
            detailed_record["verifier_result"] = verifier_results[-1] if verifier_results else None
            detailed_record["verifier_results"] = verifier_results
        if use_clip_refiner():
            refinement_actions = [
                turn.get("clip_refinement_action")
                or turn.get("vision_decision", {}).get("clip_refinement_action")
                for turn in result["turns"]
                if turn.get("clip_refinement_action")
                or turn.get("vision_decision", {}).get("clip_refinement_action")
            ]
            detailed_record["clip_refinement_action"] = refinement_actions[-1] if refinement_actions else None
            detailed_record["clip_refinement_actions"] = refinement_actions
        detailed_results.append(detailed_record)

    total_search_actions = sum(
        sum(1 for t in result["turns"] if t["action_type"] == "search")
        for result in results
    )
    total_vision_calls = sum(result.get("vision_calls_used", 0) for result in results)
    total_grounding_calls = sum(
        sum(1 for t in result["turns"] if t["action_type"] == "request_grounding") for result in results
    )
    vision_decisions = [
        turn["vision_decision"]
        for result in results
        for turn in result["turns"]
        if "vision_decision" in turn
    ]
    total_vision_frames = sum(decision.get("actual_frame_count", 0) for decision in vision_decisions)
    total_vision_clips = sum(len(decision.get("clip_ids", [])) for decision in vision_decisions)
    vision_budget_exhausted = sum(bool(decision.get("budget_exhausted")) for decision in vision_decisions)
    question_type_counts: Dict[str, int] = {}
    for decision in vision_decisions:
        question_type = decision.get("question_type", "unknown")
        question_type_counts[question_type] = question_type_counts.get(question_type, 0) + 1

    correct_count = 0
    for result in results:
        gt_answer_idx = result.get("gt_answer_idx")
        if gt_answer_idx is None:
            continue
        gt_answer = f"a{gt_answer_idx}"
        pred_answer = result["final_answer"].strip().lower()
        if pred_answer == gt_answer.lower():
            correct_count += 1
    accuracy = correct_count / len(questions) if questions else 0.0
    total_wall_seconds = time.perf_counter() - run_started_at
    total_eval_seconds = time.perf_counter() - eval_started_at
    accumulated_question_seconds = sum(r.get("elapsed_seconds", 0.0) for r in simplified_results)
    completed_count = len([r for r in simplified_results if r["final_answer"]])
    failed_cases = [r for r in simplified_results if r.get("failed") or not r["final_answer"]]
    result_count = len(simplified_results)

    simplified_output = {
        "dataset": config.dataset,
        "model_path": config.llm_path,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "vision_policy": os.getenv("VISION_POLICY", "dynamic"),
        "vision_policy_signature": vision_policy_signature(),
        "total": total,
        "max_turn": max_turn,
        "metadata": {
            "avg_turns": sum(r["num_turns"] for r in simplified_results) / len(simplified_results)
            if simplified_results
            else 0,
            "vision_calls_total": total_vision_calls,
            "vision_search_actions_total": total_search_actions,
            "grounding_calls_total": total_grounding_calls,
            "vision_frames_total": total_vision_frames,
            "vision_clips_total": total_vision_clips,
            "vision_frames_per_question": total_vision_frames / total if total else 0,
            "vision_clips_per_question": total_vision_clips / total if total else 0,
            "vision_budget_exhausted_count": vision_budget_exhausted,
            "vision_question_type_counts": question_type_counts,
            "completed_questions": completed_count,
            "completion_rate": completed_count / result_count if result_count else 0,
            "accuracy": accuracy,
            "total_wall_seconds": total_wall_seconds,
            "model_load_seconds": model_ready_at - run_started_at,
            "total_eval_seconds": total_eval_seconds,
            "accumulated_question_seconds": accumulated_question_seconds,
            "avg_time_per_question": accumulated_question_seconds / total if total else 0,
            "failed_cases_count": len(failed_cases),
            "failed_cases": [
                {
                    "idx": i + 1,
                    "vid": r["vid"],
                    "question": r["question"],
                    "final_answer": r["final_answer"],
                    "error": r.get("error", "") or "missing_final_answer",
                }
                for i, r in enumerate(simplified_results)
                if r.get("failed") or not r["final_answer"]
            ],
        },
        "results": simplified_results,
    }

    with open(config.output_filename, "w", encoding="utf-8") as f:
        json.dump(simplified_output, f, ensure_ascii=False, indent=2)
    log_info(f"Summary results saved to {config.output_filename}")

    with open(config.detailed_output_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    if simplified_results:
        metadata = simplified_output["metadata"]
        log_info(f"\nStatistics for {config.dataset}:")
        log_info(f"Model path: {config.llm_path}")
        log_info(f"GPU memory utilization: {config.gpu_memory_utilization}")
        log_info(f"Average turns per question: {metadata['avg_turns']:.2f}")
        log_info(f"Total vision calls: {metadata['vision_calls_total']}")
        log_info(f"Total grounding calls: {metadata['grounding_calls_total']}")
        log_info(f"Vision calls per question: {metadata['vision_calls_total'] / len(simplified_results):.2f}")
        log_info(f"Grounding calls per question: {metadata['grounding_calls_total'] / len(simplified_results):.2f}")
        log_info(f"Vision frames per question: {metadata['vision_frames_per_question']:.2f}")
        log_info(f"Vision clips per question: {metadata['vision_clips_per_question']:.2f}")
        log_info(f"Vision budget exhausted: {metadata['vision_budget_exhausted_count']}")
        log_info(f"Completed questions: {metadata['completed_questions']}/{len(simplified_results)}")
        log_info(f"Completion rate: {metadata['completion_rate']:.2%}")
        log_info(f"Accuracy: {metadata['accuracy']:.2%}")
        log_info(f"Average time per question: {metadata['avg_time_per_question']:.2f}s")
        log_info(f"Failed cases: {metadata['failed_cases_count']}")

        turn_counts: Dict[int, int] = {}
        for r in simplified_results:
            turns = r["num_turns"]
            turn_counts[turns] = turn_counts.get(turns, 0) + 1
        log_info("\nTurn distribution:")
        for turns in sorted(turn_counts.keys()):
            count = turn_counts[turns]
            log_info(f"  {turns} turns: {count} questions ({count / len(simplified_results) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一版本地 Master LLM 评测脚本 (TVQA / TVQA+)")
    parser.add_argument("--dataset", type=str, required=True, choices=["tvqa", "tvqa_plus"], help="数据集类型")
    parser.add_argument("--llm-path", type=str, required=True, help="本地主模型路径（vLLM 可加载目录）")
    parser.add_argument("--max_turn", "-t", type=int, default=5, help="最大对话轮数")
    parser.add_argument("--gpu_memory_utilization", "-g", type=float, default=0.4, help="GPU内存利用率 (0,1]")
    parser.add_argument("--questions-path", type=str, default=None, help="问题文件路径")
    parser.add_argument("--subs-path", type=str, default=None, help="字幕文件路径")
    parser.add_argument("--base-frame-dir", type=str, default=None, help="视频帧根目录")
    parser.add_argument("--bbox-json-path", type=str, default=None, help="bbox JSON 路径（tvqa_plus 使用）")
    parser.add_argument("--output-filename", type=str, default=None, help="汇总输出文件")
    parser.add_argument("--detailed-output-filename", type=str, default=None, help="详细日志输出文件")
    parser.add_argument("--grounding-model", type=str, default=DEFAULT_GROUNDING_MODEL, help="grounding API 模型名")
    parser.add_argument("--vision-model", type=str, default=DEFAULT_VISION_MODEL, help="vision API 模型名")
    parser.add_argument("--grounding-base-url", type=str, default=DEFAULT_GROUNDING_BASE_URL, help="grounding API base URL")
    parser.add_argument("--vision-base-url", type=str, default=DEFAULT_VISION_BASE_URL, help="vision API base URL")
    parser.add_argument("--grounding-api-key", type=str, default=None, help="grounding API key（默认读取 qdd_api）")
    parser.add_argument("--vision-api-key", type=str, default=None, help="vision API key（默认读取 aliyun_api）")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=True, help="显示进度与关键日志（默认开启）")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="关闭常规日志输出")
    parser.add_argument("--debug", action="store_true", default=False, help="输出详细调试日志")
    parser.add_argument(
        "--grounding-cache-json-path",
        type=str,
        default=DEFAULT_GROUNDING_CACHE_JSON,
        help="grounding question->clip 缓存 JSON 路径",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="逐题 JSONL checkpoint 路径（默认在 summary 文件旁生成）",
    )
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 中已完成的下一题继续")
    args = parser.parse_args()

    ds = DATASET_DEFAULTS[args.dataset]
    questions_path = args.questions_path or str(ds["questions_path"])
    subs_path = args.subs_path or str(ds["subs_path"])
    base_frame_dir = args.base_frame_dir or str(ds["base_frame_dir"])
    bbox_json_path = args.bbox_json_path if args.bbox_json_path is not None else ds["bbox_json_path"]
    output_filename = args.output_filename or str(ds["output_filename"])
    detailed_output_filename = args.detailed_output_filename or str(ds["detailed_output_filename"])
    checkpoint_path = args.checkpoint_path or f"{output_filename}.checkpoint.jsonl"

    grounding_api_key = args.grounding_api_key or os.getenv("qdd_api")
    vision_api_key = args.vision_api_key or os.getenv("aliyun_api")

    if not Path(questions_path).is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not Path(subs_path).is_file():
        raise FileNotFoundError(f"Subtitles file not found: {subs_path}")
    if not Path(base_frame_dir).is_dir():
        raise FileNotFoundError(f"Frame directory not found: {base_frame_dir}")
    # if not Path(args.llm_path).exists():
    #     raise FileNotFoundError(f"LLM path not found: {args.llm_path}")
    if args.dataset == "tvqa_plus":
        if not bbox_json_path or not Path(str(bbox_json_path)).is_file():
            raise FileNotFoundError(f"BBox JSON file not found: {bbox_json_path}")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu_memory_utilization 必须在 (0, 1] 范围内")

    global config
    config = EvalConfig(
        dataset=args.dataset,
        questions_path=questions_path,
        subs_path=subs_path,
        base_frame_dir=base_frame_dir,
        bbox_json_path=str(bbox_json_path) if bbox_json_path else None,
        output_filename=output_filename,
        detailed_output_filename=detailed_output_filename,
        llm_path=args.llm_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        grounding_model=args.grounding_model,
        vision_model=args.vision_model,
        grounding_base_url=args.grounding_base_url,
        vision_base_url=args.vision_base_url,
        grounding_api_key=grounding_api_key,
        vision_api_key=vision_api_key,
        grounding_cache_json_path=args.grounding_cache_json_path,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        verbose=args.verbose,
        debug=args.debug,
    )
    initialize_clients()

    run_enhanced_pipeline(max_turn=args.max_turn)


if __name__ == "__main__":
    main()
