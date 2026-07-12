"""Narration-seeded object identification, adapted from ~/trace/annot/identify.py.

Difference from the original: the caller supplies a narration string (Ego4D
ships one per clip; for non-Ego4D sources it defaults to a manual description)
which is injected into the prompt as a strong hint, instead of asking Qwen to
infer the task blind from keyframes alone. Blind inference previously surfaced
background scenery and needed a keyword post-filter backstop -- narration
should be a strictly stronger, already-available signal for real Ego4D clips.
The keyword backstop is kept regardless, since it's cheap insurance.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import json
import re
import sys
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# Qwen2.5-VL-3B-Instruct (used by the original ~/trace/annot/identify.py) was
# deleted from the HF cache after the switch to Qwen3-VL-8B for hand
# localization (see project_qwen_sam2_annotation_pipeline memory) -- reuse
# the already-cached Qwen3-VL-8B here too rather than re-downloading the 3B.
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_NARRATION = "a person washing lettuce in the sink"

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is None:
        _model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
    return _model, _processor


PROMPT_TEMPLATE = """You are given a narration of what happens in this video: "{narration}"

Hands are already tracked separately -- do NOT list hands, arms, or any body part.

Using the narration as your primary guide (it names the task and usually the
key object(s) directly), identify up to 5 physical objects that a hand is
actively touching, holding, or manipulating in pursuit of that task. Watch the
video to find their exact common names and to catch anything the narration
implies but doesn't name outright (e.g. a container the narration's object
sits in).

Rules for what counts as a trackable OBJECT:
- A physical, mostly-rigid or semi-rigid thing whose 2D position you could follow through the video AND that a hand actually picks up, holds, or moves -- not just touches or reaches near.
- EXCLUDE fluids, splashes, smoke, or steam (e.g. "water") -- they have no fixed shape to track.
- EXCLUDE fixtures and static background/scenery, even ones a hand brushes against or water runs over (counters, walls, sinks, faucets, appliances) -- these stay bolted in place and are never actually picked up or moved, so they are NOT trackable objects even if visually prominent or part of the task.
- EXCLUDE hands, arms, or any body part.
- Only include something the hand actively picks up, holds, or repositions in pursuit of the task.
- Prefer the narration's own wording for an object's name when it names one directly.

Respond with ONLY a JSON object, no other text, in this exact format:
{{"task": "<one-sentence description of the task/event, informed by the narration>", "objects": ["<up to 5 trackable objects -- empty list if none>"]}}

Example, narration "C cuts a tomato with a knife":
{{"task": "cutting a tomato with a knife", "objects": ["knife", "tomato"]}}
"""

# prompting alone doesn't reliably keep the model from naming fluids/scenery
# (observed previously: "running water" slipped through despite an explicit
# exclusion rule) -- this is a cheap backstop, not a replacement for the prompt.
EXCLUDED_KEYWORDS = ["water", "steam", "smoke", "fluid", "splash", "vapor", "hand", "arm", "finger"]


def is_excluded(name):
    lname = name.lower()
    return any(kw in lname for kw in EXCLUDED_KEYWORDS)


def identify(video_path, narration=DEFAULT_NARRATION, fps=2.0, max_pixels=360 * 640, max_new_tokens=256):
    model, processor = load_model()
    prompt = PROMPT_TEMPLATE.format(narration=narration)
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    raw = processor.batch_decode(out_trimmed, skip_special_tokens=True)[0]
    return raw


def parse_json_response(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw!r}")
    return json.loads(match.group(0))


def identify_objects(video_path, narration=DEFAULT_NARRATION, **kwargs):
    """Returns (task_description, filtered_object_list)."""
    raw = identify(video_path, narration=narration, **kwargs)
    parsed = parse_json_response(raw)
    task = parsed.get("task", "")
    objects = [o for o in parsed.get("objects", []) if not is_excluded(o)]
    return task, objects


if __name__ == "__main__":
    video_path = sys.argv[1]
    narration = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NARRATION
    raw = identify(video_path, narration=narration)
    print("RAW:", raw)
    task, objects = identify_objects(video_path, narration=narration)
    print("TASK:", task)
    print("OBJECTS (post-filter):", objects)
