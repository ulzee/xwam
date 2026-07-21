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


def load_model(force_download=False):
    """force_download=True bypasses the local HF cache entirely (blobs AND
    the refs/snapshots symlinks) and re-fetches from the Hub -- use when a
    resumed AWS instance's cache looks present but is actually corrupt
    (e.g. a snapshot symlink pointing at a blob that didn't survive an
    EBS/disk swap). Caller (stage1a.py) is responsible for also making sure
    HF_HUB_OFFLINE isn't set to "1" in this case, or from_pretrained will
    refuse to hit the network at all regardless of this flag."""
    global _model, _processor
    if _model is None:
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda", force_download=force_download)
        _processor = AutoProcessor.from_pretrained(MODEL_ID, force_download=force_download)
    return _model, _processor


PROMPT_TEMPLATE = """You are given a narration of what happens in this video: "{narration}"

Hands are already tracked separately -- do NOT list hands, arms, or fingers.

This is for embodied learning, where the point is to reconstruct the full
physical interaction, not just the headline object the narration names.
Watch the whole video (don't just parse the narration text) and identify
every physical object relevant to the action. The narration is a strong
hint but is frequently incomplete -- e.g. it may name only the target
("C cuts a tomato") and leave the tool implicit ("with a knife"), or name
only the task and omit a container, surface, or second party entirely.
Find what it leaves out too.

You will fill in TWO separate lists: "objects" (things actually part of the
action) and "background" (everything else physical and nameable in the
scene). Split like this because "background" items become negative
examples during training -- they teach the downstream model what NOT to
treat as a target, so being thorough there is more valuable than being
sparse. Don't let that thoroughness leak into "objects" though: that list
drives what actually gets tracked, so it must stay restricted to things
truly part of THIS action.

"objects" -- up to 8, covering (as applicable -- an object can fall into
more than one of these, and not every video has all of them):

1. TOOLS/INSTRUMENTS the embodiment actively wields to act on something
   else -- knife, sponge, screwdriver, spoon, phone, remote, brush, etc.
   Include these even when the narration only names the target and leaves
   the tool implicit.
2. TARGETS -- whatever the hand OR a wielded tool is directly acting on:
   cut, stirred, pressed, wiped, opened, typed on, poured into, etc.
   Include objects acted on indirectly through a tool, not just ones the
   bare hand touches.
3. ANIMALS/PEOPLE the embodiment interacts with as part of the narrated
   action -- petting an animal, handing something to a person, shaking
   hands, etc. These matter as anchors for the action even though they
   aren't picked up like a tool.
4. SURFACES/CONTAINERS the embodiment deliberately makes contact with as
   part of the action -- a table or floor something is set down on or
   picked up from, a bowl or box something goes into, a shelf something
   comes off of. Only if the action actually involves it (placing onto,
   taking from, resting on) -- not just anything visible nearby.

Rules for "objects": 8 is a ceiling, not a target -- most actions genuinely
involve only 1-4 objects. Do NOT pad this list with things that are just
visible in the room to reach a higher count. For every object here, you
should be able to point to a moment in the video where the embodiment, or
a tool in its hand, actually makes contact with it as part of THIS action
-- if you can't, it belongs in "background" instead, not here.

"background" -- everything else physical, mostly-rigid, and nameable that
you can see but that is NOT touched or used as part of this action:
other furniture, appliances, decor, clutter, unused utensils, other
people/animals not involved, etc. Unlike "objects", err heavily on the
side of MORE here -- list every distinct physical thing you can identify,
even minor or partially-visible ones. There's no fixed cap; a thorough
scene can easily have 15-20+ background items. Still name each as a
distinct, specific item (e.g. "toaster", "cereal box") rather than vague
groupings (e.g. "kitchen stuff").

General rules for what counts as a nameable physical object (applies to
both lists):
- A physical, mostly-rigid or semi-rigid thing whose 2D position could be followed through the video.
- EXCLUDE fluids, splashes, smoke, or steam (e.g. "water") -- no fixed shape to track.
- EXCLUDE hands, arms, or fingers (tracked separately).
- EXCLUDE pure architecture/environment with no distinct extent (a wall, the floor as an infinite plane, the sky, a room in general) -- unless the action itself is deliberately using it (see SURFACES above), in which case it goes in "objects", not "background".
- Prefer the narration's own wording for an object's name when it names one directly; otherwise use its plain common name.

Respond with ONLY a JSON object, no other text, in this exact format:
{{"task": "<one-sentence description of the task/event, informed by the narration>", "objects": ["<up to 8 objects truly part of the action -- empty list if none>"], "background": ["<as many distinct untouched physical items as you can identify -- empty list if none>"]}}

Examples:
narration "C cuts a tomato with a knife" ->
{{"task": "cutting a tomato with a knife", "objects": ["knife", "tomato", "cutting board"], "background": ["stove", "kitchen sink", "dish rack", "paper towel roll", "window", "fruit bowl"]}}
narration "C pets the dog" ->
{{"task": "petting a dog", "objects": ["dog"], "background": ["sofa", "coffee table", "TV", "rug", "lamp"]}}
narration "C puts the cup on the table" ->
{{"task": "placing a cup on the table", "objects": ["cup", "table"], "background": ["chair", "napkin holder", "salt shaker", "window blinds"]}}
"""

# prompting alone doesn't reliably keep the model from naming fluids/scenery
# (observed previously: "running water" slipped through despite an explicit
# exclusion rule) -- this is a cheap backstop, not a replacement for the prompt.
EXCLUDED_KEYWORDS = ["water", "steam", "smoke", "fluid", "splash", "vapor", "hand", "arm", "finger"]


def is_excluded(name):
    lname = name.lower()
    return any(kw in lname for kw in EXCLUDED_KEYWORDS)


def identify(video_path, narration=DEFAULT_NARRATION, fps=2.0, max_pixels=360 * 640, max_new_tokens=768):
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
    # return_video_metadata=True carries forward the real per-video VideoMetadata
    # (native fps, total_num_frames, the exact frame indices) that
    # process_vision_info computed when it decoded/sampled the video per the
    # "fps" key above. Without this, processor() below has no metadata, assumes
    # a fake native fps=24, and: (a) silently RE-samples the already-sampled
    # frames down again (observed: our 6-frame, fps~1.76 clip got re-sampled to
    # just 4 frames, hitting Qwen3VLVideoProcessor's min_frames floor), and
    # (b) mislabels each frame's <N.M seconds> prompt tag using the fake fps.
    # do_sample_frames=False (from video_kwargs) tells the processor these
    # frames are already sampled -- don't sample again.
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True)
    videos = [v for v, m in video_inputs]
    video_metadata = [m for v, m in video_inputs]
    inputs = processor(text=[text], images=image_inputs, videos=videos, video_metadata=video_metadata,
                        padding=True, return_tensors="pt", **video_kwargs).to("cuda")
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
    """Returns (task_description, filtered_object_list, filtered_background_list).
    `objects` is what gets tracked (GDINO+SAM2 in stage1b.py); `background`
    is untouched/uninvolved items the model was told to overindex on --
    kept only as negatives, never fed into tracking."""
    raw = identify(video_path, narration=narration, **kwargs)
    parsed = parse_json_response(raw)
    task = parsed.get("task", "")
    objects = [o for o in parsed.get("objects", []) if not is_excluded(o)]
    background = [o for o in parsed.get("background", []) if not is_excluded(o)]
    return task, objects, background


if __name__ == "__main__":
    video_path = sys.argv[1]
    narration = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NARRATION
    raw = identify(video_path, narration=narration)
    print("RAW:", raw)
    task, objects, background = identify_objects(video_path, narration=narration)
    print("TASK:", task)
    print("OBJECTS (post-filter):", objects)
    print("BACKGROUND (post-filter):", background)
