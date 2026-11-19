"""Read Stage 2 metadata and verify the source imagery for downstream stages."""

import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from .runs import validate_run_id


ROOT = Path(__file__).resolve().parents[1]


def read_stage2_export(boxes_path):
    with Path(boxes_path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    run_id = validate_run_id(metadata.get("run_id"))
    pair_id = metadata["pair_id"]
    if type(pair_id) is not int or pair_id < 0:
        raise ValueError("Stage 2 pair_id must be a nonnegative integer.")
    if not run_id.startswith(f"pair{pair_id}_"):
        raise ValueError("Run ID does not match pair_id.")
    before, after = metadata["before"], metadata["after"]
    boxes = after["boxes"]
    if before["boxes"] != boxes or any(section["num_detections"] != len(boxes) for section in (before, after)):
        raise ValueError("Expected matching before/after boxes in the Stage 2 export.")
    box_ids = [box["box_id"] for box in boxes]
    if any(type(box_id) is not int or box_id < 0 for box_id in box_ids) or len(set(box_ids)) != len(box_ids):
        raise ValueError("Stage 2 box IDs must be unique nonnegative integers.")
    return metadata


def load_source_images(metadata):
    before, after = metadata["before"], metadata["after"]
    expected_size = (after["width"], after["height"])
    loaded = []
    for section in (before, after):
        path = (ROOT / section["source_image"]).resolve()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != section.get("source_sha256"):
            raise ValueError(f"Source image hash mismatch: {path}. Regenerate Stage 2 for the current imagery.")
        with Image.open(BytesIO(data)) as image:
            if image.size != expected_size or (section["width"], section["height"]) != expected_size:
                raise ValueError("Stage 2 dimensions do not match the source imagery.")
            loaded.append((path, image.convert("RGB")))
    return loaded


def latest_stage2_export(output_dir):
    pointer = ROOT / output_dir / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("No latest Stage 2 export. Run the pipeline first or provide --boxes.")
    latest = json.loads(pointer.read_text(encoding="utf-8"))
    run_id = validate_run_id(latest.get("run_id"))
    return pointer.parent / run_id / "boxes.json"
