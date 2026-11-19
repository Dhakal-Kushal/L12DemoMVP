import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .runs import new_run_id, source_hash
from . import config
from .geometry import (
    nms_and_pad_boxes, compute_rotated_box_corners,
    suppress_duplicate_outlines, suppress_nested_outlines,
)


CHECKPOINTS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
}


def detection_regions(width, height, tile_size, overlap):
    """Full image first, followed by evenly spaced overlapping tiles."""
    if not isinstance(tile_size, int) or tile_size < 0:
        raise ValueError("STAGE2_TILE_SIZE must be a nonnegative integer.")
    if not 0 <= overlap < 1:
        raise ValueError("STAGE2_TILE_OVERLAP must be in [0, 1).")
    yield (0, 0, width, height)
    if tile_size == 0 or max(width, height) <= tile_size:
        return

    stride = max(1, int(tile_size * (1 - overlap)))

    def offsets(length):
        span = max(0, length - tile_size)
        intervals = (span + stride - 1) // stride
        if intervals == 0:
            return [0]
        return [i * span // intervals for i in range(intervals + 1)]

    for top in offsets(height):
        for left in offsets(width):
            yield (left, top, min(width, left + tile_size), min(height, top + tile_size))


@torch.inference_mode()
def detect_region(image, region, query, processor, model, detector, threshold):
    left, top, right, bottom = region
    view = image if region == (0, 0, *image.size) else image.crop(region)
    texts = [[query]] if detector == "owlv2" else query.strip().rstrip(".").lower() + "."
    inputs = processor(text=texts, images=[view], return_tensors="pt").to(config.DEVICE)
    outputs = model(**inputs)
    sizes = [view.size[::-1]]
    if detector == "grounding_dino":
        result = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids, threshold=threshold,
            text_threshold=config.STAGE2_DINO_TEXT_THRESHOLD, target_sizes=sizes,
        )[0]
    else:
        postprocessor = processor if hasattr(processor, "post_process_object_detection") else processor.image_processor
        result = postprocessor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=sizes,
        )[0]
    boxes, scores = result["boxes"], result["scores"]
    if not torch.isfinite(boxes).all().item() or not torch.isfinite(scores).all().item():
        raise RuntimeError(f"{detector} returned a non-finite detection for region {region}.")

    detections = []
    for (x1, y1, x2, y2), score in zip(boxes.tolist(), scores.tolist()):
        # Leave cut-off roofs to overlapping tiles or the full scene. Real image edges are valid.
        if ((left > 0 and x1 <= 2) or (top > 0 and y1 <= 2)
                or (right < image.width and x2 >= view.width - 2)
                or (bottom < image.height and y2 >= view.height - 2)):
            continue
        box = [max(left, min(right, left + x1)), max(top, min(bottom, top + y1)),
               max(left, min(right, left + x2)), max(top, min(bottom, top + y2))]
        detections.append((box, score))
    return detections


def run_stage2(before_img, after_img, before_path, after_path, user_query):
    if before_img.size != after_img.size:
        raise ValueError(
            "Before and after images must have identical pixel dimensions to "
            f"share boxes; got {before_img.size} and {after_img.size}."
        )
    run_id = new_run_id(config.PAIR_ID)
    source_hashes = [source_hash(path) for path in (before_path, after_path)]
    for image, path in ((before_img, before_path), (after_img, after_path)):
        with Image.open(path) as source:
            if source.size != image.size or source.convert("RGB").tobytes() != image.tobytes():
                raise ValueError(f"Loaded image differs from source file: {path}. Run the pipeline again.")
    print(f"Run: {run_id}")
    detector = config.STAGE2_DETECTOR
    method = config.STAGE2_OUTLINE_METHOD
    if detector not in CHECKPOINTS or method not in ("otsu", "sam2"):
        raise ValueError(f"Unsupported Stage 2 backend: {detector} / {method}")
    threshold = config.STAGE2_BOX_THRESHOLD if detector == "owlv2" else config.STAGE2_DINO_BOX_THRESHOLD
    if not 0 <= threshold <= 1 or not 0 <= config.STAGE2_DINO_TEXT_THRESHOLD <= 1:
        raise ValueError("Detection thresholds must be between 0 and 1.")
    checkpoint = CHECKPOINTS[detector]
    print(f"Stage 2: {detector} with {method} outlines (Pair {config.PAIR_ID})")
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(config.DEVICE).eval()
    regions = list(detection_regions(*before_img.size, config.STAGE2_TILE_SIZE, config.STAGE2_TILE_OVERLAP))
    raw_boxes, raw_scores = [], []
    for region in regions:
        for box, score in detect_region(before_img, region, user_query, processor, model, detector, threshold):
            raw_boxes.append(box)
            raw_scores.append(score)

    del model, processor
    retained = nms_and_pad_boxes(
        raw_boxes, raw_scores, before_img.width, before_img.height,
        iou_threshold=config.STAGE2_NMS_IOU_THRESHOLD,
    )
    boxes = [
        {
            "label": user_query,

            "score": round(raw_scores[source_index], 4),
            "bbox_xyxy": [round(value, 2) for value in box],
        }
        for source_index, box in retained
    ]
    if method == "sam2":
        from .sam2_roofs import SAM2_CHECKPOINT, segment_outlines
        outlines = segment_outlines(before_img, [raw_boxes[i] for i, _ in retained], config.DEVICE)
    else:
        before_np = np.asarray(before_img)
        outlines = [(compute_rotated_box_corners(before_np, box["bbox_xyxy"]), None) for box in boxes]
    for box, (corners, fallback_reason) in zip(boxes, outlines, strict=True):
        if fallback_reason:
            box["outline_fallback"] = fallback_reason
        box["rotated_corners"] = [
            [round(float(x), 2), round(float(y), 2)] for x, y in corners
        ]


        # The Stage 3 crop must enclose all rotated corners.
        xs, ys = zip(*box["rotated_corners"])
        xmin, ymin, xmax, ymax = box["bbox_xyxy"]
        box["bbox_xyxy"] = [min(xmin, min(xs)), min(ymin, min(ys)),
                            max(xmax, max(xs)), max(ymax, max(ys))]
    boxes = suppress_duplicate_outlines(boxes, config.STAGE2_OUTLINE_IOU_THRESHOLD)
    boxes = suppress_nested_outlines(boxes, config.STAGE2_CONTAINMENT_THRESHOLD)
    for index, box in enumerate(boxes):
        box["box_id"] = index
    print(f"[{os.path.basename(before_path)}] {len(raw_boxes)} proposals from {len(regions)} views; "
          f"{len(retained)} after box NMS; {len(boxes)} after outline and containment filtering.")
    boxed_dir = Path(config.OUTPUT_DIR) / run_id
    json_dir = Path(config.STAGE2_OUTPUT_DIR) / run_id
    boxed_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)
    if source_hashes != [source_hash(path) for path in (before_path, after_path)]:
        raise ValueError("Source imagery changed during Stage 2. Run the pipeline again.")
    metadata = {
        "run_id": run_id,
        "pair_id": config.PAIR_ID,
        "query": user_query,
        "threshold": threshold,
        "detector": checkpoint,
        "outline_method": method,
        "tile_size": config.STAGE2_TILE_SIZE,
        "tile_overlap": config.STAGE2_TILE_OVERLAP,
        "tile_count": len(regions) - 1,
        "nms_iou_threshold": config.STAGE2_NMS_IOU_THRESHOLD,
        "outline_iou_threshold": config.STAGE2_OUTLINE_IOU_THRESHOLD,
        "containment_threshold": config.STAGE2_CONTAINMENT_THRESHOLD,
    }
    if detector == "grounding_dino":
        metadata["text_threshold"] = config.STAGE2_DINO_TEXT_THRESHOLD
    if method == "sam2":
        metadata["segmenter"] = SAM2_CHECKPOINT
    for prefix, image, source_path, digest in zip(
        ("before", "after"), (before_img, after_img), (before_path, after_path), source_hashes
    ):
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for box in boxes:
            draw.polygon([tuple(point) for point in box["rotated_corners"]],
                         outline="red", width=2)
            x, y = box["rotated_corners"][0]
            draw.text((max(0, min(x, image.width - 30)), max(18, min(y, image.height - 15))),
                      f"B{box['box_id']:02d}", fill="white", stroke_width=2, stroke_fill="black")
        draw.rectangle((0, 0, image.width, 17), fill="#171b24")
        draw.text((4, 2), f"Run: {run_id}", fill="white")

        boxed_path = str(boxed_dir / f"{prefix}{config.PAIR_ID}_boxed.png")
        annotated.save(boxed_path)
        metadata[prefix] = {
            "source_sha256": digest,
            "source_image": Path(source_path).resolve().as_posix(),
            "boxed_image": Path(boxed_path).resolve().as_posix(),
            "width": image.width,
            "height": image.height,
            "num_detections": len(boxes),
            "boxes": boxes,
        }
        action = "drawn" if prefix == "before" else "projected from before"
        print(f"[{os.path.basename(source_path)}] {len(boxes)} boxes {action}. Saved: {boxed_path}")

    json_path = str(json_dir / "boxes.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    pointer = Path(config.STAGE2_OUTPUT_DIR) / "latest.json"
    temporary = pointer.with_name(f".{run_id}.tmp")
    temporary.write_text(json.dumps({"run_id": run_id, "boxes": f"{run_id}/boxes.json"}, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, pointer)
    print(f"Stage 3 metadata saved: {json_path}")
    return json_path


# Older scripts import this entry point by its original name.
run_stage2_owlv2 = run_stage2
