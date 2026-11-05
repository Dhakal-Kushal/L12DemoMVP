import json
import os

import numpy as np
import torch
from PIL import ImageDraw
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from . import config
from .geometry import clean_merge_and_pad_boxes, compute_rotated_box_corners


def run_stage2_owlv2(before_img, after_img, before_path, after_path, user_query):
    if before_img.size != after_img.size:
        raise ValueError(
            "Before and after images must have identical pixel dimensions to "
            f"share boxes; got {before_img.size} and {after_img.size}."
        )
    print(f"\n--- STAGE 2: OWLv2 OBJECT DETECTION (Pair {config.PAIR_ID}) ---")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    ).to(config.DEVICE)
    model.eval()
    # Detect only before; both images share the resulting geometry.
    texts = [[user_query]]
    images = [before_img]
    inputs = processor(text=texts, images=images, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

        # OWLv2 target sizes use height, width.
        target_sizes = torch.tensor(
            [before_img.size[::-1]], device=config.DEVICE
        )
        if hasattr(processor, "post_process_object_detection"):
            results = processor.post_process_object_detection(
                outputs=outputs,
                threshold=config.STAGE2_BOX_THRESHOLD,
                target_sizes=target_sizes,
            )
        else:

            results = processor.image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=config.STAGE2_BOX_THRESHOLD,
                target_sizes=target_sizes,
            )

    result = results[0]
    raw_boxes, raw_scores = [], []
    for box, score in zip(result["boxes"].detach().cpu(), result["scores"].detach().cpu()):
        if not torch.isfinite(box).all().item() or not torch.isfinite(score).item():
            raise RuntimeError(f"OWLv2 returned a non-finite detection for {before_path}.")
        xmin, ymin, xmax, ymax = box.tolist()
        raw_boxes.append([
            max(0.0, min(float(before_img.width), xmin)),
            max(0.0, min(float(before_img.height), ymin)),
            max(0.0, min(float(before_img.width), xmax)),
            max(0.0, min(float(before_img.height), ymax)),
        ])
        raw_scores.append(score.item())

    final_boxes, groups = clean_merge_and_pad_boxes(
        raw_boxes, before_img.width, before_img.height, return_groups=True
    )
    boxes = [
        {
            "box_id": index,
            "label": user_query,

            "score": round(max(raw_scores[member] for member in members), 4),
            "bbox_xyxy": [round(value, 2) for value in box],
        }
        for index, (box, members) in enumerate(zip(final_boxes, groups))
    ]
    # Footprints already include the 10% safety margin for contour crops.
    before_np = np.asarray(before_img)
    for box in boxes:

        corners = compute_rotated_box_corners(before_np, box["bbox_xyxy"])
        box["rotated_corners"] = [
            [round(float(x), 2), round(float(y), 2)] for x, y in corners
        ]


        # The Stage 3 crop must enclose all rotated corners.
        xs, ys = zip(*box["rotated_corners"])
        xmin, ymin, xmax, ymax = box["bbox_xyxy"]
        box["bbox_xyxy"] = [min(xmin, min(xs)), min(ymin, min(ys)),
                            max(xmax, max(xs)), max(ymax, max(ys))]
    print(f"[{os.path.basename(before_path)}] {len(raw_boxes)} raw detections; "
          f"{len(boxes)} final merged and padded boxes.")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.STAGE2_OUTPUT_DIR, exist_ok=True)
    metadata = {
        "pair_id": config.PAIR_ID,
        "query": user_query,
        "threshold": config.STAGE2_BOX_THRESHOLD,
    }
    for prefix, image, source_path in zip(
        ("before", "after"), (before_img, after_img), (before_path, after_path)
    ):
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for box in boxes:
            draw.polygon([tuple(point) for point in box["rotated_corners"]],
                         outline="red", width=2)

        boxed_path = os.path.join(config.OUTPUT_DIR, f"{prefix}{config.PAIR_ID}_boxed.png")
        annotated.save(boxed_path)
        metadata[prefix] = {
            "source_image": source_path.replace(os.sep, "/"),
            "boxed_image": boxed_path.replace(os.sep, "/"),
            "width": image.width,
            "height": image.height,
            "num_detections": len(boxes),
            "boxes": boxes,
        }
        action = "drawn" if prefix == "before" else "projected from before"
        print(f"[{os.path.basename(source_path)}] {len(boxes)} boxes {action}. Saved: {boxed_path}")

    json_path = os.path.join(config.STAGE2_OUTPUT_DIR, "boxes.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Stage 3 metadata saved: {json_path}")
    return json_path
