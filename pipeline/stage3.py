"""Score Stage 2 building crops with a shared AnomalyDINO reference bank.

Run from the repo root with an Anomalib 2.7.0 environment:
    python -m pipeline.stage3

By default the reference bank is built from the Stage 2 building crops of
the pair's before image, which must show the scene before any damage.
To use other known-undamaged buildings instead, pass a folder of PNG crops
with --references. Use the same kind of crop and surrounding context as
Stage 2; non-square references are center-cropped to a square.
Each after crop is a square window around its Stage 2 box, and its score
uses only the pixels inside the building footprint.
The first run may download the pretrained DINOv2 encoder. Stage 3 can run
in a separate Python environment from Stages 1 and 2.
"""

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from . import config


ROOT = Path(__file__).resolve().parents[1]
ENCODER = "vit_small_patch14_dinov2"
IMAGE_SIZE = (252, 252)
OVERLAY_MAX = 1.0
# Footprints covering less of the box than this are treated as failed roof fits.
MIN_FOOTPRINT_FRACTION = 0.25


def crop_bounds(bbox, size):
    xmin, ymin, xmax, ymax = bbox
    if not all(math.isfinite(value) for value in bbox) or xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid Stage 2 bounding box: {bbox}")
    width, height = size
    bounds = (
        max(0, math.floor(xmin)), max(0, math.floor(ymin)),
        min(width, math.ceil(xmax)), min(height, math.ceil(ymax)),
    )
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError(f"Stage 2 bounding box lies outside the image: {bbox}")
    return bounds


def square_bounds(bounds, size):
    # The encoder resizes to a fixed square, so square crops avoid distortion.
    xmin, ymin, xmax, ymax = bounds
    width, height = size
    side = max(xmax - xmin, ymax - ymin)
    if side > min(width, height):
        return bounds
    x0 = min(max(0, xmin - (side - (xmax - xmin)) // 2), width - side)
    y0 = min(max(0, ymin - (side - (ymax - ymin)) // 2), height - side)
    return (x0, y0, x0 + side, y0 + side)


def center_square(image):
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def scoring_region(box):
    xmin, ymin, xmax, ymax = box["bbox_xyxy"]
    corners = np.asarray(box.get("rotated_corners", []), dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError(f"Invalid Stage 2 rotated corners for box {box['box_id']}.")
    xs, ys = corners.T
    area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    if area < MIN_FOOTPRINT_FRACTION * (xmax - xmin) * (ymax - ymin):
        return "box", [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return "footprint", [tuple(point) for point in corners.tolist()]


def region_mask(polygon, crop_box, shape):
    x0, y0, x1, y1 = crop_box
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in polygon], fill=255)
    mask = np.asarray(mask.resize(shape[::-1], Image.Resampling.NEAREST)) > 0
    return mask if mask.any() else np.ones(shape, dtype=bool)


def mean_top1p(values):
    # Matches AnomalyDINO's image score: the mean of the highest 1% of values.
    count = max(int(values.size * 0.01), 1)
    return float(np.sort(values, axis=None)[-count:].mean())


def run_stage3_anomalydino(boxes_path, reference_dir, output_dir, device):
    with open(boxes_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    pair_id = metadata["pair_id"]
    before, after = metadata["before"], metadata["after"]
    boxes = after["boxes"]
    if before["boxes"] != boxes or any(
        section["num_detections"] != len(boxes) for section in (before, after)
    ):
        raise ValueError("Expected matching before/after boxes in the Stage 2 export.")
    box_ids = [box["box_id"] for box in boxes]
    if any(type(box_id) is not int or box_id < 0 for box_id in box_ids) or len(set(box_ids)) != len(box_ids):
        raise ValueError("Stage 2 box IDs must be unique nonnegative integers.")

    source_path = (ROOT / after["source_image"]).resolve()
    with Image.open(source_path) as image:
        after_img = image.convert("RGB")
    expected_size = (after["width"], after["height"])
    if after_img.size != expected_size or (before["width"], before["height"]) != expected_size:
        raise ValueError("Stage 2 dimensions do not match the source imagery.")
    bounds = [
        square_bounds(crop_bounds(box["bbox_xyxy"], after_img.size), after_img.size)
        for box in boxes
    ]
    regions = [scoring_region(box) for box in boxes]

    print(f"\n--- STAGE 3: ANOMALYDINO (Pair {pair_id}) ---")
    if not boxes:
        print("Stage 2 exported zero boxes. Nothing to score.")
        return None
    if reference_dir is None:
        before_path = (ROOT / before["source_image"]).resolve()
        if before_path == source_path:
            raise ValueError("The after image cannot be used as a normal reference.")
        with Image.open(before_path) as image:
            before_img = image.convert("RGB")
        if before_img.size != expected_size:
            raise ValueError("Stage 2 dimensions do not match the source imagery.")
        reference_images = [before_img.crop(crop_box) for crop_box in bounds]
        reference_sources = [f"{before_path.as_posix()}#box_{box_id}" for box_id in box_ids]
        reference_label = f"before-image crops of {before_path.name}"
    else:
        reference_dir = Path(reference_dir)
        references = sorted(
            path for path in reference_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ) if reference_dir.is_dir() else []
        if not references:
            raise ValueError(
                f"No reference PNGs found in '{reference_dir}'. Provide crops of separate "
                "undamaged buildings there, or omit --references to use the before image."
            )
        if source_path in {path.resolve() for path in references}:
            raise ValueError("The after image cannot be used as a normal reference.")
        reference_images = []
        for reference in references:
            with Image.open(reference) as image:
                reference_images.append(center_square(image.convert("RGB")))
        reference_sources = [path.resolve().as_posix() for path in references]
        reference_label = f"PNG crops in {reference_dir}"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run with --device cpu.')

    from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel

    model = AnomalyDINOModel(
        num_neighbours=1, encoder_name=ENCODER, masking=False, coreset_subsampling=False
    ).to(device)
    preprocess = AnomalyDINO.configure_pre_processor(image_size=IMAGE_SIZE).to(device)

    def tensor(image):
        rgb = torch.from_numpy(np.array(image.convert("RGB")))
        return preprocess(rgb.permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0)

    print(f"Building a shared bank from {len(reference_images)} {reference_label} on {device}.")
    model.train()
    model.feature_encoder.eval()
    with torch.inference_mode():
        for reference in reference_images:
            model(tensor(reference))
        model.fit()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir))
    results = []
    model.eval()
    with torch.inference_mode():
        for box_id, crop_box, (region, polygon) in zip(box_ids, bounds, regions):
            crop = after_img.crop(crop_box)
            result = model(tensor(crop))
            unmasked_score = float(result.pred_score.item())
            anomaly_map = result.anomaly_map.cpu().numpy().squeeze()
            if not math.isfinite(unmasked_score) or not np.isfinite(anomaly_map).all():
                raise RuntimeError(f"AnomalyDINO returned non-finite values for box {box_id}.")
            score = mean_top1p(anomaly_map[region_mask(polygon, crop_box, anomaly_map.shape)])

            crop_name = f"box_{box_id}_after.png"
            map_name = f"box_{box_id}_anomaly.npy"
            overlay_name = f"box_{box_id}_overlay.png"
            crop.save(run_dir / crop_name)
            np.save(run_dir / map_name, anomaly_map)
            # Apply a shared display scale while preserving the original float map.
            scaled = np.clip(anomaly_map / OVERLAY_MAX, 0, 1)
            heat = Image.fromarray((scaled * 255).astype(np.uint8))
            heat = ImageOps.colorize(heat, black="#111126", white="#fff12c", mid="#d04518")
            overlay = Image.blend(crop, heat.resize(crop.size, Image.Resampling.BILINEAR), 0.45)
            ImageDraw.Draw(overlay).polygon(
                [(x - crop_box[0], y - crop_box[1]) for x, y in polygon],
                outline="cyan", width=2,
            )
            overlay.save(run_dir / overlay_name)
            results.append({
                "box_id": box_id,
                "crop_bbox_xyxy": crop_box,
                "scoring_region": region,
                "after_crop": crop_name,
                "anomaly_score": score,
                "unmasked_anomaly_score": unmasked_score,
                "anomaly_map": map_name,
                "anomaly_overlay": overlay_name,
            })
            print(f"Box {box_id}: raw anomaly score {score:.5f} "
                  f"({region}; whole crop {unmasked_score:.5f})")

    report_path = run_dir / "anomalies.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "pair_id": pair_id,
            "source_image": source_path.as_posix(),
            "stage2_boxes": Path(boxes_path).resolve().as_posix(),
            "encoder": ENCODER,
            "reference_images": reference_sources,
            "overlay_display_range": [0, OVERLAY_MAX],
            "boxes": results,
        }, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Stage 3 results saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=Path, default=ROOT / config.STAGE2_OUTPUT_DIR / "boxes.json")
    parser.add_argument(
        "--references", type=Path, default=None,
        help="Folder of undamaged PNG crops; defaults to the pair's before-image crops.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "stage3outputs")
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()
    run_stage3_anomalydino(args.boxes, args.references, args.output, args.device)


if __name__ == "__main__":
    main()
