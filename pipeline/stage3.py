"""Score each Stage 2 after-building crop against its matching before crop."""

import argparse
import json
import math
from pathlib import Path
import hashlib
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from .runs import validate_run_id
from . import config
from .heatmaps import project_anomaly, render_heatmap


ROOT = Path(__file__).resolve().parents[1]
ENCODER = "vit_small_patch14_dinov2"
IMAGE_SIZE = (252, 252)


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


def run_stage3_anomalydino(boxes_path, output_dir, device, *, display_range=None, mask_roofs=True):
    if display_range is not None:
        low, high = map(float, display_range)
        if not np.isfinite([low, high]).all() or not 0 <= low < high:
            raise ValueError("Display range must contain finite values with 0 <= low < high.")
    with open(boxes_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    run_id = validate_run_id(metadata.get("run_id"))
    pair_id = metadata["pair_id"]
    if type(pair_id) is not int or pair_id < 0:
        raise ValueError("Stage 2 pair_id must be a nonnegative integer.")
    if not run_id.startswith(f"pair{pair_id}_"):
        raise ValueError("Run ID does not match pair_id.")
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}. Use --output with a different directory to rerun Stage 3.")
    before, after = metadata["before"], metadata["after"]
    boxes = after["boxes"]
    if before["boxes"] != boxes or any(
        section["num_detections"] != len(boxes) for section in (before, after)
    ):
        raise ValueError("Expected matching before/after boxes in the Stage 2 export.")
    box_ids = [box["box_id"] for box in boxes]
    if any(type(box_id) is not int or box_id < 0 for box_id in box_ids) or len(set(box_ids)) != len(box_ids):
        raise ValueError("Stage 2 box IDs must be unique nonnegative integers.")

    before_path = (ROOT / before["source_image"]).resolve()
    after_path = (ROOT / after["source_image"]).resolve()
    loaded = []
    for section, path in ((before, before_path), (after, after_path)):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != section.get("source_sha256"):
            raise ValueError(f"Source image hash mismatch: {path}. Regenerate Stage 2 for the current imagery.")
        with Image.open(BytesIO(data)) as image:
            loaded.append(image.convert("RGB"))
    before_img, after_img = loaded
    expected_size = (after["width"], after["height"])
    if (after_img.size != expected_size or before_img.size != expected_size
            or (before["width"], before["height"]) != expected_size):
        raise ValueError("Stage 2 dimensions do not match the source imagery.")

    print(f"Run: {run_id}")
    print(f"\n--- STAGE 3: ANOMALYDINO (Pair {pair_id}) ---")
    if not boxes:
        print("Stage 2 exported zero boxes. Nothing to score.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run with --device cpu.')

    if boxes:
        from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
        from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel

        model = AnomalyDINOModel(
            num_neighbours=1, encoder_name=ENCODER, masking=False, coreset_subsampling=False
        ).to(device)
        preprocess = AnomalyDINO.configure_pre_processor(image_size=IMAGE_SIZE).to(device)

    def tensor(image):
        rgb = torch.from_numpy(np.array(image.convert("RGB")))
        return preprocess(rgb.permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    results = []
    scene_map = np.zeros((after_img.height, after_img.width), dtype=np.float32)
    coverage = np.zeros(scene_map.shape, dtype=bool)
    with torch.inference_mode():
        for box in boxes:
            box_id = box["box_id"]
            geometry = {"bbox_xyxy": box["bbox_xyxy"], "rotated_corners": box.get("rotated_corners")}
            try:
                crop_box = crop_bounds(box["bbox_xyxy"], after_img.size)
                before_crop = before_img.crop(crop_box)
                crop = after_img.crop(crop_box)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                results.append({**geometry, "box_id": box_id, "status": "failed", "error": str(error)})
                print(f"Box {box_id}: failed crop: {error}")
                continue

            model.embedding_store.clear()
            model.memory_bank = model.memory_bank.new_empty(0)
            try:
                model.train()
                model.feature_encoder.eval()
                model(tensor(before_crop))
                model.fit()
                model.eval()
                result = model(tensor(crop))
                score = float(result.pred_score.item())
                anomaly_map = result.anomaly_map.cpu().numpy().squeeze()
                if not math.isfinite(score) or not np.isfinite(anomaly_map).all():
                    raise ValueError("AnomalyDINO returned non-finite values.")
            except torch.OutOfMemoryError:
                raise
            except (ValueError, RuntimeError) as error:
                results.append({**geometry, "box_id": box_id, "status": "failed", "error": str(error)})
                print(f"Box {box_id}: failed assessment: {error}")
                continue

            map_name = f"building_{box_id}_anomaly.npy"
            np.save(run_dir / map_name, anomaly_map)
            display_mask = project_anomaly(
                scene_map, coverage, anomaly_map, crop_box,
                box.get("rotated_corners") if mask_roofs else None,
            )
            results.append({
                **geometry,
                "box_id": box_id,
                "status": "completed",
                "crop_bbox_xyxy": crop_box,
                "anomaly_score": score,
                "anomaly_map": map_name,
                "display_mask": display_mask,
            })
            print(f"Box {box_id}: raw anomaly score {score:.5f}")

    overlay, visualization = render_heatmap(after_img, scene_map, coverage, results, display_range, run_id=run_id)
    overlay_name = f"after{pair_id}_heatmap.png"
    overlay.save(run_dir / overlay_name)

    report_path = run_dir / "anomalies.json"
    succeeded = sum(box["status"] == "completed" for box in results)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "run_id": run_id,
            "pair_id": pair_id,
            "source_sha256": {"before": before["source_sha256"], "after": after["source_sha256"]},
            "boxed_images": {"before": before["boxed_image"], "after": after["boxed_image"]},
            "status": ("empty" if not boxes else "completed" if succeeded == len(boxes)
                       else "partial" if succeeded else "failed"),
            "num_detections": len(boxes),
            "num_completed": succeeded,
            "num_failed": len(boxes) - succeeded,
            "before_image": before_path.as_posix(),
            "after_image": after_path.as_posix(),
            "stage2_boxes": Path(boxes_path).resolve().as_posix(),
            "encoder": ENCODER,
            "reference_mode": "matching_before_crop",
            "overlay_display_range": visualization["range"],
            "visualization": visualization,
            "roof_mask_requested": mask_roofs,
            "scene_heatmap": overlay_name,
            "overlap_rule": "maximum",
            "boxes": results,
        }, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Matching Stage 2 scene: {after['boxed_image']}")
    print(f"Matching Stage 3 scene: {(run_dir / overlay_name).resolve()}")
    print(f"Stage 3 results saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=Path, help="Immutable Stage 2 boxes.json; defaults to latest successful Stage 2 export")
    parser.add_argument("--output", type=Path, default=ROOT / config.STAGE3_OUTPUT_DIR)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--display-range", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()
    if args.boxes is None:
        pointer = ROOT / config.STAGE2_OUTPUT_DIR / "latest.json"
        if not pointer.exists():
            parser.error("No latest Stage 2 export. Run the pipeline first or provide --boxes.")
        latest = json.loads(pointer.read_text(encoding="utf-8"))
        run_id = validate_run_id(latest.get("run_id"))
        args.boxes = pointer.parent / run_id / "boxes.json"
    run_stage3_anomalydino(args.boxes, args.output, args.device,
                         display_range=args.display_range, mask_roofs=not args.show_context)


if __name__ == "__main__":
    main()
