"""Assign provisional xBD damage grades to buildings from a Stage 2 export."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn.functional import normalize

from . import config
from .stage2_inputs import ROOT, latest_stage2_export, load_source_images, read_stage2_export


CLASSIFIER_PATH = Path(__file__).with_name("weights") / "xbd_dinov2_after_only.json"
LABELS = ("no-damage", "minor-damage", "major-damage", "destroyed")
DISPLAY_LABELS = ("No visible damage", "Minor damage", "Major damage", "Destroyed")
ENCODER = "vit_small_patch14_dinov2"
IMAGE_SIZE = 252
PATCH_SIZE = 14
FEATURE_DIM = 384
PADDING = 0.15
BATCH_SIZE = 8


def prepare_crop(image, corners):
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Expected four finite footprint corners.")
    x, y = points.T
    if abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) == 0:
        raise ValueError("Building footprint has zero area.")
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    if xmin < 0 or ymin < 0 or xmax > image.width or ymax > image.height:
        raise ValueError("Building footprint lies outside the image.")
    x_pad, y_pad = (xmax - xmin) * PADDING, (ymax - ymin) * PADDING
    left = max(0, int(xmin - x_pad))
    top = max(0, int(ymin - y_pad))
    right = min(image.width, int(xmax + x_pad))
    bottom = min(image.height, int(ymax + y_pad))
    if right <= left or bottom <= top:
        raise ValueError("Building footprint has no crop pixels.")
    crop = image.crop((left, top, right, bottom))
    if all(low == high for low, high in crop.getextrema()):
        raise ValueError("The after-image crop is blank.")

    mask = Image.new("L", crop.size)
    ImageDraw.Draw(mask).polygon([(px - left, py - top) for px, py in points], fill=255)
    grid = IMAGE_SIZE // PATCH_SIZE
    patch_weights = np.array(mask.resize((grid, grid), Image.Resampling.BOX)).ravel()
    if not patch_weights.any():
        raise ValueError("Building footprint has no visible patch weights.")
    return crop, patch_weights, (left, top, right, bottom)


class DamageClassifier:
    def __init__(self, device):
        from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
        from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel

        model_bytes = CLASSIFIER_PATH.read_bytes()
        head = json.loads(model_bytes)
        settings = head["encoder_settings"]
        if (head["schema_version"] != 1 or head["grades"] != list(LABELS) or head["method"] != "after_only"
                or head["pad_fraction"] != PADDING or settings["encoder"] != ENCODER
                or settings["image_size"] != IMAGE_SIZE):
            raise ValueError("Bundled damage classifier is incompatible with the inference pipeline.")
        self.weights = np.asarray(head["head"]["weights"], dtype=np.float64)
        self.intercept = np.asarray(head["head"]["intercept"], dtype=np.float64)
        if (self.weights.shape != (len(LABELS), FEATURE_DIM) or self.intercept.shape != (len(LABELS),)
                or not np.isfinite(self.weights).all() or not np.isfinite(self.intercept).all()):
            raise ValueError("Bundled damage classifier has invalid coefficient or intercept values.")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run with --device cpu.")
        self.model = AnomalyDINOModel(
            num_neighbours=1, encoder_name=ENCODER, masking=False, coreset_subsampling=False
        ).eval()
        encoder_hash = hashlib.sha256()
        for name, tensor in sorted(self.model.feature_encoder.state_dict().items()):
            encoder_hash.update(name.encode())
            encoder_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.encoder_sha256 = encoder_hash.hexdigest()
        if self.encoder_sha256 != head["encoder_sha256"]:
            raise ValueError("Loaded DINOv2 weights differ from those used to train the damage classifier.")
        self.model.to(self.device)
        self.preprocess = AnomalyDINO.configure_pre_processor(image_size=(IMAGE_SIZE, IMAGE_SIZE)).to(self.device)
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()

    @torch.inference_mode()
    def score(self, crops, patch_weights):
        images = []
        for crop in crops:
            rgb = torch.from_numpy(np.array(crop)).permute(2, 0, 1)
            images.append(self.preprocess(rgb.to(self.device).float() / 255.0))
        patches = self.model.extract_features(torch.stack(images))
        expected = (len(crops), (IMAGE_SIZE // PATCH_SIZE)**2, FEATURE_DIM)
        if tuple(patches.shape) != expected:
            raise RuntimeError(f"Expected DINOv2 patch features of shape {expected}, got {tuple(patches.shape)}.")
        weights = torch.from_numpy(np.stack(patch_weights)).to(device=self.device, dtype=patches.dtype)
        weights /= weights.sum(dim=1, keepdim=True)
        pooled = torch.einsum("bpd,bp->bd", normalize(patches, dim=-1), weights)
        vectors = normalize(pooled, dim=-1).cpu().numpy()
        logits = vectors @ self.weights.T + self.intercept
        if not np.isfinite(logits).all():
            raise RuntimeError("Damage classifier returned nonfinite class scores.")
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)


def run_damage_grading(boxes_path, output_dir, device):
    metadata = read_stage2_export(boxes_path)
    run_id, pair_id = metadata["run_id"], metadata["pair_id"]
    output_dir = Path(output_dir)
    run_dir = output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}. Use --output with a different directory to rerun grading.")
    _, (_, after_image) = load_source_images(metadata)
    boxes = metadata["after"]["boxes"]
    print(f"\n--- STAGE 4: DAMAGE GRADING (Pair {pair_id}) ---")
    classifier = None
    buildings = []
    for start in range(0, len(boxes), BATCH_SIZE):
        valid, crops, weights = [], [], []
        for box in boxes[start:start + BATCH_SIZE]:
            record = {
                "box_id": box["box_id"], "bbox_xyxy": box["bbox_xyxy"], "rotated_corners": box.get("rotated_corners"),
                "damage_grade": None, "review_required": True,
            }
            buildings.append(record)
            try:
                crop, patch_weights, bounds = prepare_crop(after_image, record["rotated_corners"])
            except (TypeError, ValueError, OverflowError) as error:
                record.update(status="unable_to_assess", reason=str(error))
                print(f"Box {record['box_id']}: unable to assess ({error})")
                continue
            record["crop_bbox_xyxy"] = bounds
            valid.append(record)
            crops.append(crop)
            weights.append(patch_weights)

        if not valid:
            continue
        if classifier is None:
            classifier = DamageClassifier(device)
        for record, scores in zip(valid, classifier.score(crops, weights), strict=True):
            grade = int(scores.argmax())
            record.update(status="provisional", damage_grade=grade, damage_label=LABELS[grade],
                          class_scores=dict(zip(LABELS, scores.tolist(), strict=True)))
            print(f"Box {record['box_id']}: provisional grade {grade} ({DISPLAY_LABELS[grade]})")

    graded = sum(row["status"] == "provisional" for row in buildings)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "pair_id": pair_id,
        "stage2_boxes": Path(boxes_path).resolve().as_posix(),
        "source_sha256": {key: metadata[key]["source_sha256"] for key in ("before", "after")},
        "method": "after_only",
        "encoder": ENCODER,
        "encoder_sha256": classifier.encoder_sha256 if classifier is not None else None,
        "model_sha256": classifier.model_sha256 if classifier is not None else None,
        "score_type": "uncalibrated_softmax",
        "geometry": "image_pixels",
        "num_graded": graded,
        "num_unassessable": len(buildings) - graded,
        "buildings": buildings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".grading-", dir=output_dir) as scratch:
        staged = Path(scratch) / run_id
        staged.mkdir()
        (staged / "damage_grades.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(run_dir)
    report_path = run_dir / "damage_grades.json"
    print(f"Stage 4 saved {graded}/{len(buildings)} provisional grades: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=Path, help="Stage 2 boxes.json (defaults to the latest Stage 2 export)")
    parser.add_argument("--output", type=Path, default=ROOT / config.STAGE4_OUTPUT_DIR)
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()
    if args.boxes is None:
        try:
            args.boxes = latest_stage2_export(config.STAGE2_OUTPUT_DIR)
        except FileNotFoundError as error:
            parser.error(str(error))
    run_damage_grading(args.boxes, args.output, args.device)


if __name__ == "__main__":
    main()
