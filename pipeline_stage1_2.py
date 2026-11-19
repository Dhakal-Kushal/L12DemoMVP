import argparse
import os
import sys

import torch
from PIL import Image

from pipeline import config
from pipeline.images import find_image
from pipeline.stage1 import run_stage1_clip
from pipeline.stage2 import run_stage2_owlv2
from pipeline.stage3 import run_stage3_anomalydino
from pipeline.stage4 import run_damage_grading


def main():
    parser = argparse.ArgumentParser(description="Detect buildings and grade visible damage in an aligned image pair.")
    parser.add_argument("--pair", type=int, default=config.PAIR_ID, help="Image pair number (default: %(default)s)")
    parser.add_argument("--skip-anomaly", action="store_true", help="Skip the Stage 3 heatmap and run damage grading")
    args = parser.parse_args()
    if args.pair < 0:
        parser.error("--pair must be nonnegative")
    config.PAIR_ID = args.pair
    for directory in (config.IMAGE_DIR, config.OUTPUT_DIR, config.STAGE2_OUTPUT_DIR):
        os.makedirs(directory, exist_ok=True)
    if not 0.0 <= config.STAGE2_BOX_THRESHOLD <= 1.0:
        raise ValueError("STAGE2_BOX_THRESHOLD must be between 0.0 and 1.0.")
    device = torch.device(config.DEVICE)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Set DEVICE = "cpu" to run locally.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {config.DEVICE}")

    before_path = find_image(config.IMAGE_DIR, "before", config.PAIR_ID)
    after_path = find_image(config.IMAGE_DIR, "after", config.PAIR_ID)
    with Image.open(before_path) as image:
        before_img = image.convert("RGB")
    with Image.open(after_path) as image:
        after_img = image.convert("RGB")
    user_query = input("Enter target query for Stage 1 & 2 (e.g., 'an overhead image of a building'): ").strip()
    if not user_query:
        user_query = "an overhead image of a building"

    if not run_stage1_clip(before_img, after_img, before_path, after_path, user_query):
        print("Stage 1 Verdict: NOT USABLE. Stopping pipeline.")
        sys.exit(0)
    print("Stage 1 Verdict: USABLE. Proceeding to Stage 2.")
    boxes_path = run_stage2_owlv2(before_img, after_img, before_path, after_path, user_query)
    if not args.skip_anomaly:
        run_stage3_anomalydino(boxes_path, config.STAGE3_OUTPUT_DIR, config.DEVICE)
    run_damage_grading(boxes_path, config.STAGE4_OUTPUT_DIR, config.DEVICE)


if __name__ == "__main__":
    main()
