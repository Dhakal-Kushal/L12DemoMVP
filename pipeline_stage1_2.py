import os
import sys

import torch
from PIL import Image

from pipeline import config
from pipeline.images import find_image
from pipeline.stage1 import run_stage1_clip
from pipeline.stage2 import run_stage2_owlv2


def main():
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
    run_stage2_owlv2(before_img, after_img, before_path, after_path, user_query)


if __name__ == "__main__":
    main()
