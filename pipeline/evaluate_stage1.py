"""Evaluate Stage 1 decisions on labeled pairs or smoke-test the sample images."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

from PIL import Image

from . import config
from .images import find_image
from .stage1 import REMOTE, STANDARD, SceneScorer, assess_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("development", "heldout"), default="heldout")
    parser.add_argument("--compare-models", "--remote", action="store_true",
                        help="Compare standard CLIP and RemoteCLIP. Defaults to the configured model.")
    parser.add_argument("--margin", type=float, default=config.STAGE1_REJECT_MARGIN)
    args = parser.parse_args()
    if args.manifest:
        with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["split"] == args.split]
        if not rows:
            parser.error(f"No rows for split {args.split}.")
        root = args.manifest.parent
        for row in rows:
            if row["usable"] not in ("0", "1") or row["damaged"] not in ("0", "1"):
                parser.error("usable and damaged must be 0 or 1.")
    else:
        root = Path(".")
        rows = [dict(before=find_image(config.IMAGE_DIR, "before", i),
                     after=find_image(config.IMAGE_DIR, "after", i),
                     query="satellite image of rooftop") for i in (1, 2, 3)]
    checkpoints = [STANDARD, REMOTE] if args.compare_models else [config.STAGE1_MODEL]
    for checkpoint in checkpoints:
        scorer = SceneScorer(checkpoint, config.DEVICE)
        for variant in (["legacy", "revised"] if checkpoint == STANDARD else ["revised"]):
            rejected_damaged = rejected_usable = admitted_unusable = 0
            elapsed = 0.0
            for row in rows:
                images = []
                for key in ("before", "after"):
                    with Image.open(root / row[key]) as image:
                        images.append(image.convert("RGB"))
                start = perf_counter()
                if variant == "legacy":
                    logits = scorer.scores(images, [row["query"], *config.NEGATIVE_QUERIES])
                    passed = bool((logits.argmax(dim=1) == 0).all())
                    detail = ""
                else:
                    decisions = assess_pair(scorer, images, row["query"], args.margin)
                    passed = all(ok for ok, _ in decisions)
                    detail = " | ".join(reason for _, reason in decisions)
                elapsed += perf_counter() - start
                if args.manifest:
                    usable = row["usable"] == "1"
                    rejected_usable += usable and not passed
                    rejected_damaged += usable and not passed and row["damaged"] == "1"
                    admitted_unusable += not usable and passed
                    for phase in ("before", "after"):
                        fraction = row.get(f"{phase}_cloud_fraction")
                        if fraction:
                            cover = float(fraction)
                            if not 0 <= cover <= 1:
                                raise ValueError("Measured cloud fractions must be in [0, 1].")
                            if cover >= 0.20:
                                detail += f" | {phase} measured cloud cover {cover:.0%} (warning)"
                print(f"{checkpoint} {variant} {row['before']} {'PASS' if passed else 'REJECT'} {detail}")
            print(f"{len(rows)} pairs, {elapsed / len(rows):.3f}s/pair on {config.DEVICE} (excludes loading)")
            if args.manifest:
                print(f"usable rejected={rejected_usable}, usable damaged rejected={rejected_damaged}, "
                      f"unusable admitted={admitted_unusable}")


if __name__ == "__main__":
    main()
