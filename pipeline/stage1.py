import os

import torch
from transformers import CLIPModel, CLIPProcessor

from . import config


def run_stage1_clip(before_img, after_img, before_path, after_path, user_query):
    print(f"\n--- STAGE 1: CLIP SCENE VALIDATION (Pair {config.PAIR_ID}) ---")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(config.DEVICE)
    model.eval()
    candidate_queries = [user_query] + config.NEGATIVE_QUERIES
    inputs = processor(
        text=candidate_queries,
        images=[before_img, after_img],
        return_tensors="pt",
        padding=True,
    ).to(config.DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        percentages = (outputs.logits_per_image.softmax(dim=-1) * 100.0).cpu()
    if not torch.isfinite(percentages).all().item():
        raise RuntimeError("CLIP returned non-finite scene scores.")

    usable = []
    for path, scores in zip((before_path, after_path), percentages):
        is_usable = scores.argmax().item() == 0
        usable.append(is_usable)
        print(f"[{os.path.basename(path)}]")
        for index, (query, score) in enumerate(zip(candidate_queries, scores)):
            suffix = " (User Query)" if index == 0 else ""
            print(f'  * "{query}"{suffix}: {score.item():.2f}%')
        print(f"  -> Result: {'USABLE' if is_usable else 'NOT USABLE'}\n")

    return all(usable)
