"""Scene relevance and visibility screening for a before/after pair."""

import math
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from . import config

STANDARD = "openai/clip-vit-base-patch32"
REMOTE = "chendelong/RemoteCLIP"
OBSCURED = "dense white clouds obscuring the ground"
VISIBLE = (
    "a clear overhead image showing visible ground features",
    "a clear overhead image of damaged buildings, rubble and debris",
)


class SceneScorer:
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint
        self.device = device
        if checkpoint == STANDARD:
            self.processor = CLIPProcessor.from_pretrained(checkpoint)
            self.model = CLIPModel.from_pretrained(checkpoint)
        elif checkpoint == REMOTE:
            import open_clip
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(checkpoint, "RemoteCLIP-ViT-B-32.pt",
                                      revision="bf1d8a3ccf2ddbf7c875705e46373bfe542bce38")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=weights)
            self.tokenize = open_clip.get_tokenizer("ViT-B-32")
        else:
            raise ValueError(f"Unsupported Stage 1 checkpoint: {checkpoint}")
        self.model.to(device).eval()

    @torch.inference_mode()
    def scores(self, images, prompts):
        if self.checkpoint == STANDARD:
            inputs = self.processor(images=images, text=prompts, padding=True,
                                    truncation=True, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits_per_image
        else:
            pixels = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            image = self.model.encode_image(pixels, normalize=True)
            text = self.model.encode_text(self.tokenize(prompts).to(self.device), normalize=True)
            logits = self.model.logit_scale.exp() * (image @ text.T)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{self.checkpoint} returned non-finite scene scores.")
        return logits.cpu()


def assess_pair(scorer, images, query, margin):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("STAGE1_REJECT_MARGIN must be finite and positive.")
    negatives = config.NEGATIVE_QUERIES
    prompts = list(dict.fromkeys([query, *negatives, OBSCURED, *VISIBLE]))
    logits = scorer.scores(images, prompts)
    rows = [dict(zip(prompts, row.tolist())) for row in logits]
    decisions = []
    for phase, scores in zip(("before", "after"), rows):
        reasons = []
        visible = max(scores[p] for p in VISIBLE)
        cloud_gap = scores[OBSCURED] - visible
        if cloud_gap >= margin:
            reasons.append((False, f"ground appears obscured (cloud lead {cloud_gap:.2f})"))
        elif cloud_gap >= 0:
            reasons.append((True, f"visibility uncertain (cloud lead {cloud_gap:.2f})"))
        else:
            reasons.append((True, f"ground appears visible (visibility lead {-cloud_gap:.2f})"))
        if phase == "before":
            alternatives = [p for p in negatives if p not in (OBSCURED, query)]
            if alternatives:
                rival = max(alternatives, key=scores.get)
                gap = scores[rival] - scores[query]
                if gap >= margin:
                    reasons.append((False, f"target unlikely: '{rival}' leads by {gap:.2f}"))
                elif gap >= 0:
                    reasons.append((True, f"target relevance uncertain (alternative leads by {gap:.2f})"))
                else:
                    reasons.append((True, f"target favored (target lead {-gap:.2f})"))
        decisions.append((all(ok for ok, _ in reasons), "; ".join(msg for _, msg in reasons)))
    return decisions


def run_stage1_clip(before_img, after_img, before_path, after_path, user_query):
    scorer = SceneScorer(config.STAGE1_MODEL, config.DEVICE)
    print(f"Stage 1 pair {config.PAIR_ID}: {scorer.checkpoint}, reject margin {config.STAGE1_REJECT_MARGIN}")
    decisions = assess_pair(scorer, [before_img, after_img], user_query, config.STAGE1_REJECT_MARGIN)
    for phase, path, (ok, reason) in zip(("before", "after"), (before_path, after_path), decisions):
        print(f"[{phase} {Path(path).name}] {'PASS' if ok else 'REJECT'}: {reason}")
    return all(ok for ok, _ in decisions)
