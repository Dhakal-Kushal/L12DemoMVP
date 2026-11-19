"""Fit roof rectangles to box-prompted SAM 2 masks in original image coordinates."""

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

SAM2_CHECKPOINT = "facebook/sam2.1-hiera-tiny"


def mask_outline(mask, prompt):
    x1, y1, x2, y2 = prompt
    fallback = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback, "empty_mask"
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return fallback, "degenerate_mask"
    corners = cv2.boxPoints(cv2.minAreaRect(contour))
    prompt_area = (x2 - x1) * (y2 - y1)
    fitted_area = cv2.contourArea(corners)
    intersection, _ = cv2.intersectConvexConvex(corners, np.asarray(fallback, np.float32))
    # Require a footprint substantially inside the prompt, and reject tiny roof details.
    if intersection < max(0.5 * fitted_area, 0.1 * prompt_area):
        return fallback, "poor_prompt_overlap"
    height, width = mask.shape
    if (corners < -1e-4).any() or (corners > [width + 1e-4, height + 1e-4]).any():
        return fallback, "out_of_bounds"
    return np.clip(corners, [0, 0], [width, height]), None


@torch.inference_mode()
def segment_outlines(image, prompts, device):
    if not prompts:
        return []
    processor = Sam2Processor.from_pretrained(SAM2_CHECKPOINT)
    model = Sam2Model.from_pretrained(SAM2_CHECKPOINT).to(device).eval()
    pixels = processor(images=image, return_tensors="pt").pixel_values.to(device)
    embeddings = model.get_image_embeddings(pixels)
    sizes = [[image.height, image.width]]
    outlines = []
    # Bound decoder and full-resolution mask memory independently of the roof count.
    for start in range(0, len(prompts), 4):
        batch = prompts[start:start + 4]
        inputs = processor(input_boxes=[batch], original_sizes=sizes, return_tensors="pt")
        output = model(image_embeddings=embeddings, input_boxes=inputs.input_boxes.to(device),
                       multimask_output=False)
        if not torch.isfinite(output.pred_masks).all():
            raise RuntimeError("SAM 2 returned non-finite mask logits.")
        masks = processor.post_process_masks(output.pred_masks.cpu(), sizes)[0]
        for prompt, mask in zip(batch, masks, strict=True):
            outlines.append(mask_outline(mask.squeeze(0).numpy(), prompt))
    return outlines
