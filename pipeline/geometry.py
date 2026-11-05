import cv2
import numpy as np


def clean_merge_and_pad_boxes(
    raw_boxes, img_w, img_h, overlap_thresh=0.20, pad_pct=0.10,
    *, return_groups=False,
):
    candidates = []
    for index, b in enumerate(raw_boxes):
        w, h = b[2] - b[0], b[3] - b[1]
        if 25 < w < img_w * 0.35 and 25 < h < img_h * 0.35:
            candidates.append((list(b), [index]))

    # Recheck earlier boxes after each union to merge overlapping chains.
    merged = []
    while candidates:
        curr, members = candidates.pop(0)
        index = 0
        while index < len(merged):
            other, other_members = merged[index]
            ix1, iy1 = max(curr[0], other[0]), max(curr[1], other[1])
            ix2, iy2 = min(curr[2], other[2]), min(curr[3], other[3])
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                min_area = min(
                    (curr[2] - curr[0]) * (curr[3] - curr[1]),
                    (other[2] - other[0]) * (other[3] - other[1]),
                )
                if min_area > 0 and inter_area / min_area > overlap_thresh:
                    curr = [
                        min(curr[0], other[0]), min(curr[1], other[1]),
                        max(curr[2], other[2]), max(curr[3], other[3]),
                    ]
                    members.extend(other_members)
                    merged.pop(index)

                    index = 0
                    continue
            index += 1
        merged.append((curr, members))

    final_boxes, groups = [], []
    for b, members in merged:
        w, h = b[2] - b[0], b[3] - b[1]

        # Merged chains must also satisfy the maximum footprint size.
        if not (25 < w < img_w * 0.35 and 25 < h < img_h * 0.35):
            continue
        pw, ph = w * pad_pct, h * pad_pct
        final_boxes.append([
            float(max(0, b[0] - pw)), float(max(0, b[1] - ph)),
            float(min(img_w, b[2] + pw)), float(min(img_h, b[3] + ph)),
        ])
        groups.append(members)
    return (final_boxes, groups) if return_groups else final_boxes


def compute_rotated_box_corners(image_np, aabb):
    h_img, w_img = image_np.shape[:2]
    bounds = np.asarray(aabb, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("Expected four finite XYXY coordinates for the roof crop.")
    bounds[[0, 2]] = np.clip(bounds[[0, 2]], 0, w_img)
    bounds[[1, 3]] = np.clip(bounds[[1, 3]], 0, h_img)
    x1, y1, x2, y2 = bounds
    fallback = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if x2 - x1 < 15 or y2 - y1 < 15:
        return fallback


    xmin, ymin = int(np.floor(x1)), int(np.floor(y1))
    xmax, ymax = int(np.ceil(x2)), int(np.ceil(y2))
    crop = image_np[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Avoid fitting a roof rectangle to nearly uniform sensor noise.
    if np.ptp(blurred) < 5:
        return fallback
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 50:
            rect = cv2.minAreaRect(contour)
            box_pts = cv2.boxPoints(rect)
            box_pts[:, 0] += xmin
            box_pts[:, 1] += ymin


            if (np.isfinite(box_pts).all()
                    and (box_pts[:, 0] >= 0).all() and (box_pts[:, 0] <= w_img).all()
                    and (box_pts[:, 1] >= 0).all() and (box_pts[:, 1] <= h_img).all()):
                return box_pts
    return fallback
