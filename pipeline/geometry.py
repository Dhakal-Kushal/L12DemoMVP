import cv2
import numpy as np


def box_iou(first, second):
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def nms_and_pad_boxes(raw_boxes, scores, img_w, img_h, *, iou_threshold, pad_pct=0.10):
    """Return (source index, padded box) pairs in descending confidence order."""
    if len(raw_boxes) != len(scores):
        raise ValueError("Each detection must have one score.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be between 0 and 1.")

    retained = []
    # Stable sorting gives the earlier detection priority when scores tie.
    for index in sorted(range(len(scores)), key=scores.__getitem__, reverse=True):
        box = raw_boxes[index]
        width, height = box[2] - box[0], box[3] - box[1]
        if not (25 < width < img_w * 0.35 and 25 < height < img_h * 0.35):
            continue
        if any(box_iou(box, raw_boxes[other]) > iou_threshold for other, _ in retained):
            continue
        pad_x, pad_y = width * pad_pct, height * pad_pct
        padded = [
            max(0.0, box[0] - pad_x), max(0.0, box[1] - pad_y),
            min(img_w, box[2] + pad_x), min(img_h, box[3] + pad_y),
        ]
        retained.append((index, padded))
    return retained


def suppress_duplicate_outlines(boxes, iou_threshold):
    """Keep the highest-scoring fitted polygons, using their actual intersection area."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("Outline IoU threshold must be between 0 and 1.")
    retained = []
    polygons = []
    for box in sorted(boxes, key=lambda box: box["score"], reverse=True):
        polygon = np.asarray(box["rotated_corners"], dtype=np.float32)
        area = cv2.contourArea(polygon)
        for other, other_area in polygons:
            intersection, _ = cv2.intersectConvexConvex(polygon, other, handleNested=True)
            intersection = min(intersection, area, other_area)
            union = area + other_area - intersection
            if intersection > iou_threshold * union:
                break
        else:
            retained.append(box)
            polygons.append((polygon, area))
    return retained


def suppress_nested_outlines(boxes, containment_threshold):
    """Prefer separate roofs inside a group box, then resolve nested pairs by score."""
    if not 0 < containment_threshold <= 1:
        raise ValueError("Containment threshold must be in (0, 1].")
    ranked = sorted(boxes, key=lambda box: box["score"], reverse=True)
    polygons = [np.asarray(box["rotated_corners"], dtype=np.float32) for box in ranked]
    areas = [cv2.contourArea(polygon) for polygon in polygons]
    children = [set() for _ in ranked]
    conflicts = [set() for _ in ranked]
    overlapping = [set() for _ in ranked]
    for i, polygon in enumerate(polygons):
        for j in range(i):
            smaller = min(areas[i], areas[j])
            if smaller <= 0:
                continue
            intersection, _ = cv2.intersectConvexConvex(polygon, polygons[j], handleNested=True)
            coverage = intersection / smaller
            # Allow minor edge overlap between neighboring fitted rectangles.
            if coverage > 0.10:
                overlapping[i].add(j)
                overlapping[j].add(i)
            if coverage >= containment_threshold:
                outer, inner = (i, j) if areas[i] > areas[j] else (j, i)
                children[outer].add(inner)
                conflicts[i].add(j)
                conflicts[j].add(i)

    kept = set()
    for i, inside in enumerate(children):
        # Two distinct enclosed outlines are evidence that this box spans roofs.
        if any(inside - overlapping[child] - {child} for child in inside):
            continue
        if kept.isdisjoint(conflicts[i]):
            kept.add(i)
    return [box for i, box in enumerate(ranked) if i in kept]


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
