import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def project_anomaly(scene_map, coverage, anomaly_map, bounds, corners=None):
    xmin, ymin, xmax, ymax = bounds
    size = (xmax - xmin, ymax - ymin)
    projected = np.asarray(Image.fromarray(anomaly_map.astype(np.float32)).resize(
        size, Image.Resampling.BILINEAR
    ))
    mask = np.ones(projected.shape, dtype=bool)
    mask_type = "crop_rectangle"
    if corners is not None:
        try:
            points = np.asarray(corners, dtype=float)
        except (TypeError, ValueError):
            points = np.empty((0, 2))
        if points.shape == (4, 2) and np.isfinite(points).all():
            x, y = points[:, 0], points[:, 1]
            area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
            if area > 1:
                polygon = Image.new("L", size)
                ImageDraw.Draw(polygon).polygon(
                    [(float(x - xmin), float(y - ymin)) for x, y in points], fill=255
                )
                candidate = np.asarray(polygon) > 0
                if candidate.any():
                    mask = candidate
                    mask_type = "roof_polygon"
    region = scene_map[ymin:ymax, xmin:xmax]
    np.maximum(region, np.where(mask, projected, 0), out=region)
    coverage[ymin:ymax, xmin:xmax] |= mask
    return mask_type


def render_heatmap(image, scene_map, coverage, buildings, display_range=None, *, run_id=None):
    values = scene_map[coverage]
    mode = "fixed" if display_range is not None else "scene_relative"
    if display_range is not None:
        low, high = map(float, display_range)
        if not np.isfinite([low, high]).all() or not 0 <= low < high:
            raise ValueError("Display range must contain finite values with 0 <= low < high.")
    elif values.size:
        low, high = map(float, np.percentile(values, [30, 98]))
        if high - low < 1e-6:
            low, high = 0.0, max(1.0, high)
            mode = "flat_scene_fallback"
    else:
        low, high = 0.0, 1.0
        mode = "no_assessments"
    scaled = np.clip((scene_map - low) / (high - low), 0, 1)
    heat = ImageOps.colorize(
        Image.fromarray((scaled * 255).astype(np.uint8)),
        black="#ffe34d", mid="#ff8c1a", white="#ef2020",
    )
    alpha = Image.fromarray((coverage * scaled ** 1.4 * 0.8 * 255).astype(np.uint8))
    overlay = Image.composite(heat, image, alpha)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(10, min(18, image.width // 65)))
    occupied = []
    if run_id is not None:
        text = f"Run: {run_id}"
        run_font = ImageFont.load_default(size=10)
        bounds = (0, 0, min(image.width, int(draw.textlength(text, font=run_font)) + 8), 17)
        draw.rectangle(bounds, fill="#171b24")
        draw.text((4, 2), text, fill="white", font=run_font)
        occupied.append(bounds)
    placements = []
    for building in buildings:
        if building["status"] != "completed":
            continue
        text = f"B{building['box_id']:02d} | Anomaly: {building['anomaly_score']:.3f}"
        label_font = font
        while draw.textlength(text, font=label_font) + 10 > image.width and label_font.size > 6:
            label_font = ImageFont.load_default(size=label_font.size - 1)
        width = min(image.width, int(draw.textlength(text, font=label_font)) + 10)
        height = min(image.height, label_font.size + 10)
        x1, y1, x2, y2 = building["crop_bbox_xyxy"]
        candidates = [(x1, y1 - height - 3), (x1, y2 + 3),
                      (x2 + 3, y1), (x1 - width - 3, y1), (x1, y1)]
        candidates += [(x1, y) for y in range(0, image.height, max(1, height + 3))]
        rects = [(max(0, min(x, image.width - width)),
                  max(0, min(y, image.height - height))) for x, y in candidates]
        def placement_cost(position):
            x, y = position
            overlap = sum(max(0, min(x + width, r[2]) - max(x, r[0])) *
                          max(0, min(y + height, r[3]) - max(y, r[1])) for r in occupied)
            return overlap, abs(x - x1) + abs(y - y1)
        x, y = min(rects, key=placement_cost)
        rect = (x, y, x + width, y + height)
        anchor = ((x1 + x2) / 2, (y1 + y2) / 2)
        draw.line([anchor, (x + width / 2, y + height / 2)], fill="white", width=1)
        draw.rectangle(rect, fill="#171b24", outline="#d9dde5")
        draw.text((x + 5, y + 3), text, fill="white", font=label_font)
        occupied.append(rect)
        placements.append({"box_id": building["box_id"], "label_bbox_xyxy": list(rect)})
    return overlay, {
        "mode": mode, "range": [low, high], "percentiles": [30, 98] if mode == "scene_relative" else None,
        "max_opacity": 0.8, "opacity_gamma": 1.4, "palette": "yellow-orange-red",
        "labels": placements,
    }
