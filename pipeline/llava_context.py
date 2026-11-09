"""Chat with LLaVA (served by Ollama) about the Stage 3 results.

Run from the repo root after Stages 2 and 3, with Ollama running and the
model pulled (ollama pull llava):
    python -m pipeline.llava_context
    python -m pipeline.llava_context -q "Which building looks most damaged?" -q "What changed at building 9?"
    python -m pipeline.llava_context --dry-run

Without -q, questions are read interactively until an empty line. With
--dry-run, the images and prompts are written without calling the model.

A general question gets the after image with every building outlined and
labeled by ID. A question naming one building ("building 9") gets that
building's before and after crops side by side. Scores stay in the prompt
text because LLaVA downscales images until small drawn decimals are
unreadable. Images are padded to a square so a center crop loses nothing.
"""

import argparse
import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import config


ROOT = Path(__file__).resolve().parents[1]
OUTLINE = "#ffe600"
PANEL_SIZE = 320
PANEL_GAP = 16
BUILDING_PATTERN = re.compile(r"\b(?:building|id|box)\s*#?\s*(\d+)\b", re.IGNORECASE)
SCORE_NOTE = (
    "Anomaly score = how much the building changed compared to the before image "
    "(higher = more change). Scores are uncalibrated: use them to compare buildings, "
    "not as a damage percentage. Shadows, lighting, and slight misalignment can also "
    "raise a score."
)


def latest_anomalies(stage3_dir):
    reports = sorted(Path(stage3_dir).glob("run-*/anomalies.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(f"No Stage 3 anomalies.json found under '{stage3_dir}'.")
    return reports[-1]


def load_buildings(boxes_path, anomalies_path):
    with open(boxes_path, encoding="utf-8") as handle:
        stage2 = json.load(handle)
    with open(anomalies_path, encoding="utf-8") as handle:
        stage3 = json.load(handle)
    if stage2["pair_id"] != stage3["pair_id"]:
        raise ValueError(
            f"Stage 2 is pair {stage2['pair_id']} but Stage 3 is pair {stage3['pair_id']}; "
            "rerun Stage 3 on the current boxes.json."
        )
    scores = {box["box_id"]: box for box in stage3["boxes"]}
    boxes = stage2["after"]["boxes"]
    if set(scores) != {box["box_id"] for box in boxes}:
        raise ValueError("Stage 2 and Stage 3 box IDs differ; rerun Stage 3 on the current boxes.json.")
    ranked = sorted(boxes, key=lambda box: scores[box["box_id"]]["anomaly_score"], reverse=True)
    buildings = [{**box, **scores[box["box_id"]], "rank": rank} for rank, box in enumerate(ranked, 1)]
    return stage2, buildings


def position_hint(bbox, width, height):
    cx = (bbox[0] + bbox[2]) / 2 / width
    cy = (bbox[1] + bbox[3]) / 2 / height
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    return "center" if (row, col) == ("middle", "center") else f"{row} {col}"


def pad_square(image):
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), "black")
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def build_overview(after_img, buildings):
    image = after_img.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(24, round(0.045 * min(image.size))))
    pad = 4
    for building in buildings:
        draw.polygon([tuple(point) for point in building["rotated_corners"]], outline=OUTLINE, width=3)
    for building in buildings:
        text = str(building["box_id"])
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        width, height = right - left + 2 * pad, bottom - top + 2 * pad
        xmin, ymin, xmax, _ = building["bbox_xyxy"]
        x = min(max(0, (xmin + xmax) / 2 - width / 2), image.width - width)
        y = ymin - height if ymin >= height else ymin
        draw.rectangle((x, y, x + width, y + height), fill="black", outline=OUTLINE, width=2)
        draw.text((x + pad - left, y + pad - top), text, fill="white", font=font)
    return pad_square(image)


def build_closeup(before_img, after_img, building):
    x0, y0, x1, y1 = building["crop_bbox_xyxy"]
    side = PANEL_SIZE * 2 + PANEL_GAP
    canvas = Image.new("RGB", (side, side), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=40)
    top = (side - PANEL_SIZE) // 2
    for index, (label, source) in enumerate((("BEFORE", before_img), ("AFTER", after_img))):
        crop = source.crop((x0, y0, x1, y1))
        panel = ImageOps.pad(crop, (PANEL_SIZE, PANEL_SIZE), color="black")
        scale = min(PANEL_SIZE / crop.width, PANEL_SIZE / crop.height)
        offset_x = (PANEL_SIZE - crop.width * scale) / 2
        offset_y = (PANEL_SIZE - crop.height * scale) / 2
        ImageDraw.Draw(panel).polygon(
            [((x - x0) * scale + offset_x, (y - y0) * scale + offset_y) for x, y in building["rotated_corners"]],
            outline=OUTLINE, width=2,
        )
        left = index * (PANEL_SIZE + PANEL_GAP)
        canvas.paste(panel, (left, top))
        draw.text((left + PANEL_SIZE / 2, top - 16), label, fill="white", font=font, anchor="md")
    return canvas


def overview_context(buildings, width, height):
    total = len(buildings)
    lines = [
        "You are helping assess building damage after a disaster from satellite imagery.",
        "The image is the after-disaster satellite view. Each detected building is outlined "
        "in yellow and labeled with its ID number in a black box.",
        SCORE_NOTE,
        "",
        f"Buildings, highest score first ({total} total):",
    ]
    lines += [
        f"- ID {b['box_id']} ({position_hint(b['bbox_xyxy'], width, height)}): "
        f"score {b['anomaly_score']:.3f}, rank {b['rank']} of {total}"
        for b in buildings
    ]
    lines += ["", "Refer to buildings by ID. If the image does not clearly show damage, say so."]
    return "\n".join(lines)


def closeup_context(building, buildings, width, height):
    scores = [b["anomaly_score"] for b in buildings]
    return "\n".join([
        "You are helping assess building damage after a disaster from satellite imagery.",
        f"The image shows building {building['box_id']} only: before the disaster on the left, "
        "after on the right. The building's footprint is outlined in yellow in both halves.",
        f"Building {building['box_id']} is in the {position_hint(building['bbox_xyxy'], width, height)} "
        f"of the scene. Anomaly score {building['anomaly_score']:.3f}, rank {building['rank']} "
        f"of {len(buildings)} (scores in this scene range from {min(scores):.3f} to {max(scores):.3f}).",
        SCORE_NOTE,
        "",
        "Compare the two halves. If the after half does not clearly show damage, say so.",
    ])


def chat_messages(context, question, image_path, history=()):
    # Earlier turns stay text-only; only the current question carries an image.
    messages = [{"role": "system", "content": context}]
    for asked, answered in history:
        messages += [{"role": "user", "content": asked}, {"role": "assistant", "content": answered}]
    messages.append({"role": "user", "content": question, "images": [image_path]})
    return messages


def render_messages(messages):
    blocks = []
    for message in messages:
        header = f"[{message['role']}]"
        if "images" in message:
            header += f" (image: {', '.join(Path(path).name for path in message['images'])})"
        blocks.append(f"{header}\n{message['content']}")
    return "\n\n".join(blocks)


def ask_ollama(messages, model, url):
    payload = [
        {**message, "images": [base64.b64encode(Path(path).read_bytes()).decode("ascii") for path in message["images"]]}
        if "images" in message else message
        for message in messages
    ]
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps({
            "model": model, "messages": payload, "stream": True, "options": {"temperature": 0},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        # The first request also loads the model, which can take a while.
        response = urllib.request.urlopen(request, timeout=600)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body)["error"]
        except (ValueError, KeyError):
            detail = body or error.reason
        hint = f" Run: ollama pull {model}" if error.code == 404 else ""
        raise RuntimeError(f"Ollama error: {detail}.{hint}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {url} ({error.reason}). Start the Ollama app or run: ollama serve"
        ) from None
    chunks = []
    with response:
        for line in response:
            part = json.loads(line)
            if "error" in part:
                raise RuntimeError(f"Ollama error: {part['error']}")
            text = part.get("message", {}).get("content", "")
            print(text, end="", flush=True)
            chunks.append(text)
    print()
    return "".join(chunks).strip()


def read_questions():
    print("Ask about the buildings, e.g. 'Which building looks most damaged?' or "
          "'What changed at building 9?'. Press Enter on an empty line to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            return
        yield question


def focus_building(question, buildings):
    match = BUILDING_PATTERN.search(question)
    if match is None:
        return None
    box_id = int(match.group(1))
    for building in buildings:
        if building["box_id"] == box_id:
            return building
    valid = ", ".join(str(b["box_id"]) for b in sorted(buildings, key=lambda b: b["box_id"]))
    raise ValueError(f"Question names building {box_id}, but the valid IDs are {valid}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boxes", type=Path, default=ROOT / config.STAGE2_OUTPUT_DIR / "boxes.json")
    parser.add_argument("--anomalies", type=Path, default=None,
                        help="Stage 3 anomalies.json; defaults to the newest run in stage3outputs.")
    parser.add_argument("-q", "--question", action="append",
                        help="Chat question; repeat for a multi-turn chat. Omit to chat interactively.")
    parser.add_argument("--output", type=Path, default=ROOT / "llavaContext")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the images and prompts without calling the model.")
    parser.add_argument("--model", default=config.LLAVA_MODEL)
    parser.add_argument("--ollama-url", default=config.OLLAMA_URL)
    args = parser.parse_args()

    anomalies_path = args.anomalies or latest_anomalies(ROOT / "stage3outputs")
    stage2, buildings = load_buildings(args.boxes, anomalies_path)
    after = stage2["after"]
    width, height = after["width"], after["height"]
    with Image.open(ROOT / stage2["before"]["source_image"]) as image:
        before_img = image.convert("RGB")
    with Image.open(ROOT / after["source_image"]) as image:
        after_img = image.convert("RGB")

    interactive = not args.question and not args.dry_run
    if args.question:
        questions = args.question
    elif args.dry_run:
        questions = [
            "Which building looks most damaged?",
            f"What changed at building {buildings[0]['box_id']}?",
        ]
    else:
        questions = read_questions()
    output_dir = args.output / f"pair{stage2['pair_id']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {args.boxes} and {anomalies_path}")
    if not args.dry_run:
        print(f"Model: {args.model} via Ollama at {args.ollama_url}")

    history = []
    overview_path = output_dir / "overview.png"
    build_overview(after_img, buildings).save(overview_path)
    for question in questions:
        turn = len(history) + 1
        try:
            building = focus_building(question, buildings)
        except ValueError as error:
            if not interactive:
                raise
            print(error)
            continue
        if building is None:
            image_path = overview_path
            context = overview_context(buildings, width, height)
        else:
            image_path = output_dir / f"building_{building['box_id']}.png"
            build_closeup(before_img, after_img, building).save(image_path)
            context = closeup_context(building, buildings, width, height)
        messages = chat_messages(context, question, image_path, history)
        prompt_path = output_dir / f"turn_{turn}_prompt.txt"
        prompt_path.write_text(render_messages(messages) + "\n", encoding="utf-8")
        if args.dry_run:
            print(f"\n=== Turn {turn}: prompt {prompt_path.name} ===\n{render_messages(messages)}")
            history.append((question, f"<LLaVA's answer to turn {turn}>"))
            continue
        if not interactive:
            print(f"\nYou: {question}")
        print(f"[Turn {turn}: image {image_path.name}, prompt {prompt_path.name}]")
        print("LLaVA: ", end="", flush=True)
        try:
            answer = ask_ollama(messages, args.model, args.ollama_url)
        except RuntimeError as error:
            print()
            raise SystemExit(error) from None
        (output_dir / f"turn_{turn}_answer.txt").write_text(answer + "\n", encoding="utf-8")
        history.append((question, answer))


if __name__ == "__main__":
    main()
