import os
import re


EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def available_pair_ids(image_dir):
    ids = {"before": set(), "after": set()}
    with os.scandir(image_dir) as entries:
        for entry in entries:
            stem, suffix = os.path.splitext(entry.name)
            match = re.fullmatch(r"(before|after)(0|[1-9][0-9]*)", stem)
            if match and suffix.lower() in EXTENSIONS and entry.is_file():
                ids[match[1]].add(int(match[2]))
    return sorted(ids["before"] & ids["after"])


def find_image(image_dir, prefix, pair_id):
    stem = f"{prefix}{pair_id}"
    if os.path.isdir(image_dir):
        filenames = sorted(os.listdir(image_dir))
        for extension in EXTENSIONS:
            for filename in filenames:
                name, suffix = os.path.splitext(filename)
                path = os.path.join(image_dir, filename)
                if name == stem and suffix.lower() == extension and os.path.isfile(path):
                    return path
    raise FileNotFoundError(
        f"Missing image '{stem}' in '{image_dir}'. "
        f"Supported extensions: {', '.join(EXTENSIONS)}"
    )
