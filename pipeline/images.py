import os


def find_image(image_dir, prefix, pair_id):
    stem = f"{prefix}{pair_id}"
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if os.path.isdir(image_dir):
        filenames = sorted(os.listdir(image_dir))
        for extension in extensions:
            for filename in filenames:
                name, suffix = os.path.splitext(filename)
                path = os.path.join(image_dir, filename)
                if name == stem and suffix.lower() == extension and os.path.isfile(path):
                    return path
    raise FileNotFoundError(
        f"Missing image '{stem}' in '{image_dir}'. "
        f"Supported extensions: {', '.join(extensions)}"
    )
