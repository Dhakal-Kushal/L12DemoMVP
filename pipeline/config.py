# Configuration
PAIR_ID = 2
IMAGE_DIR = "inputImages"
OUTPUT_DIR = "boxedOutput"
STAGE2_OUTPUT_DIR = "stage2json"
STAGE3_OUTPUT_DIR = "stage3Heatmaps"


NEGATIVE_QUERIES = [
    "dense white clouds obscuring the ground",
    "open ocean or empty water with no structures",
    "empty forest or undeveloped wilderness",
    "a grass sports field"
]

STAGE1_MODEL = "chendelong/RemoteCLIP"
# Minimum logit lead for a rejection. Provisional until calibrated on labeled pairs.
STAGE1_REJECT_MARGIN = 2.0

STAGE2_DETECTOR = "owlv2"  # or "grounding_dino"
STAGE2_OUTLINE_METHOD = "sam2"  # or "otsu" for the baseline
STAGE2_DINO_BOX_THRESHOLD = 0.25
STAGE2_DINO_TEXT_THRESHOLD = 0.25
STAGE2_BOX_THRESHOLD = 0.22
STAGE2_NMS_IOU_THRESHOLD = 0.50
STAGE2_OUTLINE_IOU_THRESHOLD = 0.70
STAGE2_CONTAINMENT_THRESHOLD = 0.95
STAGE2_TILE_SIZE = 512  # Pixels. Set to 0 for full-image detection only.
STAGE2_TILE_OVERLAP = 0.25
DEVICE = "cpu"
