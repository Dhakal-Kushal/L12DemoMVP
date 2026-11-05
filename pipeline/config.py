# Configuration
PAIR_ID = 1
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

STAGE2_BOX_THRESHOLD = 0.15
DEVICE = "cpu"
