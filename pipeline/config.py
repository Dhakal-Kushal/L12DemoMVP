# Configuration
PAIR_ID = 2
IMAGE_DIR = "inputImages"
OUTPUT_DIR = "boxedOutput"
STAGE2_OUTPUT_DIR = "stage2json"


NEGATIVE_QUERIES = [
    "dense white clouds obscuring the ground",
    "open ocean or empty water with no structures",
    "empty forest or undeveloped wilderness",
    "an asphalt parking lot with cars",
    "a grass sports field"
]

STAGE2_BOX_THRESHOLD = 0.15
DEVICE = "cpu"
