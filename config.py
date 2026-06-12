from pathlib import Path

# --- Data paths ---
# HPC: DATA_ROOT = Path("/gpfs/mariana/smbhome/totahv/zod_temp")
DATA_ROOT = Path("/run/media/tom/ml/zod_temp")
CAMERA_DIR = DATA_ROOT / "camera"
LIDAR_DIR = DATA_ROOT / "lidar_png"
ANNOTATION_SAM_DIR = DATA_ROOT / "annotation_sam"
FRAMES_FILE = Path(__file__).parent / "frames" / "bad_frames.csv"

# Original 4K ZOD images — used instead of 768px camera/ when available
# Set to None to always use the downscaled camera/ images
# HPC: ZOD_DATA_ROOT = Path("/gpfs/mariana/smbhome/totahv/zod-data/single_frames")
ZOD_DATA_ROOT = Path("/run/media/tom/ml/zod-data/single_frames")

# All pipeline outputs live under one folder
OUTPUT_ROOT = DATA_ROOT / "vlm"
ANNOTATION_OUT_DIR = OUTPUT_ROOT / "annotation"      # refined masks (uint8 PNG)
RESULTS_DIR = OUTPUT_ROOT / "results"                # per-frame JSON + summary
VIS_DIR = OUTPUT_ROOT / "visualization"              # triage overlay images


def use_hpc():
    """Switch all data paths to HPC. Call before any I/O."""
    global DATA_ROOT, CAMERA_DIR, LIDAR_DIR, ANNOTATION_SAM_DIR
    global ZOD_DATA_ROOT, OUTPUT_ROOT, ANNOTATION_OUT_DIR, RESULTS_DIR, VIS_DIR
    DATA_ROOT = Path("/gpfs/mariana/smbhome/totahv/zod_temp")
    CAMERA_DIR = DATA_ROOT / "camera"
    LIDAR_DIR = DATA_ROOT / "lidar_png"
    ANNOTATION_SAM_DIR = DATA_ROOT / "annotation_sam"
    ZOD_DATA_ROOT = Path("/gpfs/mariana/smbhome/totahv/zod-data/single_frames")
    OUTPUT_ROOT = DATA_ROOT / "vlm"
    ANNOTATION_OUT_DIR = OUTPUT_ROOT / "annotation"
    RESULTS_DIR = OUTPUT_ROOT / "results"
    VIS_DIR = OUTPUT_ROOT / "visualization"

# --- Class definitions ---
CLASS_ID_TO_NAME = {
    0: "background",
    1: "ignore",
    2: "vehicle",
    3: "sign",
    4: "cyclist",
    5: "pedestrian",
}

# Matches SAM generator thresholds
MIN_OBJECT_PIXELS = {
    2: 30,   # vehicle
    3: 15,   # sign
    4: 15,   # cyclist
    5: 15,   # pedestrian
}

# Padding (px) added around tight bbox when cropping — in 768px space
CROP_PADDING = 16

# Minimum crop dimension sent to VLM (px). Crops smaller than this are upscaled.
MIN_CROP_SIZE = 224

# Mask overlay alpha (0–1)
OVERLAY_ALPHA = 0.5

# BGR colors per class (matches SAM generator)
CLASS_COLORS_BGR = {
    2: (0, 0, 255),      # vehicle — red
    3: (0, 255, 255),    # sign — yellow
    4: (255, 0, 255),    # cyclist — magenta
    5: (0, 255, 0),      # pedestrian — green
}

# --- VLM backend ---
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5vl:7b"
VLM_TIMEOUT = 60          # seconds per call
VLM_MAX_RETRIES = 1

# --- Metadata pre-filter (skip VLM entirely) ---
# Only skip VLM for cases where the answer is unambiguous from geometry alone.
# Everything else goes to VLM so we can measure its actual contribution.
# NOTE: there is deliberately no auto-accept for large masks — on flagged frames
# the largest connected components are often exactly the leaked noise blobs.
AUTO_REJECT_MAX_ASPECT = 6.0      # clearly leaked: extreme horizontal/vertical bleed
AUTO_REJECT_MIN_ASPECT = 0.12     # clearly leaked: extreme horizontal/vertical bleed

# --- Deterministic consistency check ---
# Fraction of mask pixels with LiDAR returns below which the mask is flagged
# as geometrically unsupported. Real masks on ZOD have median support ~0.63
# and <1.5% of them fall below 0.1, so this only fires on sky/void leaks.
LIDAR_SUPPORT_MIN = 0.10
