from pathlib import Path

# --- Data paths ---
DATA_ROOT = Path("/run/media/tom/ml/zod_temp")
CAMERA_DIR = DATA_ROOT / "camera"
LIDAR_DIR = DATA_ROOT / "lidar_png"
ANNOTATION_SAM_DIR = DATA_ROOT / "annotation_sam"
FRAMES_FILE = Path(__file__).parent / "frames" / "frames.txt"

# Original 4K ZOD images — used instead of 768px camera/ when available
# Set to None to always use the downscaled camera/ images
ZOD_DATA_ROOT = Path("/run/media/tom/ml/zod-data/single_frames")

OUTPUT_ROOT = DATA_ROOT
ANNOTATION_OUT_DIR = OUTPUT_ROOT / "annotation_vllm"
RESULTS_DIR = OUTPUT_ROOT / "vllm_results"

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

# --- Processing ---
WORKERS = 1               # parallel frame workers (increase if GPU allows)

# --- Metadata pre-filter (skip VLM entirely) ---
# Only skip VLM for cases where the answer is unambiguous from geometry alone.
# Everything else goes to VLM so we can measure its actual contribution.
AUTO_ACCEPT_LARGE_PIXELS = 5000   # clearly real: SAM can't produce 5K+ px on a ZOD bbox by mistake
AUTO_REJECT_MAX_ASPECT = 6.0      # clearly leaked: extreme horizontal/vertical bleed
AUTO_REJECT_MIN_ASPECT = 0.12     # clearly leaked: extreme horizontal/vertical bleed
