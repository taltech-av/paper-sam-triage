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
ANNOTATION_OUT_DIR = OUTPUT_ROOT / "annotation_full"  # triage + discovery masks (uint8 PNG)
RESULTS_DIR = OUTPUT_ROOT / "results"                # per-frame JSON + summary
VIS_DIR = OUTPUT_ROOT / "visualization"              # triage overlay images


def _set_output_dirs(root: Path) -> None:
    global OUTPUT_ROOT, ANNOTATION_OUT_DIR, RESULTS_DIR, VIS_DIR
    OUTPUT_ROOT = root
    ANNOTATION_OUT_DIR = root / "annotation_full"
    RESULTS_DIR = root / "results"
    VIS_DIR = root / "visualization"


def set_run_tag(tag: str) -> None:
    """Namespace all outputs under vlm/<tag>/ so multiple runs never collide."""
    _set_output_dirs(DATA_ROOT / "vlm" / tag)


def use_hpc():
    """Switch all data paths to HPC. Call before any I/O."""
    global DATA_ROOT, CAMERA_DIR, LIDAR_DIR, ANNOTATION_SAM_DIR
    global ZOD_DATA_ROOT, FUSION_DIR, SWIN_CFG_PATH, SWIN_CKPT_PATH, WORKERS
    global DISCOVERY_MAX_CANDIDATES, VLM_TIMEOUT
    WORKERS = 4
    DISCOVERY_MAX_CANDIDATES = 20   # HPC GPU responds faster, can handle more per frame
    VLM_TIMEOUT = 120               # larger models (72B/90B) need more time per call
    DATA_ROOT = Path("/gpfs/mariana/smbhome/totahv/zod_temp")
    FUSION_DIR = Path("/gpfs/mariana/smbhome/totahv/fusion-training")
    _swin_model_dir = Path("/gpfs/mariana/smbhome/totahv/models/swin")
    SWIN_CFG_PATH  = _swin_model_dir / "config_9.json"
    SWIN_CKPT_PATH = _swin_model_dir / "best.pth"
    CAMERA_DIR = DATA_ROOT / "camera"
    LIDAR_DIR = DATA_ROOT / "lidar_png"
    ANNOTATION_SAM_DIR = DATA_ROOT / "annotation_sam"
    ZOD_DATA_ROOT = Path("/gpfs/mariana/smbhome/totahv/zod-data/single_frames")
    _set_output_dirs(DATA_ROOT / "vlm")

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
# Maximum crop dimension sent to VLM (px). Crops larger than this are downscaled.
# 4K source crops can be 2000px+ which wastes tokens; 512px is sufficient for a 7B model.
MAX_CROP_SIZE = 512

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
VLM_TIMEOUT = 120          # seconds per call (local 7B; use_hpc() raises to 120 for 72B/90B)
VLM_MAX_RETRIES = 1

# --- Parallelism ---
# Local: 1 worker keeps Ollama calls sequential — no queue build-up, no timeouts.
# HPC: use_hpc() raises this to 16; the dedicated A100 can handle concurrent calls.
WORKERS = 8

# --- Object discovery (Swin-detected regions absent from annotation_sam) ---
# Minimum connected-component size in 384×384 Swin space to be considered a candidate.
# Corresponds to ~80 px in the 768px camera image.
DISCOVERY_MIN_PIXELS = 20
# Max VLM confirmation calls per frame (largest candidates first).
# 4 workers × 10 candidates = 40 max concurrent Ollama calls — within safe limits.
# On HPC with faster GPUs this can be raised; 10 is a conservative default.
DISCOVERY_MAX_CANDIDATES = 10

# --- Swin quality agent ---
FUSION_DIR = Path("/run/media/tom/ml/projects/fusion-training")
SWIN_CFG_PATH  = FUSION_DIR / "config/zod/swin/config_9.json"
SWIN_CKPT_PATH = FUSION_DIR / "logs/zod/swin/config_9/best.pth"
SWIN_DEVICE = "cuda:0"

# Default thresholds (used when no per-class override exists)
SWIN_AGREEMENT_THRESHOLD  = 0.30  # α ≥ threshold → quality "good"
SWIN_SKIP_BBOX_THRESHOLD  = 0.70  # α ≥ threshold → skip BBox VLM entirely

# Per-class overrides — cyclist/pedestrian get lower thresholds because
# Swin achieves only ~35% mIoU on small objects vs ~70%+ on vehicles/signs.
# Using the same threshold as large objects causes excessive false rejections.
SWIN_AGREEMENT_THRESHOLD_BY_CLASS = {
    2: 0.30,  # vehicle      — Swin reliable (~70% mIoU)
    3: 0.30,  # sign         — Swin reliable
    4: 0.15,  # cyclist      — Swin ~35% mIoU on small objects
    5: 0.15,  # pedestrian   — Swin ~35% mIoU on small objects
}
SWIN_SKIP_BBOX_THRESHOLD_BY_CLASS = {
    2: 0.70,  # vehicle
    3: 0.70,  # sign
    4: 0.40,  # cyclist      — rarely achieves 0.70 even on valid masks
    5: 0.40,  # pedestrian
}


def swin_quality_threshold(class_id: int) -> float:
    return SWIN_AGREEMENT_THRESHOLD_BY_CLASS.get(class_id, SWIN_AGREEMENT_THRESHOLD)


def swin_skip_threshold(class_id: int) -> float:
    return SWIN_SKIP_BBOX_THRESHOLD_BY_CLASS.get(class_id, SWIN_SKIP_BBOX_THRESHOLD)

# --- Metadata pre-filter (skip VLM entirely) ---
# Only skip VLM for cases where the answer is unambiguous from geometry alone.
# Everything else goes to VLM so we can measure its actual contribution.
# NOTE: there is deliberately no auto-accept for large masks — on flagged frames
# the largest connected components are often exactly the leaked noise blobs.
AUTO_REJECT_MAX_ASPECT = 4.5      # clearly leaked: extreme horizontal/vertical bleed
AUTO_REJECT_MIN_ASPECT = 0.12     # clearly leaked: extreme horizontal/vertical bleed

# --- Deterministic consistency check ---
# Fraction of mask pixels with LiDAR returns below which the mask is flagged
# as geometrically unsupported. Real masks on ZOD have median support ~0.63
# and <1.5% of them fall below 0.1, so this only fires on sky/void leaks.
LIDAR_SUPPORT_MIN = 0.10
