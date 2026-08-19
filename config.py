import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from `.env` into the environment, without overriding
    anything already exported. Keeps machine-specific paths out of the source
    tree: see .env.example for the full list of variables."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


_load_dotenv(_REPO_ROOT / ".env")


def _env_path(name: str, default):
    """Path from the environment, or `default` when the variable is unset."""
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return Path(default).expanduser() if isinstance(default, str) else default


# --- Data paths ---
# Every path below is machine-specific and is therefore read from the environment.
# Copy .env.example to .env and set them once; nothing else in the codebase
# hardcodes a location. The defaults are repo-relative so a fresh clone runs
# without editing source, and so no author's home directory ships in the repo.
DATA_ROOT = _env_path("VLM_DATA_ROOT", _REPO_ROOT / "data")
CAMERA_DIR = DATA_ROOT / "camera"
LIDAR_DIR = DATA_ROOT / "lidar_png"
ANNOTATION_SAM_DIR = DATA_ROOT / "annotation_sam"
FRAMES_FILE = _env_path("VLM_FRAMES_FILE", _REPO_ROOT / "frames" / "bad_frames.csv")

# Original 4K ZOD images — used instead of 768px camera/ when available.
# Unset VLM_ZOD_ROOT (the default) to always use the downscaled camera/ images.
ZOD_DATA_ROOT = _env_path("VLM_ZOD_ROOT", None)

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


# Variants whose decisions read no VLM output at all: `raw_sam` accepts every
# mask, `swin_only` thresholds the Swin agreement score. Two runs of different
# models produce byte-identical results for these, so they are written once to
# DATA_ROOT alongside annotation_sam rather than duplicated under each
# vlm/<tag>/. Filing them under a model tag also implies a provenance they do
# not have — nothing in `vlm/llava_34b/annotation_raw_sam` came from LLaVA.
VLM_INDEPENDENT_VARIANTS = frozenset({"raw_sam", "swin_only"})


def variant_dir(variant_name: str, tag_root: Path) -> Path:
    """Where a variant's annotation PNGs belong: dataset root, or under the tag."""
    root = DATA_ROOT if variant_name in VLM_INDEPENDENT_VARIANTS else tag_root
    return root / f"annotation_{variant_name}"


def set_run_tag(tag: str) -> None:
    """Namespace all outputs under vlm/<tag>/ so multiple runs never collide."""
    _set_output_dirs(DATA_ROOT / "vlm" / tag)


def use_hpc():
    """Switch all data paths to HPC. Call before any I/O."""
    global DATA_ROOT, CAMERA_DIR, LIDAR_DIR, ANNOTATION_SAM_DIR
    global ZOD_DATA_ROOT, FUSION_DIR, SWIN_CFG_PATH, SWIN_CKPT_PATH, WORKERS
    # Only paths and throughput belong here. Anything that changes what a run
    # *contains* — thresholds, candidate caps, vocabularies — stays a module
    # constant, so the same code produces the same annotations on either
    # machine. DISCOVERY_MAX_CANDIDATES used to be set here and silently made
    # local runs unable to reproduce the published candidate set.
    WORKERS = 4                     # matches the wall-clock figure reported in the paper
    DATA_ROOT = _env_path("VLM_HPC_DATA_ROOT", DATA_ROOT)
    FUSION_DIR = _env_path("VLM_HPC_FUSION_DIR", FUSION_DIR)
    _swin_model_dir = _env_path("VLM_HPC_SWIN_DIR", None)
    if _swin_model_dir is not None:
        SWIN_CFG_PATH  = _swin_model_dir / "config_9.json"
        SWIN_CKPT_PATH = _swin_model_dir / "best.pth"
    CAMERA_DIR = DATA_ROOT / "camera"
    LIDAR_DIR = DATA_ROOT / "lidar_png"
    ANNOTATION_SAM_DIR = DATA_ROOT / "annotation_sam"
    ZOD_DATA_ROOT = _env_path("VLM_HPC_ZOD_ROOT", ZOD_DATA_ROOT)
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
_ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_URL = _ollama_host if _ollama_host.startswith("http") else f"http://{_ollama_host}"
OLLAMA_MODEL = "qwen2.5vl:7b"
VLM_TIMEOUT = 120          # seconds per call — sized for the 34B/72B backends
VLM_MAX_RETRIES = 1

# --- Parallelism ---
# Throughput only — this changes how long a run takes, never what it produces.
# use_hpc() sets 4, which is the figure the paper's wall-clock timings assume.
WORKERS = 8

# --- Object discovery (Swin-detected regions absent from annotation_sam) ---
# Minimum connected-component size in 384×384 Swin space to be considered a candidate.
# Corresponds to ~80 px in the 768px camera image.
DISCOVERY_MIN_PIXELS = 20
# Max VLM confirmation calls per frame (largest candidates first).
# This is not a tuning knob: it fixes the candidate set, so changing it changes
# which objects can be discovered at all. Both published runs used 20, giving
# the 55,256 candidates the paper reports, and it must stay 20 to reproduce
# them. It is deliberately not overridden in use_hpc() — a value that decides
# what the results contain cannot depend on which machine the run lands on.
DISCOVERY_MAX_CANDIDATES = 20

# --- Swin quality agent ---
# Checkout of the fusion-training repo that supplies the CLFTv2/Swin model; the
# checkpoint is the one trained on the 2,319-frame clean partition. Both are
# released with the artifact bundle — see DATA.md.
FUSION_DIR = _env_path("VLM_FUSION_DIR", _REPO_ROOT.parent / "fusion-training")
SWIN_CFG_PATH  = _env_path("VLM_SWIN_CFG", FUSION_DIR / "config/zod/swin/config_9.json")
SWIN_CKPT_PATH = _env_path("VLM_SWIN_CKPT", FUSION_DIR / "logs/zod/swin/config_9/best.pth")
SWIN_DEVICE = os.environ.get("VLM_SWIN_DEVICE", "cuda:0")

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
