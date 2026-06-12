#!/bin/bash
# Upload only the frames listed in a CSV file to HPC.
# Transfers annotation_sam, camera (768px), lidar_png, and 4K images.
#
# Usage:
#   bash slurms/upload_frames.sh                         # uses bad_frames.csv
#   bash slurms/upload_frames.sh frames/good_frames.csv  # explicit CSV

set -e

CSV="${1:-frames/bad_frames.csv}"
HPC="totahv@base.hpc.taltech.ee"

# ── Update ZOD destination paths to match your HPC storage ───────────────────
ZOD_TEMP_DST="$HPC:/gpfs/mariana/smbhome/totahv/zod_temp"
ZOD_4K_DST="$HPC:/gpfs/mariana/smbhome/totahv/zod-data/single_frames"
# ──────────────────────────────────────────────────────────────────────────────

ZOD_TEMP_SRC="/run/media/tom/ml/zod_temp"
ZOD_4K_SRC="/run/media/tom/ml/zod-data/single_frames"

if [[ ! -f "$CSV" ]]; then
    echo "ERROR: CSV not found: $CSV"
    exit 1
fi

# Extract unique frame IDs (strips 'camera/' prefix and '.png' suffix)
FRAME_IDS=$(grep -oP 'frame_\d+' "$CSV" | sort -u)
N=$(echo "$FRAME_IDS" | wc -l)
echo "Frames to upload: $N  (from $CSV)"

# Build file list for zod_temp (annotation_sam + camera + lidar_png)
TMPLIST=$(mktemp)
for fid in $FRAME_IDS; do
    echo "annotation_sam/${fid}.png" >> "$TMPLIST"
    echo "camera/${fid}.png"         >> "$TMPLIST"
    echo "lidar_png/${fid}.png"      >> "$TMPLIST"
done

# Build file list for 4K images (one subfolder per frame)
TMPLIST_4K=$(mktemp)
for fid in $FRAME_IDS; do
    num="${fid#frame_}"              # frame_099990 → 099990
    echo "${num}/camera_front_blur/" >> "$TMPLIST_4K"
done

echo ""
echo "=== Uploading zod_temp (masks + camera + lidar) ==="
rsync -av --progress --files-from="$TMPLIST" \
    "$ZOD_TEMP_SRC/" "$ZOD_TEMP_DST/"

echo ""
echo "=== Uploading 4K camera images ==="
rsync -av --progress --files-from="$TMPLIST_4K" \
    "$ZOD_4K_SRC/" "$ZOD_4K_DST/"

rm -f "$TMPLIST" "$TMPLIST_4K"

echo ""
echo "Done. Make sure config.py on HPC points to the ZOD paths you used above."
