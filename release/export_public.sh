#!/usr/bin/env bash
#
# Export the public subset of this repository to a clean directory.
#
#     ./release/export_public.sh ~/paper-vlm-annotation-pipeline-public
#
# This repo doubles as the working tree for the manuscripts, which are not part
# of the open-source release. Rather than deleting paper/
# from a repo that is still being written in, the release is an export: this
# script copies everything the public repo needs and nothing it does not, then
# initialises a fresh history so no HPC path or personal directory from the
# development history ships with it.
#
# Re-runnable: the destination is refreshed in place, and a re-run after edits
# updates the tree without touching the fresh repo's history.

set -euo pipefail

DEST="${1:?usage: export_public.sh <destination-dir>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -e "$DEST" && ! -d "$DEST" ]]; then
    echo "error: $DEST exists and is not a directory" >&2
    exit 1
fi
mkdir -p "$DEST"

echo "exporting $SRC -> $DEST"

rsync -a --delete-excluded \
    --exclude '.git/' \
    --exclude 'paper/' \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'commands.txt' \
    --exclude 'logs/' \
    --exclude 'data/' \
    --exclude 'human_verified_output/' \
    --exclude 'qualitative_candidates/' \
    --exclude '.claude/' \
    "$SRC"/ "$DEST"/

# Verify nothing machine-specific survived the copy.
echo "checking for leaked paths..."
# --exclude: this script carries the pattern itself and would always match.
if grep -rIn --exclude-dir=.git --exclude='export_public.sh' \
        -E '/gpfs/|/mnt/ml/|totahv|hotmail' "$DEST" ; then
    echo "error: machine-specific paths or personal identifiers found above" >&2
    exit 1
fi
echo "  clean"

# The reproduction commands expect this directory to exist.
mkdir -p "$DEST/human_verified_output"
cat > "$DEST/human_verified_output/.gitkeep" <<'EOF'
EOF

if [[ ! -d "$DEST/.git" ]]; then
    git -C "$DEST" init -q
    echo "initialised a fresh git history in $DEST"
fi

echo
echo "exported. Next:"
echo "  cd $DEST && git add -A && git commit -m 'Initial public release'"
echo "  python release/make_bundle.py --out ~/zenodo_bundle   # from the working repo"
