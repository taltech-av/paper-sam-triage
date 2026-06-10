#!/usr/bin/env python3
"""
Quick smoke test: load a real frame, build a bundle for the first mask proposal,
and send all 3 images to Ollama with a BBox agent prompt.
"""

import cv2
from PIL import Image

import config
from core.bundle import build_bundle
from core.mask_extractor import extract_proposals
from vlm.ollama_client import OllamaClient

FRAME_ID = "frame_000004"

ann_path = config.ANNOTATION_SAM_DIR / f"{FRAME_ID}.png"
cam_path = config.CAMERA_DIR / f"{FRAME_ID}.png"
lid_path = config.LIDAR_DIR / f"{FRAME_ID}.png"

print(f"Frame: {FRAME_ID}")
print(f"Model: {config.OLLAMA_MODEL}")
print()

proposals = extract_proposals(ann_path, FRAME_ID)
proposal = proposals[0]
print(f"Using proposal: mask_id={proposal.mask_id} class={proposal.class_name} "
      f"pixels={proposal.pixel_count} bbox={proposal.bbox}")
print()

camera_img = cv2.imread(str(cam_path))
lidar_img = cv2.imread(str(lid_path))
bundle = build_bundle(proposal, camera_img, lidar_img)

print(f"Bundle metadata: {bundle.metadata}")
print()

# Show image sizes being sent
for name, img in [("rgb_crop", bundle.rgb_crop),
                  ("overlay_crop", bundle.overlay_crop),
                  ("depth_crop", bundle.depth_crop)]:
    print(f"  {name}: {img.size} px")
print()

prompt = (
    f"You are evaluating a bounding-box proposal from an autonomous driving dataset.\n"
    f"You are given three images:\n"
    f"  1. RGB camera crop\n"
    f"  2. The same crop with the segmentation mask highlighted\n"
    f"  3. LiDAR depth projection of the same region\n\n"
    f"Class: {proposal.class_name}\n"
    f"Mask pixels: {proposal.pixel_count}\n"
    f"LiDAR support ratio: {bundle.metadata['lidar_support_ratio']}\n\n"
    f"Does the highlighted region correspond to a real, plausible {proposal.class_name} "
    f"in a driving scene?\n\n"
    f"Reply with exactly one word: VALID or INVALID"
)

print("Prompt:")
print(prompt)
print()
print("Querying Ollama...")

client = OllamaClient()
response = client.query([bundle.rgb_crop, bundle.overlay_crop, bundle.depth_crop], prompt)

print(f"Raw response: {repr(response)}")
