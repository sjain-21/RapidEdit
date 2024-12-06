import torch
import numpy as np
from PIL import Image
import os
import random
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Directory to store model checkpoints
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Path to the SAM model checkpoint
SAM_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "sam_vit_h_4b8939.pth")

# Download the SAM checkpoint if not already present
if not os.path.exists(SAM_CHECKPOINT_PATH):
    print("Downloading SAM checkpoint...")
    os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O {SAM_CHECKPOINT_PATH}")

# Load the SAM model
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
sam = sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)


def generate_sam_masks(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Generate masks
    masks = mask_generator.generate(image_np)

    # Create an overlay with masks
    mask_overlay = image_np.copy()
    for idx, mask in enumerate(masks):
        color = [random.randint(0, 255) for _ in range(3)]
        segmentation = mask["segmentation"]
        for c in range(3):
            mask_overlay[..., c] = np.where(segmentation, color[c], mask_overlay[..., c])

    overlay_path = os.path.join(os.path.dirname(image_path), "mask_overlay.png")
    Image.fromarray(mask_overlay).save(overlay_path)
    return masks, overlay_path


def stitch_selected_masks(image_path, selected_masks, save_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    combined_mask = np.zeros_like(image_np[..., 0], dtype=np.uint8)

    masks = mask_generator.generate(image_np)
    for mask_idx in selected_masks:
        segmentation = masks[int(mask_idx) - 1]["segmentation"]
        combined_mask = np.logical_or(combined_mask, segmentation).astype(np.uint8)

    Image.fromarray(combined_mask * 255).save(save_path)
