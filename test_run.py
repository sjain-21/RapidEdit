import os
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import AutoPipelineForInpainting, LCMScheduler
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import random
import cv2

# Paths
IMAGE_PATH = "/scratch/vd2298/ComputerVision24/inpaint.png"  # Replace with your input image path
SAM_CHECKPOINT_PATH = "/scratch/vd2298/ComputerVision24/models/sam_vit_h_4b8939.pth"  # SAM checkpoint
OUTPUT_DIR = "./outputs"  # Directory for outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SAM Model
print("Loading SAM...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to("cuda")
print(next(sam.parameters()).is_cuda)  # This should print True
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,                # Higher resolution for the grid
    pred_iou_thresh=0.9,               # Confidence threshold for masks
    stability_score_thresh=0.9,        # Stability score for masks
    crop_n_layers=1,                   # Number of crop layers for refinement
    crop_n_points_downscale_factor=2,  # Downscale factor for crops
    min_mask_region_area=500           # Minimum mask area to filter small artifacts
)

# Load SDXL Inpainting Pipeline
print("Loading Inpainting Pipeline...")
pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

# Load Image
print(f"Loading Image: {IMAGE_PATH}")
input_image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(input_image)

# Generate SAM Masks
print("Generating Masks with SAM...")
sam_masks = mask_generator.generate(image_np)

# Filter Masks by Area
filtered_masks = [m for m in sam_masks if m["area"] > 500]  # Keep only masks with area > 500 pixels

# Visualize Masks
mask_overlay = image_np.copy()
mask_labels = []
for idx, mask in enumerate(filtered_masks):
    segmentation = mask["segmentation"]
    color = [random.randint(0, 255) for _ in range(3)]
    for c in range(3):
        mask_overlay[..., c] = np.where(segmentation, color[c], mask_overlay[..., c])
    ys, xs = np.where(segmentation)
    center_x, center_y = xs.mean().astype(int), ys.mean().astype(int)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(mask_overlay, str(idx + 1), (center_x, center_y), font, 0.5, (0, 0, 0), 2)
    cv2.putText(mask_overlay, str(idx + 1), (center_x, center_y), font, 0.5, (255, 255, 255), 1)
    mask_labels.append((idx + 1, segmentation))

# Save Overlay Image
overlay_path = os.path.join(OUTPUT_DIR, "mask_overlay.png")
Image.fromarray(mask_overlay).save(overlay_path)
print(f"Mask overlay saved at {overlay_path}")

# User selects masks
print(f"Generated {len(mask_labels)} masks.")
selected_indices = input("Enter mask indices (comma-separated, or leave blank for highest-probability mask): ")
if not selected_indices.strip():
    # Use the highest-probability mask
    highest_confidence_mask = max(filtered_masks, key=lambda x: x["predicted_iou"])
    combined_mask = highest_confidence_mask["segmentation"].astype(np.uint8)
else:
    # Combine selected masks
    selected_indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
    combined_mask = np.zeros_like(image_np[..., 0], dtype=np.uint8)
    for idx in selected_indices:
        combined_mask = np.logical_or(combined_mask, filtered_masks[idx]["segmentation"]).astype(np.uint8)

# Save Combined Mask
combined_mask_image = Image.fromarray(combined_mask * 255)
combined_mask_path = os.path.join(OUTPUT_DIR, "combined_mask.png")
combined_mask_image.save(combined_mask_path)
print(f"Combined mask saved at {combined_mask_path}")

# Add padding to make the image square
max_dim = max(input_image.size)
padded_image = ImageOps.expand(
    input_image,
    border=(
        (max_dim - input_image.width) // 2,
        (max_dim - input_image.height) // 2,
    ),
    fill=(255, 255, 255),
)
padded_mask_image = ImageOps.expand(
    combined_mask_image,
    border=(
        (max_dim - input_image.width) // 2,
        (max_dim - input_image.height) // 2,
    ),
    fill=(0),
)

# Save Padded Image and Mask
padded_image.save(os.path.join(OUTPUT_DIR, "padded_input_image.png"))
padded_mask_image.save(os.path.join(OUTPUT_DIR, "padded_mask_image.png"))

# Run Inpainting
print("Running Inpainting...")
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
negative_prompt = "bad architecture, unstable, poor details, blurry"
generator = torch.manual_seed(0)
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=padded_image,
    generator=generator,
    mask_image=padded_mask_image,
    num_inference_steps=4,
    guidance_scale=4,
).images[0]

# Crop Output Image Back to Original Dimensions
output_image = result.crop(
    (
        (max_dim - input_image.width) // 2,
        (max_dim - input_image.height) // 2,
        (max_dim + input_image.width) // 2,
        (max_dim + input_image.height) // 2,
    )
)

# Save Output Image
output_image_path = os.path.join(OUTPUT_DIR, "final_output_image.png")
output_image.save(output_image_path)
print(f"Final output image saved at {output_image_path}")
