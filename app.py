import streamlit as st
import os
from PIL import Image, ImageDraw
import random
import cv2
from diffusers import AutoPipelineForInpainting, LCMScheduler
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch


# Ensure uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_dummy_masks(image_path):
    """
    Simulates generating masks for an image by creating dummy data.
    Returns a list of dummy masks and a path to the mask overlay image.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  # SAM checkpoint
    print("Loading SAM...")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
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
    sam_masks = mask_generator.generate(image_np)

    # Filter Masks by Area
    filtered_masks = [m for m in sam_masks if m["area"] > 500]  # Keep only masks with area > 500 pixels
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

    # Save the mask overlay image
    mask_overlay_path = os.path.join(UPLOAD_FOLDER, "mask_overlay.png")
    Image.fromarray(mask_overlay).save(mask_overlay_path)

    return filtered_masks, mask_overlay_path

def stitch_dummy_masks(image_path, filtered_masks, selected_indices):
    """
    Simulates stitching selected masks into a single combined mask.
    """
    # Add selected masks to the combined mask
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    selected_indices = [int(idx.strip()) - 1 for idx in selected_indices]
    combined_mask = np.zeros_like(image_np[..., 0], dtype=np.uint8)
    for idx in selected_indices:
        combined_mask = np.logical_or(combined_mask, filtered_masks[idx]["segmentation"]).astype(np.uint8)

    # Save the combined mask
    combined_mask_image = Image.fromarray(combined_mask * 255)
    stitched_mask_path = os.path.join(UPLOAD_FOLDER, "stitched_mask.png")
    combined_mask_image.save(stitched_mask_path)

def generate_result(input_image_path, mask_image_path, prompt, negative_prompt):
    """
    Generates the result using the diffusers pipeline.
    """
    # Load SDXL Inpainting Pipeline
    print("Loading Inpainting Pipeline...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.fuse_lora()

    # Run Inpainting
    print("Running Inpainting...")
    generator = torch.manual_seed(0)
    input_image = Image.open(input_image_path).convert("RGB")
    masked_image = Image.open(mask_image_path).convert("RGB")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,  # Directly use the input image
        generator=generator,
        mask_image=masked_image,
        num_inference_steps=20,  # Increased steps for better refinement
        guidance_scale=7.5,      # Reduced guidance scale
    ).images[0]

    # Save Output Image
    output_image_path = os.path.join(UPLOAD_FOLDER, "result.png")
    result.save(output_image_path)
    print(f"Final output image saved at {output_image_path}")
    return output_image_path

def get_current_stage():
    """
    Retrieve the current stage from URL query parameters
    """
    return st.query_params.get("stage", "upload")

def update_query_params(stage, **kwargs):
    """
    Update URL query parameters for the current stage
    """
    # Clear existing query parameters
    st.query_params.clear()
    
    # Set new parameters
    st.query_params["stage"] = stage
    for key, value in kwargs.items():
        st.query_params[key] = value

def main():
    st.set_page_config(page_title="RapidEdit")
    st.title("RapidEdit")

    # Determine current stage from URL
    current_stage = get_current_stage()

    # Initialize session state if not already set
    if 'image_path' not in st.session_state:
        st.session_state.image_path = None
        st.session_state.prompt = None
        st.session_state.mask_overlay_path = None
        st.session_state.masks = None
        st.session_state.stitched_mask_path = None
        st.session_state.selected_masks = []
        st.session_state.output_image_path = None

    # Restore state based on query parameters if needed
    if current_stage == 'upload':
        st.session_state.stage = 'upload'

    elif current_stage == 'mask_selection':
        # Restore image path and masks from query params if available
        image_filename = st.query_params.get("image", None)
        if image_filename:
            st.session_state.image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            st.session_state.stage = 'mask_selection'
            
            # Regenerate masks if needed
            if st.session_state.mask_overlay_path is None:
                st.session_state.masks, st.session_state.mask_overlay_path = generate_dummy_masks(st.session_state.image_path)

    elif current_stage == 'check':
        # Restore state for check stage
        st.session_state.stage = 'check'

    elif current_stage == 'result':
        # Restore state for result stage
        st.session_state.stage = 'result'

    # Upload Page
    if current_stage == 'upload':
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        prompt = st.text_input("Enter Prompt:")
        negative_promt = st.text_input("Enter Negative Prompt:")
        if uploaded_file is not None:
            # Save uploaded image
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.image(image_path)

            # Generate masks when file is uploaded
            if st.button("Create Mask") and prompt:
                # Generate masks
                masks, mask_overlay_path = generate_dummy_masks(image_path)
                
                # Update session state
                st.session_state.image_path = image_path
                st.session_state.mask_overlay_path = mask_overlay_path
                st.session_state.masks = masks
                st.session_state.prompt = prompt
                st.session_state.negative_prompt = negative_promt
                
                # Update URL to mask selection stage
                update_query_params("mask_selection", image=os.path.basename(image_path))
                st.rerun()

    # Mask Selection Page
    elif current_stage == 'mask_selection':
        st.header("Select Masks")
        
        # Display mask overlay
        st.image(st.session_state.mask_overlay_path, caption="Mask Overlay")
        
        # Mask selection
        selected_masks = st.multiselect(
            "Select the masks you want to use:", 
            [f"{id}" for id in range(len(st.session_state.masks))]
        )
        
        if st.button("Proceed"):
            # Convert selected masks to indices
            # selected_mask_indices = [int(mask.split()[-1]) for mask in selected_masks]
            
            # # Stitch masks
            stitched_mask_path = os.path.join(UPLOAD_FOLDER, "stitched_mask.png")
            stitch_dummy_masks(st.session_state.image_path, st.session_state.masks, selected_masks)
            
            # Update session state
            st.session_state.stitched_mask_path = stitched_mask_path
            st.session_state.selected_masks = selected_masks
            
            # Update URL to check stage
            update_query_params("check", image=os.path.basename(st.session_state.image_path))
            st.rerun()

    # Check Page
    elif current_stage == 'check':
        st.header("Check and Update")
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.image_path, caption="Input Image")
        with col2:
            st.image(st.session_state.stitched_mask_path, caption="Stitched Mask")
        
        # Prompt update
        updated_prompt = st.text_input("Update Prompt:", value=st.session_state.prompt or "")
        updated_negative_prompt = st.text_input("Update Prompt:", value=st.session_state.negative_prompt or "")
        if st.button("Update and Proceed"):
            # Update prompt
            st.session_state.prompt = updated_prompt
            st.session_state.negative_prompt = updated_negative_prompt
            # Update URL to result stage
            result_path = os.path.join(UPLOAD_FOLDER, "result.png")
            result_path = generate_result(st.session_state.image_path, st.session_state.stitched_mask_path, updated_prompt, updated_negative_prompt)
            st.session_state.result_path = result_path
            update_query_params("result", image=os.path.basename(st.session_state.result_path))
            st.rerun()

    # Result Page
    elif current_stage == 'result':
        st.header("Result")
        
        # Display prompt
        st.write(f"**Prompt:** {st.session_state.prompt}")
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.image_path, caption="Input Image")
        with col2:
            # Simulate output by copying input image
            result_path = os.path.join(UPLOAD_FOLDER, "result.png")
            Image.open(st.session_state.result_path).save(result_path)
            st.image(result_path, caption="Result Image")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reuse Output Image"):
                # Update session state
                st.session_state.image_path = output_image_path
                
                # Update URL to mask selection stage
                update_query_params("mask_selection", image=os.path.basename(output_image_path))
                st.rerun()

        with col2:
            if st.button("Restart Process"):
                # Reset session state
                st.session_state.image_path = None
                st.session_state.prompt = None
                st.session_state.mask_overlay_path = None
                st.session_state.masks = None
                st.session_state.stitched_mask_path = None
                
                # Update URL to upload stage
                update_query_params("upload")
                st.rerun()

if __name__ == "__main__":
    main()
