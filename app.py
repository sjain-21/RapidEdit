import os
import random
import numpy as np
import torch
from PIL import Image
import cv2
import streamlit as st
from diffusers import AutoPipelineForInpainting, LCMScheduler
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
OUTPUT_DIR = "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sam_model(checkpoint_path):
    """Load Segment Anything Model (SAM)."""
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500,
    )
    return mask_generator

def generate_masks(image_np, mask_generator):
    """Generate and process segmentation masks."""
    # Set the image for the predictor
    mask_generator.predictor.set_image(image_np)
    
    # Generate masks
    masks = mask_generator.generate(image_np)
    filtered_masks = [m for m in masks if m["area"] > 500]
    
    # Create mask overlay with labels
    mask_overlay = image_np.copy()
    mask_labels = []
    
    for idx, mask in enumerate(filtered_masks):
        segmentation = mask["segmentation"]
        color = [random.randint(0, 255) for _ in range(3)]
        
        # Color the mask
        for c in range(3):
            mask_overlay[..., c] = np.where(segmentation, color[c], mask_overlay[..., c])
        
        # Add mask number
        ys, xs = np.where(segmentation)
        center_x, center_y = xs.mean().astype(int), ys.mean().astype(int)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(mask_overlay, str(idx + 1), (center_x, center_y), font, 0.5, (0, 0, 0), 2)
        cv2.putText(mask_overlay, str(idx + 1), (center_x, center_y), font, 0.5, (255, 255, 255), 1)
        
        mask_labels.append((idx + 1, segmentation))
    
    # Reset the image
    mask_generator.predictor.reset_image()
    
    return filtered_masks, mask_overlay, mask_labels

def create_combined_mask(filtered_masks, selected_indices=None):
    """Create a combined mask from selected or highest confidence mask."""
    combined_mask = np.zeros_like(filtered_masks[0]["segmentation"], dtype=np.uint8)
    
    if selected_indices:
        for idx in map(int, selected_indices):
            combined_mask = np.logical_or(combined_mask, filtered_masks[idx - 1]["segmentation"]).astype(np.uint8)
    else:
        highest_confidence_mask = max(filtered_masks, key=lambda x: x["predicted_iou"])
        combined_mask = highest_confidence_mask["segmentation"].astype(np.uint8)
    
    return Image.fromarray(combined_mask * 255)

def load_inpainting_pipeline():
    """Load and configure the inpainting pipeline."""
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.fuse_lora()
    return pipe

def run_inpainting(pipe, input_image, combined_mask_image, prompt, negative_prompt):
    """Run the inpainting process."""
    generator = torch.manual_seed(0)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        generator=generator,
        mask_image=combined_mask_image,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]
    return result

def save_image(image, filename):
    """Save an image to the output directory."""
    output_path = os.path.join(OUTPUT_DIR, filename)
    image.save(output_path)
    return output_path

def page_image_upload():
    """First page for image upload."""
    st.title("Image Upload")
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Save image to session state
        st.session_state.input_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.image_np = np.array(st.session_state.input_image)
        
        # Display uploaded image
        st.image(st.session_state.input_image, caption="Uploaded Image", use_column_width=True)
        
        # Proceed button
        if st.button("Generate Masks"):
            st.session_state.page = 2
            st.rerun()

def page_mask_generation():
    """Second page for mask generation and selection."""
    st.title("Mask Generation")
    
    # Generate masks
    st.write("Generating masks...")
    mask_generator = load_sam_model(MODEL_CHECKPOINT_PATH)
    st.session_state.filtered_masks, st.session_state.mask_overlay, st.session_state.mask_labels = generate_masks(
        st.session_state.image_np, 
        mask_generator
    )
    
    # Display mask overlay
    st.image(st.session_state.mask_overlay, caption="Mask Overlay", use_column_width=True)
    st.write(f"Generated {len(st.session_state.mask_labels)} masks.")
    
    # Mask selection
    selected_indices = st.multiselect(
        "Select mask indices (e.g., 1, 2, 3):", 
        [str(idx) for idx, _ in st.session_state.mask_labels]
    )
    
    if st.button("Create Combined Mask"):
        # Create combined mask
        st.session_state.combined_mask_image = create_combined_mask(
            st.session_state.filtered_masks, 
            selected_indices
        )
        st.image(st.session_state.combined_mask_image, caption="Combined Mask", use_column_width=True)
        
        # Proceed button
        if st.button("Go to Prompts"):
            st.session_state.page = 3 
            st.experimental_rerun() 
    
    # Back button
    if st.button("Back to Image Upload"):
        st.session_state.page = 1
        st.rerun()

def page_prompt_input():
    """Third page for prompt input."""
    st.title("Inpainting Prompts")
    
    # Prompt inputs
    prompt = st.text_input("Enter the prompt:", "Add a cat with black color")
    negative_prompt = st.text_input("Enter negative prompt:", "artifacts, spots, glow, light effects")
    
    if st.button("Run Inpainting"):
        # Load inpainting pipeline
        pipe = load_inpainting_pipeline()
        
        # Run inpainting
        st.session_state.result = run_inpainting(
            pipe, 
            st.session_state.input_image, 
            st.session_state.combined_mask_image, 
            prompt, 
            negative_prompt
        )
        
        # Save result
        save_image(st.session_state.result, "final_output_image.png")
        
        # Move to results page
        st.session_state.page = 4
        st.rerun()
    
    # Back button
    if st.button("Back to Mask Selection"):
        st.session_state.page = 2
        st.rerun()

def page_results():
    """Fourth page for displaying results."""
    st.title("Inpainting Results")
    
    # Display original image
    st.subheader("Original Image")
    st.image(st.session_state.input_image, caption="Input Image", use_column_width=True)
    
    # Display combined mask
    st.subheader("Combined Mask")
    st.image(st.session_state.combined_mask_image, caption="Combined Mask", use_column_width=True)
    
    # Display result
    st.subheader("Inpainted Image")
    st.image(st.session_state.result, caption="Inpainted Result", use_column_width=True)
    
    # Buttons to restart or go back
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Over"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.page = 1
            st.rerun()
    
    with col2:
        if st.button("Back to Prompts"):
            st.session_state.page = 3
            st.rerun()

def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 1
    
    # Page routing
    if st.session_state.page == 1:
        page_image_upload()
    elif st.session_state.page == 2:
        page_mask_generation()
    elif st.session_state.page == 3:
        page_prompt_input()
    elif st.session_state.page == 4:
        page_results()

if __name__ == "__main__":
    main()
