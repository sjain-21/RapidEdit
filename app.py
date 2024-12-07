import streamlit as st
import os
from sam_utils import generate_sam_masks, stitch_selected_masks
from lcm_pipeline import run_inpainting

# Uploads folder configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    st.set_page_config(page_title="RAPIDEdit")
    st.title("RAPIDEdit")

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

    # Upload Page
    if current_stage == 'upload':
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        prompt = st.text_input("Enter Prompt:")
        
        if uploaded_file is not None:
            # Save uploaded image
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.image(image_path)

            if st.button("Create Mask") and prompt:
                # Generate masks
                masks, mask_overlay_path = generate_sam_masks(image_path)
                
                # Update session state
                st.session_state.image_path = image_path
                st.session_state.mask_overlay_path = mask_overlay_path
                st.session_state.masks = masks
                st.session_state.prompt = prompt
                
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
            [f"Mask {mask['id']}" for mask in st.session_state.masks]
        )
        
        if st.button("Proceed"):
            # Convert selected masks to indices
            selected_mask_indices = [int(mask.split()[-1]) for mask in selected_masks]
            
            # Stitch masks
            stitched_mask_path = os.path.join(UPLOAD_FOLDER, "stitched_mask.png")
            stitch_selected_masks(st.session_state.image_path, selected_mask_indices, stitched_mask_path)
            
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
        
        if st.button("Update and Proceed"):
            # Update prompt
            st.session_state.prompt = updated_prompt
            
            # Update URL to result stage
            update_query_params("result", image=os.path.basename(st.session_state.image_path))
            st.rerun()

    # Result Page
    elif current_stage == 'result':
        st.header("Result")
        
        # Display prompt
        st.write(f"**Prompt:** {st.session_state.prompt}")
        
        # Run inpainting
        output_image_path = os.path.join(UPLOAD_FOLDER, "output.png")
        run_inpainting(st.session_state.image_path, st.session_state.stitched_mask_path, 
                       st.session_state.prompt, output_image_path)
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.image_path, caption="Input Image")
        with col2:
            st.image(output_image_path, caption="Output Image")
        
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