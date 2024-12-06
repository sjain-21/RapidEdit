from flask import Flask, render_template, request, redirect, url_for
from sam_utils import generate_sam_masks, stitch_selected_masks
from lcm_pipeline import run_inpainting
import os

app = Flask(__name__)

# Uploads folder configuration
app.config['UPLOAD_FOLDER'] = os.path.join("static", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Routes
@app.route("/", methods=["GET", "POST"])
def input_page():
    if request.method == "POST":
        # Save uploaded image
        image = request.files["image"]
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(image_path)

        # Get the user prompt
        prompt = request.form["prompt"]

        # Redirect to mask selection
        return redirect(url_for("mask_selection_page", image_path=image_path, prompt=prompt))
    return render_template("input.html")


@app.route("/mask_selection", methods=["GET", "POST"])
def mask_selection_page():
    image_path = request.args.get("image_path")
    prompt = request.args.get("prompt")

    if request.method == "POST":
        # Get selected masks from the form
        selected_masks = request.form.getlist("selected_masks")

        # Stitch selected masks
        stitched_mask_path = os.path.join(app.config["UPLOAD_FOLDER"], "stitched_mask.png")
        stitch_selected_masks(image_path, selected_masks, stitched_mask_path)

        # Redirect to the check page
        return redirect(url_for("check_page", image_path=image_path, mask_path=stitched_mask_path, prompt=prompt))

    # Generate masks and overlay
    masks, mask_overlay_path = generate_sam_masks(image_path)
    return render_template("mask_selection.html", image_path=image_path, mask_overlay=mask_overlay_path, masks=masks)


@app.route("/check", methods=["GET", "POST"])
def check_page():
    image_path = request.args.get("image_path")
    mask_path = request.args.get("mask_path")
    prompt = request.args.get("prompt")

    if request.method == "POST":
        # Update the prompt and proceed to the result page
        updated_prompt = request.form["prompt"]
        return redirect(url_for("result_page", image_path=image_path, mask_path=mask_path, prompt=updated_prompt))

    return render_template("check.html", image_path=image_path, mask_path=mask_path, prompt=prompt)


@app.route("/result", methods=["GET", "POST"])
def result_page():
    image_path = request.args.get("image_path")
    mask_path = request.args.get("mask_path")
    prompt = request.args.get("prompt")

    # Run the LCM pipeline
    output_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.png")
    run_inpainting(image_path, mask_path, prompt, output_image_path)

    if request.method == "POST":
        if "reuse" in request.form:
            # Reuse the output image for new editing
            return redirect(url_for("mask_selection_page", image_path=output_image_path, prompt=prompt))
        elif "restart" in request.form:
            # Restart the process
            return redirect(url_for("input_page"))

    return render_template("result.html", input_image=image_path, mask_image=mask_path, output_image=output_image_path, prompt=prompt)


if __name__ == "__main__":
    app.run(debug=True)
