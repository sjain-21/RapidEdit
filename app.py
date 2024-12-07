from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image, ImageDraw
import random

app = Flask(__name__)

# Uploads folder configuration
app.config['UPLOAD_FOLDER'] = os.path.join("static", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def generate_dummy_masks(image_path):
    """
    Simulates generating masks for an image by creating dummy data.
    Returns a list of dummy masks and a path to the mask overlay image.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_np = image.copy()
    draw = ImageDraw.Draw(image_np)

    # Create dummy masks (bounding boxes for simplicity)
    masks = []
    for i in range(1, 6):  # Generate 5 dummy masks
        x1, y1 = random.randint(0, image.width // 2), random.randint(0, image.height // 2)
        x2, y2 = random.randint(x1 + 20, image.width), random.randint(y1 + 20, image.height)
        masks.append({"id": i, "bbox": (x1, y1, x2, y2)})
        draw.rectangle([x1, y1, x2, y2], outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)

    # Save the mask overlay image
    mask_overlay_path = os.path.join(os.path.dirname(image_path), "mask_overlay.png")
    image_np.save(mask_overlay_path)

    return masks, mask_overlay_path


def stitch_dummy_masks(image_path, selected_masks, save_path):
    """
    Simulates stitching selected masks into a single combined mask.
    """
    # Create a blank mask image
    image = Image.open(image_path).convert("RGB")
    combined_mask = Image.new("RGB", image.size, (0, 0, 0))
    draw = ImageDraw.Draw(combined_mask)

    # Add selected masks to the combined mask
    for mask_id in selected_masks:
        bbox = (random.randint(0, image.width // 2), random.randint(0, image.height // 2),
                random.randint(image.width // 2, image.width), random.randint(image.height // 2, image.height))
        draw.rectangle(bbox, fill=(255, 255, 255))

    # Save the combined mask
    combined_mask.save(save_path)

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
        print("Selected masks: ", selected_masks)

        # Stitch selected masks
        stitched_mask_path = os.path.join(app.config["UPLOAD_FOLDER"], "stitched_mask.png")
        stitch_dummy_masks(image_path, selected_masks, stitched_mask_path)

        # Redirect to the check page
        return redirect(url_for("check_page", image_path=image_path, mask_path=stitched_mask_path, prompt=prompt))

    # Generate dummy masks and overlay
    masks, mask_overlay_path = generate_dummy_masks(image_path)
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

    # Simulate output image
    output_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.png")
    Image.open(image_path).save(output_image_path)  # Just save the input image as the output for testing

    if request.method == "POST":
        if "reuse" in request.form:
            # Reuse the output image for new editing
            return redirect(url_for("mask_selection_page", image_path=output_image_path, prompt=prompt))
        elif "restart" in request.form:
            # Restart the process
            return redirect(url_for("input_page"))

    return render_template("result.html", input_image=image_path, output_image=output_image_path, prompt=prompt)


if __name__ == "__main__":
    app.run(debug=True)
