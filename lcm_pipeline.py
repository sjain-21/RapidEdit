import torch
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers import DiffusionPipeline
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

# pipe = DiffusionPipeline.from_pretrained(
#     "timbrooks/instruct-pix2pix",
#     torch_dtype=torch.float16,
# ).to("cuda")

pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")



#pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("your_cool_model", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
# pipe.safety_checker = lambda images, clip_input: (images, False)

def run_inpainting(image_path, mask_path, prompt, output_path, seed = None):
    init_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB")

    # result = pipe(
    #     prompt=prompt,
    #     image=init_image,
    #     mask_image=mask_image,
    #     num_inference_steps=4,
    #     guidance_scale=4,
    # )
    seed = 0
    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)

    result = pipe(
        prompt = prompt,
        image = init_image,
        mask_image = mask_image,
        generator = generator,
        num_inference_steps = 4,
        guidance_scale =4
    )
    result = result.images[0]
    result.save(output_path)
