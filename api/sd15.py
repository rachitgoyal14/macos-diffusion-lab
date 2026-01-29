import torch
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from pathlib import Path
from PIL import Image
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Pipelines (lazy-loaded)
txt2img_pipe = None
img2img_pipe = None
inpaint_pipe = None


# ---------- LOADERS ----------
SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

def load_sd15_txt2img():
    global txt2img_pipe
    if txt2img_pipe is None:
        txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            SD15_REPO,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("mps")
    return txt2img_pipe


def load_sd15_img2img():
    global img2img_pipe
    if img2img_pipe is None:
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD15_REPO,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("mps")
    return img2img_pipe


def load_sd15_inpaint():
    global inpaint_pipe
    if inpaint_pipe is None:
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD15_REPO,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("mps")
    return inpaint_pipe

# ---------- HELPERS ----------

def auto_background_mask(image: Image.Image, threshold: int = 180):
    """
    Auto background mask using brightness.
    White = editable, Black = protected.
    """
    gray = image.convert("L")
    gray_np = np.array(gray)
    mask_np = np.where(gray_np > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_np).convert("RGB")


# ---------- TXT2IMG ----------

def run_sd15_txt2img(
    prompt: str,
    steps: int = 25,
    guidance_scale: float = 7.5,
):
    pipe = load_sd15_txt2img()

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]

    out = OUTPUT_DIR / f"sd15_txt2img_{uuid.uuid4().hex}.png"
    image.save(out)
    return out


# ---------- IMG2IMG ----------

def run_sd15_img2img(
    prompt: str,
    init_image_path: Path,
    strength: float = 0.6,
    steps: int = 25,
    guidance_scale: float = 7.5,
):
    pipe = load_sd15_img2img()

    init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))

    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]

    out = OUTPUT_DIR / f"sd15_img2img_{uuid.uuid4().hex}.png"
    image.save(out)
    return out


# ---------- INPAINT ----------

def run_sd15_inpaint(
    prompt: str,
    image_path: Path,
    mode: str = "background",   # "background" | "full"
    steps: int = 25,
    guidance_scale: float = 7.5,
):
    pipe = load_sd15_inpaint()

    image = Image.open(image_path).convert("RGB").resize((512, 512))

    if mode == "full":
        mask = Image.fromarray(
            np.ones((512, 512), dtype=np.uint8) * 255
        ).convert("RGB")
    else:
        mask = auto_background_mask(image)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]

    out = OUTPUT_DIR / f"sd15_inpaint_{uuid.uuid4().hex}.png"
    result.save(out)
    return out
