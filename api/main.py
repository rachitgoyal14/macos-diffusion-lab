from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid

from api.sd15 import (
    run_sd15_txt2img,
    run_sd15_img2img,
    run_sd15_inpaint,
)

app = FastAPI(
    title="Local Stable Diffusion 1.5 API",
    description="Local txt2img, img2img, and inpainting using Stable Diffusion 1.5 (MPS)",
    version="1.0.0",
)

# ---------- DIRECTORIES ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------- TXT2IMG ----------

@app.post("/txt2img")
def txt2img(
    prompt: str = Form(...),
    steps: int = Form(25),
    guidance_scale: float = Form(7.5),
):
    """
    Generate an image from text using Stable Diffusion 1.5
    """
    try:
        output_path = run_sd15_txt2img(
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
        )
        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- IMG2IMG ----------

@app.post("/img2img")
def img2img(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    strength: float = Form(0.6),
    steps: int = Form(25),
    guidance_scale: float = Form(7.5),
):
    """
    Modify an existing image using Stable Diffusion 1.5
    """
    try:
        temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{image.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        output_path = run_sd15_img2img(
            prompt=prompt,
            init_image_path=temp_path,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
        )

        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- INPAINT ----------

@app.post("/inpaint")
def inpaint(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mode: str = Form("background"),  # "background" or "full"
    steps: int = Form(25),
    guidance_scale: float = Form(7.5),
):
    """
    Inpaint an image using Stable Diffusion 1.5.
    - background: auto background mask
    - full: entire image is editable
    """
    if mode not in {"background", "full"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'background' or 'full'."
        )

    try:
        temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{image.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        output_path = run_sd15_inpaint(
            prompt=prompt,
            image_path=temp_path,
            mode=mode,
            steps=steps,
            guidance_scale=guidance_scale,
        )

        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
