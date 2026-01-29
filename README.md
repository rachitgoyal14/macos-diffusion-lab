# macOS Diffusion Lab

A local experimentation environment for running and testing various diffusion models on macOS with Metal Performance Shaders (MPS) acceleration. Currently supports Stable Diffusion 1.5 with easy extensibility for SDXL, SD 2.x, and other models.

**Built for Apple Silicon** | **Privacy-First** | **Fully Local** | **Production-Ready API**

## Overview

This repository serves as an experimental playground for running various diffusion models locally on macOS. The goal is to test, benchmark, and optimize different models (SD 1.5, SDXL, SD 2.x, custom fine-tunes) with a focus on:

- 🧪 **Model Experimentation**: Easy swapping between different diffusion models
- 🍎 **macOS Optimization**: Leveraging MPS for Apple Silicon performance
- 🔒 **Privacy**: Everything runs locally, no data leaves your machine
- ⚡ **Performance Tuning**: Testing various optimization techniques
- 🔧 **Developer-Friendly**: Clean API for integration into other projects

**Current Status**: Production-ready with SD 1.5. SDXL, SD 2.1, and ControlNet support coming soon.

## Features

- 🚀 **MPS Acceleration**: Optimized for Apple Silicon (M1/M2/M3/M4) using Metal Performance Shaders
- 🎨 **Multiple Workflows**: txt2img, img2img, and inpaint capabilities
- 🔄 **Model Flexibility**: Easy configuration to swap between different models
- 💾 **Smart Loading**: Lazy-loaded pipelines to conserve memory
- 🔧 **FP16 Optimization**: Half-precision for faster inference and reduced memory
- 📊 **Benchmarking Ready**: Built-in structure for performance testing
- 🌐 **REST API**: FastAPI endpoints for easy integration
- 📁 **Organized Outputs**: Automatic file management and versioning

## Supported Models & Roadmap

### ✅ Currently Supported
- **Stable Diffusion 1.5** (stable-diffusion-v1-5/stable-diffusion-v1-5)
  - Text-to-image
  - Image-to-image
  - Inpainting

### 🚧 Coming Soon
- **SDXL 1.0** - Higher quality, 1024x1024 output (requires 16GB+ RAM)
- **Stable Diffusion 2.1** - Improved prompt understanding
- **ControlNet** - Guided generation with edge detection, pose, depth
- **Custom Fine-tunes** - Support for community models from Hugging Face/CivitAI
- **LoRA Support** - Lightweight model adaptations
- **Upscaling Models** - Real-ESRGAN, SwinIR integration

### 📝 Experiment Ideas
- Benchmark different schedulers (DPM++, Euler, DDIM)
- Compare inference speeds across M1/M2/M3 chips
- Memory optimization techniques
- Custom model training pipelines
- Batch processing workflows

Want to contribute? Check the [Contributing](#contributing) section below!

## Requirements

### Hardware
- macOS 12.3+ (Monterey or later)
- Apple Silicon (M1/M2/M3) or AMD GPU recommended
- Minimum 8GB RAM (16GB+ recommended)
- ~10GB free disk space for models

### Software
- Python 3.9, 3.10, or 3.11
- Xcode Command Line Tools

## Installation

### 1. Install Xcode Command Line Tools (if not already installed)

```bash
xcode-select --install
```

### 2. Clone or Create Project Directory

```bash
mkdir sd-local-api
cd sd-local-api
```

### 3. Create Project Structure

```
sd-local-api/
├── api/
│   ├── __init__.py
│   ├── sd15.py          # Your first code file
│   └── server.py        # Your second code file
├── outputs/             # Generated images (auto-created)
├── uploads/             # Temporary uploads (auto-created)
├── setup_models.py      # Model setup script
├── requirements.txt
└── README.md
```

### 4. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies

Create `requirements.txt`:

```txt
torch>=2.0.0
torchvision
diffusers>=0.25.0
transformers>=4.35.0
accelerate>=0.25.0
safetensors>=0.4.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
Pillow>=10.0.0
numpy>=1.24.0
```

Install packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Setup Models

Run the setup script to download and cache models:

```bash
python setup_models.py
```

This will:
- Download Stable Diffusion 1.5 (~4GB)
- Convert models to FP16 for MPS
- Cache everything locally (~/.cache/huggingface/)
- Verify MPS availability

**Note**: First-time setup takes 10-20 minutes depending on internet speed.

## Usage

### Starting the Server

```bash
# Make sure you're in the project root and virtual environment is activated
source venv/bin/activate

# Start the FastAPI server
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API docs: `http://localhost:8000/docs`

### Endpoint Examples

#### 1. Text-to-Image (txt2img)

Generate images from text prompts:

```bash
curl -X POST "http://localhost:8000/txt2img" \
  -F "prompt=a serene mountain landscape at sunset, highly detailed, 4k" \
  -F "steps=25" \
  -F "guidance_scale=7.5" \
  --output generated.png
```

**Parameters:**
- `prompt` (string, required): Text description of desired image
- `steps` (int, default: 25): Number of denoising steps (15-50 recommended)
- `guidance_scale` (float, default: 7.5): How strictly to follow prompt (5-15 range)

#### 2. Image-to-Image (img2img)

Transform existing images:

```bash
curl -X POST "http://localhost:8000/img2img" \
  -F "prompt=same scene but in winter with snow" \
  -F "image=@input.jpg" \
  -F "strength=0.6" \
  -F "steps=25" \
  -F "guidance_scale=7.5" \
  --output transformed.png
```

**Parameters:**
- `prompt` (string, required): Description of desired transformation
- `image` (file, required): Input image
- `strength` (float, default: 0.6): How much to change (0.0-1.0, where 1.0 = complete change)
- `steps` (int, default: 25): Denoising steps
- `guidance_scale` (float, default: 7.5): Prompt adherence

#### 3. Inpainting (inpaint)

Fill in or modify parts of images:

```bash
curl -X POST "http://localhost:8000/inpaint" \
  -F "prompt=a red sports car" \
  -F "image=@scene.jpg" \
  -F "mode=background" \
  -F "steps=25" \
  -F "guidance_scale=7.5" \
  --output inpainted.png
```

**Parameters:**
- `prompt` (string, required): Description of what to paint
- `image` (file, required): Input image
- `mode` (string, default: "background"): Masking mode
  - `background`: Auto-detect and edit bright areas (threshold > 180)
  - `full`: Edit entire image
- `steps` (int, default: 25): Denoising steps
- `guidance_scale` (float, default: 7.5): Prompt adherence

### Python Client Example

```python
import requests

# Text-to-image
response = requests.post(
    "http://localhost:8000/txt2img",
    data={
        "prompt": "a cyberpunk city at night, neon lights, rain",
        "steps": 30,
        "guidance_scale": 8.0
    }
)

with open("output.png", "wb") as f:
    f.write(response.content)

# Image-to-image
with open("input.jpg", "rb") as img:
    response = requests.post(
        "http://localhost:8000/img2img",
        data={
            "prompt": "convert to oil painting style",
            "strength": 0.7,
            "steps": 25
        },
        files={"image": img}
    )

with open("styled.png", "wb") as f:
    f.write(response.content)
```

## Performance Tips

### Memory Optimization

1. **Close other applications** before running to free up RAM
2. **Reduce steps** (15-20) for faster generation at slight quality cost
3. **Monitor Activity Monitor** for memory pressure

### Speed Optimization

1. **First generation is slow** (~30-60s) due to model loading
2. **Subsequent generations are faster** (~10-20s) as models stay in memory
3. **Keep server running** between requests to avoid reload penalty
4. **Use smaller step counts** (20-25) for quicker results

### Quality Optimization

1. **Increase steps** (30-50) for higher quality
2. **Adjust guidance_scale**: 
   - Lower (5-7): More creative/diverse
   - Higher (10-15): More prompt-faithful
3. **Use detailed prompts** with style descriptors
4. **For img2img**: Start with strength 0.4-0.6, adjust as needed

## Troubleshooting

### MPS Not Available

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

If False:
- Ensure macOS 12.3+
- Update to latest PyTorch: `pip install --upgrade torch torchvision`
- Check Python version (3.9-3.11 supported)

### Out of Memory Errors

- Reduce image size (default is 512x512)
- Close other applications
- Restart the server
- Consider using CPU: Change `.to("mps")` to `.to("cpu")` in sd15.py (slower)

### Slow Generation

- First run downloads models (~4GB), be patient
- Subsequent runs should be 10-20s per image
- Check Activity Monitor for CPU/GPU usage
- Ensure no thermal throttling (keep Mac cool)

### Model Download Issues

```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python setup_models.py
```

### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

## Project Structure

```
macos-diffusion-lab/
├── api/
│   ├── __init__.py
│   ├── sd15.py              # SD 1.5 implementation
│   ├── sdxl.py              # SDXL (coming soon)
│   ├── controlnet.py        # ControlNet (coming soon)
│   └── server.py            # FastAPI routes
├── experiments/             # Benchmarking and testing scripts
│   ├── benchmark.py         # Performance testing
│   └── compare_models.py    # Model comparison
├── models/                  # Custom model configs (optional)
├── outputs/                 # Generated images
├── uploads/                 # Temporary uploads
├── tests/                   # Unit tests
├── setup_models.py          # Model setup and verification
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

**Note**: The `outputs/`, `uploads/`, and model cache directories are git-ignored to keep the repo lightweight.

## Advanced Configuration

### Using Different Models

Edit `SD15_REPO` in `api/sd15.py`:

```python
# Alternative models (Hugging Face model IDs)
SD15_REPO = "runwayml/stable-diffusion-v1-5"  # Original
# SD15_REPO = "stabilityai/stable-diffusion-2-1"  # SD 2.1
# SD15_REPO = "CompVis/stable-diffusion-v1-4"  # SD 1.4
```

### Custom Output Directory

In `api/sd15.py`, modify:

```python
OUTPUT_DIR = PROJECT_ROOT / "my_custom_outputs"
```

### Changing Default Port

```bash
uvicorn api.server:app --port 8080
```

## Recommended .gitignore

```gitignore
# Python
venv/
env/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Generated content
outputs/
uploads/
experiments/results/

# Model cache (stored in ~/.cache/huggingface/)
models/
*.ckpt
*.safetensors
*.pth

# IDE
.vscode/
.idea/
*.swp
*.swo
*.swn
.DS_Store

# Environment
.env
.env.local

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/
```

## Safety and Ethical Use

- This setup **disables the safety checker** for local use
- Be responsible with generated content
- Respect copyright and usage rights
- Don't generate harmful or illegal content
- Consider re-enabling safety checker for shared deployments

## Resources & Learning

- [Diffusers Documentation](https://huggingface.co/docs/diffusers/) - Official library docs
- [Stable Diffusion Guide](https://stable-diffusion-art.com/) - Techniques and tips
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - API framework
- [Apple MPS Documentation](https://developer.apple.com/metal/pytorch/) - Metal acceleration
- [Hugging Face Models](https://huggingface.co/models?pipeline_tag=text-to-image) - Browse models
- [r/StableDiffusion](https://reddit.com/r/StableDiffusion) - Community discussion

## Acknowledgments

This project is built on the shoulders of giants and inspired by the amazing open-source community:

- **Stability AI** for Stable Diffusion models
- **Hugging Face** for the Diffusers library and model hosting
- **Apple's [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)** - Apple's Core ML optimizations for Stable Diffusion on Apple Silicon provided valuable insights for MPS acceleration techniques
- **[Original Inspiration Repository]** - This project was forked/inspired by [REPO_LINK_HERE] for initial experimentation
- The broader **open-source AI community** for pushing the boundaries of what's possible

Special thanks to everyone contributing models, techniques, and knowledge to make local AI accessible to all.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Model Licenses:**
- Stable Diffusion 1.5: [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- SDXL: [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)

Always respect the license terms of models you use.

## Support & Community

- 🐛 **Found a bug?** [Open an issue](../../issues)
- 💡 **Have an idea?** [Start a discussion](../../discussions)
- 📊 **Want to share benchmarks?** Post your hardware specs and results in discussions
- ⭐ **Like this project?** Give it a star!
- 🤝 **Want to contribute?** Fork the repo and submit a PR

For questions about:
- **PyTorch/MPS**: [PyTorch Forums](https://discuss.pytorch.org/)
- **Diffusers**: [Hugging Face Forums](https://discuss.huggingface.co/)
- **Apple's Core ML approach**: [ml-stable-diffusion repo](https://github.com/apple/ml-stable-diffusion)

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{macos_diffusion_lab,
  title = {macOS Diffusion Lab: Local Diffusion Model Experimentation},
  author = {Rachit Goyal},
  year = {2026},
  url = {https://github.com/rachitgoyal14/macos-diffusion-lab}
}
```

---

**Built with ❤️ for local, private, and experimental AI image generation on macOS**

*Made by the community, for the community. Happy experimenting! 🧪🎨*# macos-diffusion-lab
