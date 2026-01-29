#!/usr/bin/env python3
"""
Setup script for Stable Diffusion 1.5 on macOS with MPS acceleration.

This script:
1. Checks system requirements (Python, macOS, MPS availability)
2. Downloads and caches Stable Diffusion 1.5 models
3. Converts models to FP16 for optimal MPS performance
4. Verifies the setup with a test generation
5. Creates necessary directories

Run this before starting the API server.
"""

import sys
import platform
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step_num, total_steps, text):
    """Print a formatted step"""
    print(f"[{step_num}/{total_steps}] {text}...")


def check_python_version():
    """Verify Python version compatibility"""
    print_step(1, 6, "Checking Python version")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 9 or version.minor > 11:
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Recommended: Python 3.9, 3.10, or 3.11")
        print("   Continue at your own risk!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")


def check_macos():
    """Verify macOS version"""
    print_step(2, 6, "Checking macOS version")
    
    if platform.system() != "Darwin":
        print("❌ Not running on macOS")
        print("   MPS acceleration requires macOS 12.3+")
        sys.exit(1)
    
    macos_version = platform.mac_ver()[0]
    major_version = int(macos_version.split('.')[0])
    
    if major_version < 12:
        print(f"❌ macOS {macos_version} detected")
        print("   MPS requires macOS 12.3 (Monterey) or later")
        sys.exit(1)
    
    print(f"✅ macOS {macos_version} - Compatible")


def check_mps_availability():
    """Check if MPS (Metal Performance Shaders) is available"""
    print_step(3, 6, "Checking MPS (Metal) availability")
    
    try:
        import torch
        
        if not torch.backends.mps.is_available():
            print("❌ MPS not available")
            print("   Possible reasons:")
            print("   - macOS version too old (need 12.3+)")
            print("   - PyTorch not built with MPS support")
            print("   - Running on Intel Mac (MPS requires Apple Silicon)")
            print("\n   Attempting to continue with CPU (will be slower)...")
            return False
        
        if not torch.backends.mps.is_built():
            print("⚠️  MPS available but PyTorch not built with MPS support")
            print("   Try: pip install --upgrade torch torchvision")
            return False
        
        print("✅ MPS available and ready")
        
        # Get chip info
        try:
            chip_info = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            print(f"   Chip: {chip_info}")
        except:
            print("   Unable to detect chip model")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Run: pip install torch torchvision")
        sys.exit(1)


def create_directories():
    """Create necessary project directories"""
    print_step(4, 6, "Creating project directories")
    
    project_root = Path(__file__).resolve().parent
    
    directories = [
        project_root / "outputs",
        project_root / "uploads",
        project_root / "api",
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"   ✓ {directory.name}/")
    
    # Create __init__.py in api directory
    init_file = project_root / "api" / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"   ✓ api/__init__.py")
    
    print("✅ Directories ready")


def download_models(use_mps=True):
    """Download and cache Stable Diffusion models"""
    print_step(5, 6, "Downloading Stable Diffusion 1.5 models")
    print("   This may take 10-20 minutes on first run...")
    print("   Models will be cached in ~/.cache/huggingface/")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        
        print(f"\n   📥 Downloading from: {model_id}")
        print("   Components: VAE, UNet, Text Encoder, Tokenizer, Scheduler")
        
        # Download and convert to FP16
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        
        print("   ✓ Models downloaded and cached")
        
        # Move to device for testing
        device = "mps" if use_mps else "cpu"
        print(f"   ✓ Loading models to {device.upper()}...")
        
        pipe = pipe.to(device)
        
        print("   ✓ Models ready on device")
        print("\n✅ Model download complete")
        
        return pipe
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Ensure enough disk space (~10GB)")
        print("3. Try: pip install --upgrade diffusers transformers")
        print("4. Clear cache: rm -rf ~/.cache/huggingface/")
        sys.exit(1)


def test_generation(pipe, use_mps=True):
    """Run a test generation to verify setup"""
    print_step(6, 6, "Running test generation")
    print("   Generating test image (this may take 30-60 seconds)...")
    
    try:
        device = "mps" if use_mps else "cpu"
        
        # Simple test prompt
        test_prompt = "a small red cube on a white background, simple, minimalist"
        
        print(f"   Prompt: \"{test_prompt}\"")
        print(f"   Device: {device.upper()}")
        print(f"   Steps: 10 (reduced for testing)")
        
        # Generate with minimal steps for speed
        image = pipe(
            prompt=test_prompt,
            num_inference_steps=10,
            guidance_scale=7.5,
        ).images[0]
        
        # Save test image
        project_root = Path(__file__).resolve().parent
        output_dir = project_root / "outputs"
        test_output = output_dir / "test_generation.png"
        
        image.save(test_output)
        
        print(f"\n   ✅ Test generation successful!")
        print(f"   📁 Saved to: {test_output}")
        print(f"   🖼️  Size: {image.size[0]}x{image.size[1]}")
        
        return True
        
    except Exception as e:
        print(f"\n   ⚠️  Test generation failed: {e}")
        print("\n   This might be okay - the models are downloaded.")
        print("   Try running the server and testing via API.")
        return False


def print_next_steps():
    """Print instructions for next steps"""
    print_header("Setup Complete! 🎉")
    
    print("Next steps:")
    print("\n1. Start the server:")
    print("   uvicorn api.server:app --reload")
    print("\n2. Open API docs:")
    print("   http://localhost:8000/docs")
    print("\n3. Test with curl:")
    print('   curl -X POST "http://localhost:8000/txt2img" \\')
    print('     -F "prompt=a beautiful sunset over mountains" \\')
    print('     -F "steps=25" \\')
    print('     --output test.png')
    print("\n4. Check outputs:")
    print("   ls outputs/")
    
    print("\n" + "="*60)
    print("Need help? Check README.md for full documentation")
    print("="*60 + "\n")


def main():
    """Main setup routine"""
    print_header("Stable Diffusion 1.5 Setup for macOS")
    
    # Run checks
    check_python_version()
    check_macos()
    use_mps = check_mps_availability()
    
    # Create directories
    create_directories()
    
    # Download models
    pipe = download_models(use_mps=use_mps)
    
    # Test generation
    test_generation(pipe, use_mps=use_mps)
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("\nPlease report this issue with the full error message.")
        sys.exit(1)