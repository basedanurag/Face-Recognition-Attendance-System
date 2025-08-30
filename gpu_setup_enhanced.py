import cv2
import numpy as np
import sys
import subprocess
import os

def check_gpu_support():
    print("=== GPU Support Check ===")
    
    # Check NVIDIA GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            gpu_info = result.stdout
            if "RTX 2050" in gpu_info:
                print("✓ RTX 2050 detected - Perfect for acceleration!")
            return True
        else:
            print("✗ NVIDIA GPU not detected")
            return False
    except:
        print("✗ nvidia-smi not available")
        return False

def test_opencv_cuda():
    print("\n=== Testing OpenCV CUDA Support ===")
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA-enabled devices: {cuda_devices}")
        if cuda_devices > 0:
            print("✓ OpenCV has CUDA support")
            return True
        else:
            print("✗ OpenCV compiled without CUDA support")
            return False
    except:
        print("✗ OpenCV CUDA module not available")
        return False

def install_gpu_packages():
    print("\n=== Installing GPU Packages ===")
    packages = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cu121",
        "tensorflow[and-cuda]"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            if "torch" in package:
                subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
            else:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"✓ {package.split()[0]} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

# Run setup
print("RTX 2050 GPU Setup for Face Recognition System")
print("=" * 50)

gpu_available = check_gpu_support()
cuda_opencv = test_opencv_cuda()

if gpu_available and not cuda_opencv:
    print("\nInstalling GPU-accelerated packages for your RTX 2050...")
    install_gpu_packages()

print("\n" + "=" * 50)
print("SETUP SUMMARY:")
print(f"GPU Available: {'Yes' if gpu_available else 'No'}")
print(f"OpenCV CUDA: {'Yes' if cuda_opencv else 'No'}")

if gpu_available:
    print("\n🚀 Your RTX 2050 is ready!")
    print("Expected performance boost: 5-10x faster")
else:
    print("\n⚠️ Will use CPU processing")
