#!/usr/bin/env python3
"""
Setup and Testing Script for Face Recognition Attendance System
This script helps diagnose and fix common issues with the attendance system
"""

import cv2
import os
import sys
import subprocess

def test_opencv_installation():
    """Test if OpenCV is properly installed with face recognition support"""
    print("Testing OpenCV installation...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test if face module is available
        if hasattr(cv2, 'face'):
            print("✓ OpenCV face module is available")
            
            # Test if LBPHFaceRecognizer is available
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                print("✓ LBPHFaceRecognizer is available")
                return True
            except AttributeError:
                print("✗ LBPHFaceRecognizer not found")
                return False
            except Exception as e:
                print(f"✗ Error creating LBPHFaceRecognizer: {e}")
                return False
        else:
            print("✗ OpenCV face module not found")
            return False
            
    except ImportError as e:
        print(f"✗ OpenCV not installed: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("\nTesting camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera opened successfully")
            cap.release()
            return True
        else:
            print("✗ Failed to open camera")
            return False
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return False

def check_required_files():
    """Check if required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        "haarcascade_frontalface_default.xml",
        "background_image2.jpg"
    ]
    
    all_files_present = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} missing")
            all_files_present = False
    
    return all_files_present

def create_required_directories():
    """Create required directories if they don't exist"""
    print("\nCreating required directories...")
    
    directories = [
        "TrainingImage",
        "TrainingImageLabel", 
        "StudentDetails",
        "Attendance"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    # Try to install from requirements.txt first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install from requirements.txt: {e}")
        print("Trying to install packages individually...")
        
        # Try installing individual packages
        packages = [
            "opencv-contrib-python",
            "pillow", 
            "pandas",
            "numpy"
        ]
        
        success = True
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade"])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                success = False
        
        return success

def download_haarcascade():
    """Download haarcascade file if missing"""
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        print("\nDownloading haarcascade_frontalface_default.xml...")
        
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
            print("✓ haarcascade_frontalface_default.xml downloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to download haarcascade file: {e}")
            print("Please download it manually from:")
            print("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
            return False
    else:
        print("✓ haarcascade_frontalface_default.xml already exists")
        return True

def main():
    """Main setup function"""
    print("Face Recognition Attendance System - Setup & Diagnostics")
    print("=" * 60)
    
    # Create directories
    create_required_directories()
    
    # Install requirements if requirements.txt exists
    if os.path.exists("requirements.txt"):
        install_requirements()
    
    # Download haarcascade if missing
    download_haarcascade()
    
    # Run tests
    print("\n" + "=" * 60)
    print("RUNNING DIAGNOSTICS")
    print("=" * 60)
    
    opencv_ok = test_opencv_installation()
    camera_ok = test_camera()
    files_ok = check_required_files()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if opencv_ok and camera_ok and files_ok:
        print("✓ All tests passed! The system should work properly.")
        print("\nYou can now run: python main.py")
    else:
        print("✗ Some issues were found:")
        if not opencv_ok:
            print("  - OpenCV installation issues")
            print("    Solution: pip install opencv-contrib-python")
        if not camera_ok:
            print("  - Camera issues")
            print("    Solution: Check if camera is connected and not used by other applications")
        if not files_ok:
            print("  - Missing required files")
            print("    Solution: Ensure all required files are in the same directory as main.py")

if __name__ == "__main__":
    main()
