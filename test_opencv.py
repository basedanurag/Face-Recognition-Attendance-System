import cv2
import sys

# Print OpenCV version
print("OpenCV version:", cv2.__version__)

# Test if VideoCapture exists
print("VideoCapture exists:", hasattr(cv2, 'VideoCapture'))

# Try to open camera
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Camera opened successfully!")
        cap.release()
    else:
        print("Failed to open camera")
except Exception as e:
    print("Error:", e)

# Print all attributes of cv2
print("\nAll cv2 attributes:")
print(dir(cv2)) 