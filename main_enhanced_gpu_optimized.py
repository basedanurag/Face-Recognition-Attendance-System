############################################# GPU-Enhanced Face Recognition System ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
from tkinter import PhotoImage
from PIL import Image, ImageTk
import tkinter.simpledialog as tsd
import cv2, os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import threading
import warnings
warnings.filterwarnings("ignore")

# GPU acceleration imports
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as T

# Email imports
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

############################################# GPU Configuration ################################################

class GPUManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**2
            print(f"🚀 Using GPU: {self.gpu_name} ({self.gpu_memory} MB)")
        else:
            print("Using CPU for processing")
    
    def get_device(self):
        return self.device
    
    def is_gpu_available(self):
        return self.gpu_available

# Initialize GPU manager
gpu_manager = GPUManager()

############################################# Enhanced Face Detection ################################################

class EnhancedFaceDetector:
    def __init__(self):
        self.device = gpu_manager.get_device()
        self.gpu_available = gpu_manager.is_gpu_available()
        
        # Load Haar cascade
        self.haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Load DNN model if available
        try:
            self.net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
            if self.gpu_available and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✓ Using GPU-accelerated DNN face detection")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("✓ Using CPU DNN face detection")
            self.use_dnn = True
        except:
            self.use_dnn = False
            print("✓ Using Haar cascade face detection")
    
    def detect_faces_dnn(self, image):
        """GPU-accelerated face detection using DNN"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Further lowered threshold for better detection
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Additional checks for face region validity
                width = x2-x1
                height = y2-y1
                
                # Filter out too small or too large faces
                if width > 30 and height > 30 and width < w*0.9 and height < h*0.9:
                    # Check aspect ratio
                    aspect_ratio = width / height
                    if 0.5 < aspect_ratio < 2.0:  # Normal face aspect ratio range
                        faces.append((x1, y1, width, height, confidence))
        
        return faces
    
    def detect_faces_haar(self, gray):
        """Very permissive Haar cascade detection"""
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,  # More aggressive scaling
            minNeighbors=2,   # Minimum neighbors required (more permissive)
            minSize=(20, 20), # Smaller minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(x, y, w, h, 1.0) for x, y, w, h in faces]
    
    def detect_faces(self, image):
        """Main face detection method"""
        if self.use_dnn:
            return self.detect_faces_dnn(image)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            return self.detect_faces_haar(gray)
    
    def assess_face_quality(self, face_img):
        """Advanced face quality assessment"""
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        # Resize for consistent processing
        gray = cv2.resize(gray, (200, 200))
        
        # Calculate multiple blur metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        gaussian_blur = cv2.GaussianBlur(gray, (5,5), 0)
        blur_diff = np.abs(gray.astype(np.float32) - gaussian_blur.astype(np.float32)).mean()
        
        # Enhanced brightness analysis
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        dark_pixels = np.mean(gray < 30)
        bright_pixels = np.mean(gray > 225)
        
        # Advanced contrast analysis
        contrast = gray.std()
        local_contrast = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
        
        # Edge density for feature richness
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        # Calculate weighted quality score with more lenient thresholds
        quality = 40  # Start with a base quality score
        
        # Blur assessment (30 points)
        blur_score = min(30, (laplacian_var / 300) * 30)  # More lenient blur threshold
        if blur_diff > 3:  # More lenient blur difference check
            blur_score = max(blur_score, 15)
        quality += blur_score
        
        # Brightness assessment (20 points)
        if 20 < brightness < 230:  # Much wider brightness range
            quality += 15
            if brightness_std > 20:  # More lenient dynamic range
                quality += 5
        elif 10 < brightness < 245:  # Even wider acceptable range
            quality += 10
        if dark_pixels < 0.2 and bright_pixels < 0.2:  # More tolerant of extreme pixels
            quality += 5
        
        # Contrast and feature assessment (10 points)
        if contrast > 20:  # Lower contrast requirement
            quality += 5
        if local_contrast > 50:  # Lower local contrast requirement
            quality += 2.5
        if edge_density > 0.05:  # Lower feature presence requirement
            quality += 2.5
        
        return min(quality, 100)

# Initialize enhanced face detector
face_detector = EnhancedFaceDetector()

############################################# Enhanced Face Recognition ################################################

class EnhancedFaceRecognizer:
    def __init__(self):
        self.device = gpu_manager.get_device()
        # Very lenient LBPH parameters for maximum recognition
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=3,  # Increased radius for more general patterns
            neighbors=8,  # Standard number of neighbors
            grid_x=8,  # Standard grid
            grid_y=8,  # Standard grid
            threshold=200.0  # Very lenient threshold
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((200, 200)),  # Increased resolution
            transforms.ToTensor(),
        ])
    
    def preprocess_image(self, image):
        """Simple and effective preprocessing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Basic resize
        gray = cv2.resize(gray, (100, 100))  # Smaller size for faster processing
        
        # Simple histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Basic noise reduction
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray
    
    def train(self, faces, labels):
        """Train the recognizer with enhanced preprocessing and data augmentation"""
        processed_faces = []
        augmented_labels = []
        
        for face, label in zip(faces, labels):
            # Basic preprocessing
            enhanced_face = self.preprocess_image(face)
            processed_faces.append(enhanced_face)
            augmented_labels.append(label)
            
            # Data augmentation for more robust training
            # 1. Slightly brighter
            brighter = cv2.convertScaleAbs(enhanced_face, alpha=1.1, beta=10)
            processed_faces.append(brighter)
            augmented_labels.append(label)
            
            # 2. Slightly darker
            darker = cv2.convertScaleAbs(enhanced_face, alpha=0.9, beta=-10)
            processed_faces.append(darker)
            augmented_labels.append(label)
            
            # 3. Small rotations
            rows, cols = enhanced_face.shape
            for angle in [-5, 5]:  # Small rotation angles
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                rotated = cv2.warpAffine(enhanced_face, M, (cols, rows))
                processed_faces.append(rotated)
                augmented_labels.append(label)
        
        # Train with augmented dataset
        self.recognizer.train(processed_faces, np.array(augmented_labels))
        
        # Save training parameters
        self.recognizer.setThreshold(150)  # Set more lenient threshold
    
    def predict(self, face):
        """Enhanced prediction with confidence and multiple checks"""
        enhanced_face = self.preprocess_image(face)
        predictions = []
        
        # Make multiple predictions with slight variations
        faces_to_try = [
            enhanced_face,
            cv2.convertScaleAbs(enhanced_face, alpha=1.1, beta=10),  # Brighter
            cv2.convertScaleAbs(enhanced_face, alpha=0.9, beta=-10),  # Darker
        ]
        
        for test_face in faces_to_try:
            label, confidence = self.recognizer.predict(test_face)
            predictions.append((label, confidence))
        
        # Get the most common prediction
        labels = [p[0] for p in predictions]
        confidences = [p[1] for p in predictions]
        
        from collections import Counter
        most_common_label = Counter(labels).most_common(1)[0][0]
        best_confidence = min([conf for label, conf in predictions if label == most_common_label])
        
        # Convert OpenCV confidence to similarity score (lower is better in OpenCV)
        similarity = max(0, 100 - best_confidence)
        
        return most_common_label, best_confidence, similarity

# Initialize enhanced recognizer
face_recognizer = EnhancedFaceRecognizer()

############################################# Performance Monitor ################################################

class PerformanceMonitor:
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.gpu_usage = 0
        self.recognition_threshold = 80.0  # Adjustable threshold
    
    def update_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_fps(self):
        return self.current_fps
    
    def get_gpu_usage(self):
        if gpu_manager.is_gpu_available():
            try:
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                return 0
        return 0
    
    def set_recognition_threshold(self, threshold):
        self.recognition_threshold = threshold
    
    def get_recognition_threshold(self):
        return self.recognition_threshold

performance_monitor = PerformanceMonitor()

############################################# Utility Functions ################################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def tick():
    current_time = time.strftime("%I:%M:%S %p")
    fps = performance_monitor.get_fps()
    gpu_usage = performance_monitor.get_gpu_usage()
    threshold = performance_monitor.get_recognition_threshold()
    
    # Update clock and performance info
    clock.config(text=f"{current_time} | FPS: {fps} | GPU: {gpu_usage:.1f}% | Threshold: {threshold}")
    clock.after(1000, tick)

def contact():
    mess._show(title="Contact us", message="Please contact us on : 'anuragsrivastava241529@gmail.com' ")

def check_haarcascadefile():
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    exists = os.path.isfile(cascade_path)
    if exists:
        pass
    else:
        mess._show(title="Some file missing", message="Please make sure haarcascade_frontalface_default.xml is in the same folder as this script.")
        window.destroy()

def adjust_threshold():
    """Allow user to adjust recognition threshold"""
    current_threshold = performance_monitor.get_recognition_threshold()
    new_threshold = tsd.askfloat("Adjust Recognition Threshold", 
                                f"Current threshold: {current_threshold}\nEnter new threshold (20-100):", 
                                initialvalue=current_threshold, minvalue=20.0, maxvalue=100.0)
    if new_threshold:
        performance_monitor.set_recognition_threshold(new_threshold)
        mess._show(title="Threshold Updated", message=f"Recognition threshold set to {new_threshold}")

############################################# Password Functions ################################################

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\\psd.txt")
    if exists1:
        with open("TrainingImageLabel\\psd.txt", "r") as tf:
            key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring("Old Password not found", "Please enter a new password below", show="*")
        if new_pas == None:
            mess._show(title="No Password Entered", message="Password not set!! Please try again")
        else:
            with open("TrainingImageLabel\\psd.txt", "w") as tf:
                tf.write(new_pas)
            mess._show(title="Password Registered", message="New password was registered successfully!!")
            return
    
    op = old.get()
    newp = new.get()
    nnewp = nnew.get()
    
    if op == key:
        if newp == nnewp:
            with open("TrainingImageLabel\\psd.txt", "w") as txf:
                txf.write(newp)
            mess._show(title="Password Changed", message="Password changed successfully!!")
            master.destroy()
        else:
            mess._show(title="Error", message="Confirm new password again!!!")
            return
    else:
        mess._show(title="Wrong Password", message="Please enter correct old password.")
        return

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    
    lbl4 = tk.Label(master, text="    Enter Old Password", bg="white", font=("comic", 12, " bold "))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(master, width=25, fg="black", relief="solid", font=("comic", 12, " bold "), show="*")
    old.place(x=180, y=10)
    
    lbl5 = tk.Label(master, text="   Enter New Password", bg="white", font=("comic", 12, " bold "))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black", relief="solid", font=("comic", 12, " bold "), show="*")
    new.place(x=180, y=45)
    
    lbl6 = tk.Label(master, text="Confirm New Password", bg="white", font=("comic", 12, " bold "))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief="solid", font=("comic", 12, " bold "), show="*")
    nnew.place(x=180, y=80)
    
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25, activebackground="white", font=("comic", 10, " bold "))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#00fcca", height=1, width=25, activebackground="white", font=("comic", 10, " bold "))
    save1.place(x=10, y=120)
    master.mainloop()

def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\\psd.txt")
    if exists1:
        with open("TrainingImageLabel\\psd.txt", "r") as tf:
            key = tf.read()
    else:
        new_pas = tsd.askstring("Old Password not found", "Please enter a new password below", show="*")
        if new_pas == None:
            mess._show(title="No Password Entered", message="Password not set!! Please try again")
        else:
            with open("TrainingImageLabel\\psd.txt", "w") as tf:
                tf.write(new_pas)
            mess._show(title="Password Registered", message="New password was registered successfully!!")
            return
    
    password = tsd.askstring("Password", "Enter Password", show="*")
    if password == key:
        TrainImages()
    elif password == None:
        pass
    else:
        mess._show(title="Wrong Password", message="You have entered wrong password")

############################################# Image Capture Functions ################################################

def clear():
    txt.delete(0, "end")
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

def clear2():
    txt2.delete(0, "end")
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

def TakeImages():
    check_haarcascadefile()
    columns = ["SERIAL NO.", "", "ID", "", "NAME"]
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    
    serial = 0
    exists = os.path.isfile("StudentDetails\\StudentDetails.csv")
    if exists:
        with open("StudentDetails\\StudentDetails.csv", "r") as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
    else:
        with open("StudentDetails\\StudentDetails.csv", "a+") as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
    
    Id = txt.get().strip()
    name = txt2.get().strip()
    
    # Enhanced input validation
    if not Id:
        mess._show(title="Error", message="Please enter an ID")
        return
    if not name:
        mess._show(title="Error", message="Please enter a name")
        return
    if not (name.replace(" ", "").isalpha()):
        mess._show(title="Error", message="Name should contain only letters and spaces")
        return
    
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            mess._show(title="Camera Error", message="Could not open camera")
            return
        
        sampleNum = 0
        high_quality_samples = 0
        
        while True:
            ret, img = cam.read()
            if not ret:
                break
            
            # Update FPS counter
            performance_monitor.update_fps()
            
            # Detect faces using enhanced detector
            faces = face_detector.detect_faces(img)
            
            for (x, y, w, h, confidence) in faces:
                # Draw rectangle
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Extract face
                face_img = img[y:y + h, x:x + w]
                
                # Assess face quality
                quality = face_detector.assess_face_quality(face_img)
                
                # Lowered quality threshold for more samples
                if quality > 40:  # Lowered from 60
                    sampleNum += 1
                    if quality > 70:  # Lowered from 80
                        high_quality_samples += 1
                    
                    # Save the captured face
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    filename = f"TrainingImage\\{name}.{serial}.{Id}.{sampleNum}.jpg"
                    cv2.imwrite(filename, gray_face)
                
                # Display quality info
                cv2.putText(img, f"Quality: {quality:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display info
            info_text = f"Samples: {sampleNum}/100 | High Quality: {high_quality_samples} | FPS: {performance_monitor.get_fps()}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Taking Images - Press 'q' to stop", img)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif sampleNum >= 100:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        if sampleNum > 0:
            res = f"✓ {sampleNum} images captured for ID: {Id} ({high_quality_samples} high quality)"
            row = [serial, "", Id, "", name]
            with open("StudentDetails\\StudentDetails.csv", "a+") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            message1.configure(text=res)
        else:
            mess._show(title="No Images", message="No faces detected. Please try again with better lighting.")
    
    except Exception as e:
        mess._show(title="Error", message=f"An error occurred: {str(e)}")

############################################# Training Functions ################################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    
    faces, ID = getImagesAndLabels("TrainingImage")
    
    if len(faces) == 0:
        mess._show(title="No Data", message="No training images found. Please register someone first!")
        return
    
    print(f"Training with {len(faces)} face samples...")
    
    try:
        # Train the enhanced recognizer
        face_recognizer.train(faces, ID)
        face_recognizer.recognizer.save("TrainingImageLabel\\Trainner.yml")
        
        unique_ids = len(set(ID))
        res = "✓ Profile Saved Successfully"
        message1.configure(text=res)
        message.configure(text=f"Total Registrations: {unique_ids} people, {len(faces)} samples")
        
        print(f"Training completed: {unique_ids} people, {len(faces)} samples")
        
    except Exception as e:
        mess._show(title="Training Error", message=f"Training failed: {str(e)}")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".jpg")]
    faces = []
    Ids = []
    
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert("L")
            imageNp = np.array(pilImage, "uint8")
            filename = os.path.split(imagePath)[-1]
            parts = filename.split(".")
            
            if len(parts) >= 5:
                ID = int(parts[2])
                faces.append(imageNp)
                Ids.append(ID)
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
    
    return faces, Ids

############################################# Attendance Functions ################################################

def TrackImages():
    print("Starting attendance tracking...")
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    
    # Initialize variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ["Id", "", "Name", "", "Date", "", "Time"]
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    attendance_marked = False  # Flag to track if attendance has been marked
    
    # Clear treeview
    try:
        for k in tv.get_children():
            tv.delete(k)
    except Exception as e:
        print(f"Treeview error: {e}")
    
    # Load recognizer
    exists3 = os.path.isfile("TrainingImageLabel\\Trainner.yml")
    if exists3:
        face_recognizer.recognizer.read("TrainingImageLabel\\Trainner.yml")
    else:
        mess._show(title="Data Missing", message="Please click on Save Profile to reset data!!")
        return
    
    # Load student details
    exists1 = os.path.isfile("StudentDetails\\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\\StudentDetails.csv")
    else:
        mess._show(title="Details Missing", message="Students details are missing, please check!")
        return
    
    print("Initializing camera...")
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cam.isOpened():
            mess._show(title="Camera Error", message="Could not open camera")
            return
        
        # Configure camera
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        print("Camera initialized successfully")
        
        while True:
            ret, im = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Update FPS counter
            performance_monitor.update_fps()
            
            # Detect faces
            faces = face_detector.detect_faces(im)
            
            for (x, y, w, h, confidence) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract and predict face
                face_img = im[y:y + h, x:x + w]
                quality = face_detector.assess_face_quality(face_img)
                
                if quality > 30:  # Lowered quality threshold
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    serial, conf, similarity = face_recognizer.predict(gray_face)
                    
                    name = "Unknown"
                    try:
                        aa = df.loc[df["SERIAL NO."] == serial]["NAME"].values
                        ID = df.loc[df["SERIAL NO."] == serial]["ID"].values
                        if len(aa) > 0 and len(ID) > 0:
                            name = str(aa[0])
                            student_id = str(ID[0])
                            
                            # Show name and prompt
                            cv2.putText(im, name, (x, y-10), font, 0.7, (0, 255, 0), 2)
                            
                            if not attendance_marked:
                                cv2.putText(im, "Press ENTER to mark attendance", (10, 50), font, 0.7, (0, 255, 255), 2)
                                
                                # Check for ENTER key
                                if cv2.waitKey(1) & 0xFF == 13:  # ENTER key
                                    ts = time.time()
                                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%I:%M:%S %p")
                                    
                                    # Create/append to attendance file
                                    attendance_file = f"Attendance\\Attendance_{current_date}.csv"
                                    exists = os.path.isfile(attendance_file)
                                    
                                    if exists:
                                        # Check if already marked
                                        df_attendance = pd.read_csv(attendance_file)
                                        if not any(df_attendance["Id"].astype(str) == student_id):
                                            with open(attendance_file, "a", newline="") as csvFile:
                                                writer = csv.writer(csvFile)
                                                writer.writerow([student_id, "", name, "", current_date, "", timeStamp])
                                    else:
                                        with open(attendance_file, "w", newline="") as csvFile:
                                            writer = csv.writer(csvFile)
                                            writer.writerow(col_names)
                                            writer.writerow([student_id, "", name, "", current_date, "", timeStamp])
                                    
                                    # Update treeview
                                    tv.insert("", 0, text=student_id, values=(name, current_date, timeStamp))
                                    attendance_marked = True
                                    
                                    # Show success message
                                    cv2.putText(im, "Attendance Marked Successfully!", (10, 90), font, 0.7, (0, 255, 0), 2)
                            else:
                                cv2.putText(im, "Attendance already marked", (10, 50), font, 0.7, (255, 165, 0), 2)
                            
                    except Exception as e:
                        print(f"Recognition error: {e}")
                        continue
            
            # Display performance info
            fps = performance_monitor.get_fps()
            gpu_usage = performance_monitor.get_gpu_usage()
            cv2.putText(im, f"FPS: {fps} | GPU: {gpu_usage:.1f}%", (10, 30), font, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Attendance", im)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                cv2.destroyAllWindows()
                adjust_threshold()
                cv2.namedWindow("Attendance")
            
    except Exception as e:
        print(f"Error in attendance tracking: {e}")
        mess._show(title="Error", message=f"An error occurred: {str(e)}")
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()

############################################# GUI Setup ################################################

# Global variables
global key
key = ""

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
day, month, year = date.split("-")

mont = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December"
}

# Main window
window = tk.Tk()
window.geometry("1600x700")
window.resizable(True, False)
window.title("🚀 GPU-Enhanced Face Recognition Attendance System (Optimized)")
window.configure(background="#000000")

# Background image
try:
    bg_image = Image.open("background_image2.jpg")
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = tk.Label(window, image=bg_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1.05)
except FileNotFoundError:
    print("Background image not found, using default background")

# Frames
frame1 = tk.Frame(window, bg="#87CEEB")
frame1.place(relx=0.07, rely=0.17, relwidth=0.30, relheight=0.72)

frame2 = tk.Frame(window, bg="#87CEEB")
frame2.place(relx=0.63, rely=0.17, relwidth=0.30, relheight=0.72)

# Title
gpu_info = f" (RTX 2050 Enabled)" if gpu_manager.is_gpu_available() else " (CPU Mode)"
message3 = tk.Label(window, text=f"🚀 GPU-Enhanced Face Recognition System{gpu_info}", 
                   fg="white", bg="#000000", width=55, height=1, font=("comic", 24, " bold "))
message3.place(x=50, y=10)

# Date and clock
datef = tk.Label(window, text=day + "-" + mont[month] + "-" + year, fg="#000000", bg="white",
                width=20, font=("comic", 15, " bold "))
datef.pack(fill="both", expand=True)
datef.place(relx=0.30, rely=0.09)

clock = tk.Label(window, fg="#000000", bg="White", width=45, font=("comic", 10, " bold "))
clock.place(relx=0.40, rely=0.09)
tick()

# Frame headers
head2 = tk.Label(frame2, text="                       For New Registrations                       ", 
                fg="black", bg="#00fcca", font=("comic", 17, " bold "))
head2.place(x=0, y=-5)

head1 = tk.Label(frame1, text="                       For Already Registered                       ", 
                fg="black", bg="#00fcca", font=("comic", 17, " bold "))
head1.place(x=0, y=-5)

# Input fields
lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#87CEEB", font=("comic", 17, " bold "))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=("comic", 15, " bold "))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#87CEEB", font=("comic", 17, " bold "))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=("comic", 15, " bold "))
txt2.place(x=30, y=173)

# Status messages
message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="#87CEEB", fg="black", 
                   width=39, height=1, activebackground="#3ffc00", font=("comic", 15, " bold "))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#87CEEB", fg="black", width=39, height=1, 
                  activebackground="#3ffc00", font=("comic", 16, " bold "))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#87CEEB", height=1, font=("comic", 15, " bold "))
lbl3.place(x=100, y=125)

# Calculate registrations
res = 0
exists = os.path.isfile("StudentDetails\\StudentDetails.csv")
if exists:
    with open("StudentDetails\\StudentDetails.csv", "r") as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
else:
    res = 0
message.configure(text="Total Registrations: " + str(res))

# Menu bar
menubar = tk.Menu(window, relief="ridge")
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Change Password", command=change_pass)
filemenu.add_command(label="Adjust Threshold", command=adjust_threshold)
filemenu.add_command(label="Contact Us", command=contact)
filemenu.add_command(label="Exit", command=window.destroy)
menubar.add_cascade(label="Settings", font=("comic", 29, " bold "), menu=filemenu)

# Treeview attendance table
tv = ttk.Treeview(frame1, height=13, columns=("name", "date", "time"))
tv.column("#0", width=82)
tv.column("name", width=130)
tv.column("date", width=133)
tv.column("time", width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading("#0", text="ID")
tv.heading("name", text="NAME")
tv.heading("date", text="DATE")
tv.heading("time", text="TIME")

# Scrollbars
scroll = ttk.Scrollbar(frame1, orient="vertical", command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky="ns")
tv.configure(yscrollcommand=scroll.set)

scroll_x = ttk.Scrollbar(frame1, orient="horizontal", command=tv.xview)
scroll_x.grid(row=3, column=0, pady=(0, 20), padx=(0, 100), sticky="ew")
tv.configure(xscrollcommand=scroll_x.set)

# Buttons
clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ff7221", width=11, 
                       activebackground="white", font=("comic", 11, " bold "))
clearButton.place(x=335, y=86)

clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="#ff7221", width=11, 
                        activebackground="white", font=("comic", 11, " bold "))
clearButton2.place(x=335, y=172)

takeImg = tk.Button(frame2, text="🎥 Take Images (GPU)", command=TakeImages, fg="white", bg="#6d00fc", 
                   width=34, height=1, activebackground="white", font=("comic", 15, " bold "))
takeImg.place(x=30, y=300)

trainImg = tk.Button(frame2, text="🧠 Save Profile (Enhanced)", command=psw, fg="white", bg="#6d00fc", 
                    width=34, height=1, activebackground="white", font=("comic", 15, " bold "))
trainImg.place(x=30, y=380)

trackImg = tk.Button(frame1, text="🚀 Take Attendance", command=TrackImages, fg="black", bg="#3ffc00", 
                    width=13, height=1, activebackground="white", font=("comic", 12, " bold "))
trackImg.place(x=160, y=85)

# Add threshold adjustment button
thresholdButton = tk.Button(frame1, text="⚙️ Adjust Threshold", command=adjust_threshold, fg="black", bg="#ffcc00", 
                           width=13, height=1, activebackground="white", font=("comic", 8, " bold "))
thresholdButton.place(x=5, y=115)

quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="#eb4600", 
                      width=35, height=1, activebackground="white", font=("comic", 15, " bold "))
quitWindow.place(x=30, y=460)

############################################# Email Functions ################################################

email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "gnindia.dronacharya.info"]

def send_email():
    recipient_email = recipient_email_entry.get()
    selected_domain = domain_var.get()
    
    if not recipient_email:
        mess._show(title="Error", message="Please enter a recipient email address.")
        return
    
    recipient_email += "@" + selected_domain
    from_email = "anuragsrivastava241529@gmail.com"
    password = "tvwj lmke pnjv ficd"
    
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = recipient_email
    msg["Subject"] = f"GPU-Enhanced Attendance Report - {date} - {time.strftime('%I:%M:%S %p')}"
    
    threshold = performance_monitor.get_recognition_threshold()
    body = f"Please find attached the attendance report generated by GPU-Enhanced Face Recognition System.\\n\\nSystem Performance:\\n- GPU: {gpu_manager.gpu_name if gpu_manager.is_gpu_available() else 'CPU Mode'}\\n- Average FPS: {performance_monitor.get_fps()}\\n- Recognition Threshold: {threshold}"
    msg.attach(MIMEText(body, "plain"))
    
    filename = f"Attendance\\Attendance_{date}.csv"
    if os.path.exists(filename):
        with open(filename, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {filename}")
        msg.attach(part)
        
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(from_email, password)
            text = msg.as_string()
            server.sendmail(from_email, recipient_email, text)
            server.quit()
            mess._show(title="Success", message="Attendance report sent successfully!")
        except Exception as e:
            mess._show(title="Error", message=f"Failed to send email: {str(e)}")
    else:
        mess._show(title="Error", message="No attendance data found for today.")

# Email interface
recipient_email_label = tk.Label(frame1, text="Recipient's Email", width=31, fg="black", bg="pink",
                                font=("comic", 9, " bold "))
recipient_email_label.place(x=6, y=30)

recipient_email_entry = tk.Entry(frame1, width=20, fg="black", bg="#d3f0dc", font=("comic", 15, " bold "))
recipient_email_entry.place(x=5, y=50)

domain_label = tk.Label(frame1, text="Domain:", width=20, fg="black", bg="pink",
                       font=("comic", 9, " bold "))
domain_label.place(x=250, y=30)

domain_var = tk.StringVar(frame1)
domain_var.set(email_domains[0])
domain_dropdown = tk.OptionMenu(frame1, domain_var, *email_domains)
domain_dropdown.config(width=15, font=("comic", 9, " bold "))
domain_dropdown.place(x=250, y=50)

at_symbol_label = tk.Label(frame1, text="@", width=2, fg="black", bg="white",
                          font=("comic", 10, " bold "))
at_symbol_label.place(x=230, y=50)

send_email_button = tk.Button(frame1, text="📧 Send Report", command=send_email, fg="black", bg="sky blue", 
                             width=13, activebackground="white", font=("comic", 8, " bold "))
send_email_button.place(x=400, y=50)

############################################# Data Management Functions ################################################

def delete_registration_csv():
    registration_csv_path = "StudentDetails\\StudentDetails.csv"
    if os.path.exists(registration_csv_path):
        os.remove(registration_csv_path)
        mess.showinfo("Success", "Registration CSV file deleted successfully.")
    else:
        mess.showinfo("Error", "Registration CSV file not found.")

def delete_attendance_csv():
    today = datetime.datetime.now().strftime("%d-%m-%Y")
    attendance_csv_path = f"Attendance\\Attendance_{today}.csv"
    if os.path.exists(attendance_csv_path):
        os.remove(attendance_csv_path)
        mess.showinfo("Success", f"Attendance CSV file for {today} deleted successfully.")
    else:
        mess.showinfo("Error", f"Attendance CSV file for {today} not found.")

def delete_registered_images():
    folder_path = "TrainingImage/"
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                os.remove(file_path)
            except Exception as e:
                mess.showinfo("Error", f"Failed to delete {file}: {e}")
        mess.showinfo("Success", "Registered images deleted successfully.")
    else:
        mess.showinfo("Error", "TrainingImage folder not found.")

# Data management buttons
delete_registration_button = tk.Button(frame1, text="🗑️ Delete Registration", command=delete_registration_csv, 
                                     fg="white", bg="red", width=19, font=("comic", 8, "bold"))
delete_registration_button.place(x=160, y=115)

delete_attendance_button = tk.Button(frame1, text="🗑️ Delete Attendance", command=delete_attendance_csv, 
                                   fg="white", bg="red", width=19, font=("comic", 8, "bold"))
delete_attendance_button.place(x=320, y=85)

delete_images_button = tk.Button(frame1, text="🗑️ Delete Images", command=delete_registered_images, 
                                fg="white", bg="red", width=20, font=("comic", 8, "bold"))
delete_images_button.place(x=320, y=115)

############################################# Run Application ################################################

window.configure(menu=menubar)
window.mainloop()
