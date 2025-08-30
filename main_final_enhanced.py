############################################# Enhanced Face Recognition System ################################################
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
            if confidence > 0.4:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
        return faces
    
    def detect_faces_haar(self, gray):
        """Traditional Haar cascade detection"""
        faces = self.haar_cascade.detectMultiScale(gray, 1.2, 3, minSize=(30, 30))
        return [(x, y, w, h, 1.0) for x, y, w, h in faces]
    
    def detect_faces(self, image):
        """Main face detection method"""
        if self.use_dnn:
            return self.detect_faces_dnn(image)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            return self.detect_faces_haar(gray)
    
    def assess_face_quality(self, face_img):
        """Assess the quality of detected face"""
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Calculate blur (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = gray.std()
        
        # Simple quality score (0-100)
        quality = 0
        if blur_score > 50:
            quality += 30
        if 40 < brightness < 220:
            quality += 30
        if contrast > 20:
            quality += 40
        
        return min(quality, 100)

# Initialize enhanced face detector
face_detector = EnhancedFaceDetector()

############################################# Enhanced Face Recognition ################################################

class EnhancedFaceRecognizer:
    def __init__(self):
        self.device = gpu_manager.get_device()
        # LBPH parameters for better recognition
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=90.0)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])
    
    def preprocess_image(self, image):
        """Enhanced image preprocessing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        gray = cv2.resize(gray, (150, 150))
        
        # Histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Slight Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return denoised
    
    def train(self, faces, labels):
        """Train the recognizer with enhanced preprocessing"""
        processed_faces = []
        for face in faces:
            enhanced_face = self.preprocess_image(face)
            processed_faces.append(enhanced_face)
        
        self.recognizer.train(processed_faces, np.array(labels))
    
    def predict(self, face):
        """Enhanced prediction with confidence"""
        enhanced_face = self.preprocess_image(face)
        label, confidence = self.recognizer.predict(enhanced_face)
        similarity = max(0, 100 - confidence)
        return label, confidence, similarity

# Initialize enhanced recognizer
face_recognizer = EnhancedFaceRecognizer()

############################################# Performance Monitor ################################################

class PerformanceMonitor:
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.gpu_usage = 0
        self.recognition_threshold = 85.0  # More generous threshold
    
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
                                f"Current threshold: {current_threshold}\nEnter new threshold (30-100):", 
                                initialvalue=current_threshold, minvalue=30.0, maxvalue=100.0)
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
            mess._show(title="Password Registered", message="New password successfully registered!!")
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
    master.geometry("450x200")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="#f0f0f0")
    
    lbl4 = tk.Label(master, text="Enter Old Password:", bg="#f0f0f0", font=("Helvetica", 12, " bold "))
    lbl4.place(x=10, y=20)
    global old
    old = tk.Entry(master, width=25, fg="black", relief="solid", font=("Helvetica", 12, " bold "), show="*")
    old.place(x=200, y=20)
    
    lbl5 = tk.Label(master, text="Enter New Password:", bg="#f0f0f0", font=("Helvetica", 12, " bold "))
    lbl5.place(x=10, y=60)
    global new
    new = tk.Entry(master, width=25, fg="black", relief="solid", font=("Helvetica", 12, " bold "), show="*")
    new.place(x=200, y=60)
    
    lbl6 = tk.Label(master, text="Confirm New Password:", bg="#f0f0f0", font=("Helvetica", 12, " bold "))
    lbl6.place(x=10, y=100)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief="solid", font=("Helvetica", 12, " bold "), show="*")
    nnew.place(x=200, y=100)
    
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="white", bg="#dc3545", height=1, width=12, font=("Helvetica", 11, " bold "))
    cancel.place(x=250, y=150)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="white", bg="#28a745", height=1, width=12, font=("Helvetica", 11, " bold "))
    save1.place(x=120, y=150)
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
            mess._show(title="Password Registered", message="New password successfully registered!!")
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
    res = "1) Capture Images  >>>  2) Train Model"
    message1.configure(text=res)

def clear2():
    txt2.delete(0, "end")
    res = "1) Capture Images  >>>  2) Train Model"
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
                
                if quality > 40:
                    sampleNum += 1
                    if quality > 70:
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
            
            cv2.imshow("Capturing Images - Press 'q' to stop", img)
            
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
        res = "✓ Model Trained Successfully"
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
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    
    try:
        for k in tv.get_children():
            tv.delete(k)
    except Exception as e:
        print(f"Treeview error: {e}")
        return
    
    exists3 = os.path.isfile("TrainingImageLabel\\Trainner.yml")
    if exists3:
        face_recognizer.recognizer.read("TrainingImageLabel\\Trainner.yml")
        print("✓ Model loaded successfully")
    else:
        mess._show(title="Data Missing", message="Please click on Train Model first!!")
        return
    
    exists1 = os.path.isfile("StudentDetails\\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\\StudentDetails.csv")
        print(f"✓ Student database loaded: {len(df)} records")
        print("Database columns:", df.columns.tolist())
        print("Sample data:", df.head().to_string())
    else:
        mess._show(title="Details Missing", message="Students details are missing, please check!")
        return
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess._show(title="Camera Error", message="Could not open camera")
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ["Id", "", "Name", "", "Date", "", "Time"]
    recognized_today = set()
    
    try:
        while True:
            ret, im = cam.read()
            if not ret:
                break
            
            performance_monitor.update_fps()
            
            faces = face_detector.detect_faces(im)
            
            for (x, y, w, h, confidence) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_img = im[y:y + h, x:x + w]
                quality = face_detector.assess_face_quality(face_img)
                
                if quality > 30:
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    serial, conf, similarity = face_recognizer.predict(gray_face)
                    
                    name = "Unknown"
                    threshold = performance_monitor.get_recognition_threshold()
                    
                    print(f"Recognition: serial={serial}, confidence={conf:.2f}, threshold={threshold}")
                    
                    if conf < threshold:
                        try:
                            # Debug: print what we are looking for
                            print(f"Looking for SERIAL NO. == {serial}")
                            
                            # Find by serial number
                            matching_rows = df[df["SERIAL NO."] == serial]
                            print(f"Matching rows: {len(matching_rows)}")
                            
                            if len(matching_rows) > 0:
                                name_col = matching_rows["NAME"].iloc[0]
                                id_col = matching_rows["ID"].iloc[0]
                                
                                print(f"Found: ID={id_col}, Name={name_col}")
                                
                                name = str(name_col)
                                student_id = str(id_col)
                                
                                if student_id not in recognized_today:
                                    ts = time.time()
                                    date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%I:%M:%S %p")
                                    attendance = [student_id, "", name, "", date, "", timeStamp]
                                    
                                    attendance_file = f"Attendance\\Attendance_{date}.csv"
                                    file_exists = os.path.isfile(attendance_file)
                                    
                                    with open(attendance_file, "a+") as csvFile:
                                        writer = csv.writer(csvFile)
                                        if not file_exists:
                                            writer.writerow(col_names)
                                        writer.writerow(attendance)
                                    
                                    recognized_today.add(student_id)
                                    
                                    tv.insert("", 0, text=student_id, values=(name, date, timeStamp))
                                    print(f"✓ Attendance marked for {name}")
                            else:
                                print(f"No matching record found for serial {serial}")
                                
                        except Exception as e:
                            print(f"Recognition error: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    info_text = f"{name} (Conf:{conf:.1f}/{threshold})"
                    cv2.putText(im, info_text, (x, y-10), font, 0.5, (255, 255, 255), 2)
                    cv2.putText(im, f"Quality: {quality:.0f}%", (x, y+h+20), font, 0.4, (0, 255, 0), 1)
            
            fps = performance_monitor.get_fps()
            gpu_usage = performance_monitor.get_gpu_usage()
            threshold = performance_monitor.get_recognition_threshold()
            cv2.putText(im, f"FPS: {fps} | GPU: {gpu_usage:.1f}% | Threshold: {threshold}", (10, 30), font, 0.5, (255, 255, 255), 2)
            cv2.putText(im, f"Recognized today: {len(recognized_today)}", (10, 60), font, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Attendance System - Press 'q' to stop", im)
            
            if cv2.waitKey(1) == ord("q"):
                break
    
    except Exception as e:
        print(f"Attendance tracking error: {e}")
        import traceback
        traceback.print_exc()
    finally:
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

# Main window with enhanced visuals
window = tk.Tk()
window.geometry("1600x750")
window.resizable(True, False)
window.title("🚀 Enhanced Face Recognition Attendance System")
window.configure(background="#2c3e50")

# Background image with glass effect overlay
try:
    bg_image = Image.open("background_image2.jpg")
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = tk.Label(window, image=bg_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Semi-transparent overlay for modern look
    overlay = tk.Frame(window, bg="#2c3e50")
    overlay.place(x=0, y=0, relwidth=1, relheight=1)
    overlay.configure(bg="#2c3e50")
    overlay.attributes("-alpha", 0.3)
    
except FileNotFoundError:
    print("Background image not found, using gradient background")

# Create gradient-like frames with shadows
frame1 = tk.Frame(window, bg="#ecf0f1", relief="raised", bd=3)
frame1.place(relx=0.05, rely=0.22, relwidth=0.42, relheight=0.70)

frame2 = tk.Frame(window, bg="#ecf0f1", relief="raised", bd=3)
frame2.place(relx=0.53, rely=0.22, relwidth=0.42, relheight=0.70)

# Modern title with shadow effect
title_bg = tk.Frame(window, bg="#34495e", relief="raised", bd=2)
title_bg.place(relx=0.05, rely=0.02, relwidth=0.90, relheight=0.15)

gpu_info = f" • RTX 2050 ENABLED" if gpu_manager.is_gpu_available() else " • CPU MODE"
message3 = tk.Label(title_bg, text=f"🚀 ENHANCED FACE RECOGNITION SYSTEM{gpu_info}", 
                   fg="#ecf0f1", bg="#34495e", font=("Helvetica", 20, " bold "))
message3.place(relx=0.5, rely=0.2, anchor="center")

# Subtitle
subtitle = tk.Label(title_bg, text="AI-Powered Attendance Management with GPU Acceleration", 
                   fg="#bdc3c7", bg="#34495e", font=("Helvetica", 12))
subtitle.place(relx=0.5, rely=0.5, anchor="center")

# Date and clock with modern styling
date_frame = tk.Frame(title_bg, bg="#3498db", relief="raised", bd=1)
date_frame.place(relx=0.05, rely=0.65, relwidth=0.25, relheight=0.25)

datef = tk.Label(date_frame, text=day + "-" + mont[month] + "-" + year, fg="white", bg="#3498db",
                font=("Helvetica", 12, " bold "))
datef.place(relx=0.5, rely=0.5, anchor="center")

clock_frame = tk.Frame(title_bg, bg="#e74c3c", relief="raised", bd=1)
clock_frame.place(relx=0.70, rely=0.65, relwidth=0.25, relheight=0.25)

clock = tk.Label(clock_frame, fg="white", bg="#e74c3c", font=("Helvetica", 10, " bold "))
clock.place(relx=0.5, rely=0.5, anchor="center")
tick()

# Modern frame headers with gradient effect
head1_frame = tk.Frame(frame1, bg="#27ae60", relief="raised", bd=2)
head1_frame.place(x=0, y=0, relwidth=1, height=50)

head1 = tk.Label(head1_frame, text="👥 REGISTERED PROFILES", 
                fg="white", bg="#27ae60", font=("Helvetica", 14, " bold "))
head1.place(relx=0.5, rely=0.5, anchor="center")

head2_frame = tk.Frame(frame2, bg="#8e44ad", relief="raised", bd=2)
head2_frame.place(x=0, y=0, relwidth=1, height=50)

head2 = tk.Label(head2_frame, text="➕ NEW REGISTRATION", 
                fg="white", bg="#8e44ad", font=("Helvetica", 14, " bold "))
head2.place(relx=0.5, rely=0.5, anchor="center")

# Modern input fields
input_frame = tk.Frame(frame2, bg="#ecf0f1")
input_frame.place(x=20, y=70, relwidth=0.9, height=120)

lbl = tk.Label(input_frame, text="🆔 Enter ID:", fg="#2c3e50", bg="#ecf0f1", font=("Helvetica", 12, " bold "))
lbl.place(x=10, y=10)

txt = tk.Entry(input_frame, width=30, fg="#2c3e50", bg="white", font=("Helvetica", 12), relief="solid", bd=2)
txt.place(x=120, y=10)

lbl2 = tk.Label(input_frame, text="👤 Enter Name:", fg="#2c3e50", bg="#ecf0f1", font=("Helvetica", 12, " bold "))
lbl2.place(x=10, y=50)

txt2 = tk.Entry(input_frame, width=30, fg="#2c3e50", bg="white", font=("Helvetica", 12), relief="solid", bd=2)
txt2.place(x=120, y=50)

# Clear buttons with modern styling
clearButton = tk.Button(input_frame, text="🗑️", command=clear, fg="white", bg="#e67e22", width=3, 
                       height=1, font=("Helvetica", 10, " bold "), relief="raised")
clearButton.place(x=400, y=10)

clearButton2 = tk.Button(input_frame, text="🗑️", command=clear2, fg="white", bg="#e67e22", width=3, 
                        height=1, font=("Helvetica", 10, " bold "), relief="raised")
clearButton2.place(x=400, y=50)

# Status messages with modern styling
status_frame = tk.Frame(frame2, bg="#ecf0f1")
status_frame.place(x=20, y=200, relwidth=0.9, height=80)

message1 = tk.Label(status_frame, text="1) Capture Images  >>>  2) Train Model", bg="#ecf0f1", fg="#2c3e50", 
                   font=("Helvetica", 11, " bold "))
message1.place(x=10, y=10)

message = tk.Label(status_frame, text="", bg="#ecf0f1", fg="#27ae60", font=("Helvetica", 11, " bold "))
message.place(x=10, y=40)

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
message.configure(text=f"📊 Total Registrations: {res}")

# Modern buttons with gradient effect
button_frame = tk.Frame(frame2, bg="#ecf0f1")
button_frame.place(x=20, y=300, relwidth=0.9, height=150)

takeImg = tk.Button(button_frame, text="📸 CAPTURE FACES", command=TakeImages, fg="white", bg="#3498db", 
                   width=35, height=2, font=("Helvetica", 12, " bold "), relief="raised", bd=3)
takeImg.place(x=10, y=10)

trainImg = tk.Button(button_frame, text="🧠 TRAIN MODEL", command=psw, fg="white", bg="#9b59b6", 
                    width=35, height=2, font=("Helvetica", 12, " bold "), relief="raised", bd=3)
trainImg.place(x=10, y=70)

# Enhanced Treeview with modern styling
tree_frame = tk.Frame(frame1, bg="#ecf0f1")
tree_frame.place(x=10, y=110, relwidth=0.95, height=280)

# Configure treeview style
style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview", background="#ffffff", foreground="#2c3e50", fieldbackground="#ffffff")
style.configure("Treeview.Heading", background="#34495e", foreground="white", font=("Helvetica", 10, "bold"))

tv = ttk.Treeview(tree_frame, height=12, columns=("name", "date", "time"))
tv.column("#0", width=80)
tv.column("name", width=140)
tv.column("date", width=120)
tv.column("time", width=120)
tv.place(x=0, y=0, relwidth=0.9, relheight=1)

tv.heading("#0", text="ID")
tv.heading("name", text="NAME")
tv.heading("date", text="DATE")
tv.heading("time", text="TIME")

# Modern scrollbars
scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tv.yview)
scroll.place(relx=0.9, y=0, relheight=1)
tv.configure(yscrollcommand=scroll.set)

# Control buttons with modern styling
control_frame = tk.Frame(frame1, bg="#ecf0f1")
control_frame.place(x=10, y=60, relwidth=0.95, height=40)

trackImg = tk.Button(control_frame, text="🚀 START ATTENDANCE", command=TrackImages, fg="white", bg="#27ae60", 
                    width=20, height=1, font=("Helvetica", 11, " bold "), relief="raised", bd=2)
trackImg.place(x=10, y=5)

thresholdButton = tk.Button(control_frame, text="⚙️ SETTINGS", command=adjust_threshold, fg="white", bg="#f39c12", 
                           width=15, height=1, font=("Helvetica", 11, " bold "), relief="raised", bd=2)
thresholdButton.place(x=250, y=5)

# Control panel at bottom
bottom_frame = tk.Frame(frame1, bg="#ecf0f1")
bottom_frame.place(x=10, y=400, relwidth=0.95, height=80)

# Menu bar with modern styling
menubar = tk.Menu(window, relief="flat", bg="#34495e", fg="white", font=("Helvetica", 10))
filemenu = tk.Menu(menubar, tearoff=0, bg="#34495e", fg="white")
filemenu.add_command(label="🔐 Change Password", command=change_pass)
filemenu.add_command(label="⚙️ Adjust Threshold", command=adjust_threshold)
filemenu.add_command(label="📞 Contact Us", command=contact)
filemenu.add_separator()
filemenu.add_command(label="❌ Exit", command=window.destroy)
menubar.add_cascade(label="Settings", menu=filemenu)

# Email functionality
email_frame = tk.Frame(bottom_frame, bg="#ecf0f1")
email_frame.place(x=0, y=0, relwidth=1, height=80)

email_label = tk.Label(email_frame, text="📧 Email Reports:", fg="#2c3e50", bg="#ecf0f1", font=("Helvetica", 10, "bold"))
email_label.place(x=10, y=10)

recipient_email_entry = tk.Entry(email_frame, width=20, fg="#2c3e50", bg="white", font=("Helvetica", 9))
recipient_email_entry.place(x=10, y=35)

# Email domains and send functionality (simplified for space)
email_domains = ["gmail.com", "yahoo.com", "hotmail.com"]

def send_email():
    recipient_email = recipient_email_entry.get()
    if not recipient_email:
        mess._show(title="Error", message="Please enter a recipient email address.")
        return
    
    if "@" not in recipient_email:
        recipient_email += "@gmail.com"  # Default domain
    
    from_email = "anuragsrivastava241529@gmail.com"
    password = "tvwj lmke pnjv ficd"
    
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = recipient_email
        msg["Subject"] = f"Enhanced Attendance Report - {date}"
        
        threshold = performance_monitor.get_recognition_threshold()
        body = f"Attendance report generated by Enhanced Face Recognition System.\\n\\nSystem Info:\\n- GPU: {gpu_manager.gpu_name if gpu_manager.is_gpu_available() else 'CPU Mode'}\\n- Recognition Threshold: {threshold}"
        msg.attach(MIMEText(body, "plain"))
        
        filename = f"Attendance\\Attendance_{date}.csv"
        if os.path.exists(filename):
            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {filename}")
            msg.attach(part)
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, recipient_email, text)
        server.quit()
        mess._show(title="Success", message="Attendance report sent successfully!")
        
    except Exception as e:
        mess._show(title="Error", message=f"Failed to send email: {str(e)}")

send_email_button = tk.Button(email_frame, text="📤 SEND", command=send_email, fg="white", bg="#3498db", 
                             width=8, height=1, font=("Helvetica", 9, " bold "))
send_email_button.place(x=200, y=32)

# Data management buttons
def delete_all_data():
    result = mess.askyesno("Confirm Delete", "Are you sure you want to delete ALL data? This cannot be undone!")
    if result:
        try:
            # Delete CSV files
            if os.path.exists("StudentDetails\\StudentDetails.csv"):
                os.remove("StudentDetails\\StudentDetails.csv")
            
            # Delete training images
            if os.path.exists("TrainingImage"):
                for file in os.listdir("TrainingImage"):
                    os.remove(os.path.join("TrainingImage", file))
            
            # Delete training model
            if os.path.exists("TrainingImageLabel\\Trainner.yml"):
                os.remove("TrainingImageLabel\\Trainner.yml")
            
            mess.showinfo("Success", "All data deleted successfully!")
            
            # Update display
            message.configure(text="📊 Total Registrations: 0")
            for k in tv.get_children():
                tv.delete(k)
                
        except Exception as e:
            mess.showerror("Error", f"Failed to delete data: {str(e)}")

delete_all_button = tk.Button(email_frame, text="🗑️ RESET ALL", command=delete_all_data, fg="white", bg="#e74c3c", 
                             width=12, height=1, font=("Helvetica", 9, " bold "))
delete_all_button.place(x=300, y=32)

############################################# Run Application ################################################

window.configure(menu=menubar)
window.mainloop()
