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
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=70.0)
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
        self.recognition_threshold = 70.0
    
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
    
    clock.config(text=f"{current_time} | FPS: {fps} | GPU: {gpu_usage:.1f}% | Confidence: {threshold}")
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
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    
    lbl4 = tk.Label(master, text=" Enter Old Password", bg="white", font=("Verdana", 12, " bold "))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(master, width=25, fg="black", relief="solid", font=("Verdana", 12, " bold "), show="*")
    old.place(x=180, y=10)
    
    lbl5 = tk.Label(master, text=" Enter New Password", bg="white", font=("Verdana", 12, " bold "))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black", relief="solid", font=("Verdana", 12, " bold "), show="*")
    new.place(x=180, y=45)
    
    lbl6 = tk.Label(master, text="Confirm New Password", bg="white", font=("Verdana", 12, " bold "))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief="solid", font=("Verdana", 12, " bold "), show="*")
    nnew.place(x=180, y=80)
    
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25, activebackground="white", font=("Verdana", 10, " bold "))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#00fcca", height=1, width=25, activebackground="white", font=("Verdana", 10, " bold "))
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
    else:
        mess._show(title="Data Missing", message="Please click on Save Profile to reset data!!")
        return
    
    exists1 = os.path.isfile("StudentDetails\\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\\StudentDetails.csv")
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
                    if conf < threshold:
                        try:
                            aa = df.loc[df["SERIAL NO."] == serial]["NAME"].values
                            ID = df.loc[df["SERIAL NO."] == serial]["ID"].values
                            if len(aa) > 0 and len(ID) > 0:
                                name = str(aa[0])
                                student_id = str(ID[0])
                                
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
                        except Exception as e:
                            print(f"Recognition error: {e}")
                    
                    info_text = f"{name} (Conf:{conf:.1f}/{threshold})"
                    cv2.putText(im, info_text, (x, y-10), font, 0.5, (255, 255, 255), 2)
                    cv2.putText(im, f"Match Quality: {quality:.0f}%", (x, y+h+20), font, 0.4, (0, 255, 0), 1)
            
            fps = performance_monitor.get_fps()
            gpu_usage = performance_monitor.get_gpu_usage()
            threshold = performance_monitor.get_recognition_threshold()
            cv2.putText(im, f"FPS: {fps} | GPU: {gpu_usage:.1f}% | Threshold: {threshold}", (10, 30), font, 0.5, (255, 255, 255), 2)
            cv2.putText(im, f"Recognized today: {len(recognized_today)}", (10, 60), font, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Taking Attendance - Press 'q' to stop", im)
            
            if cv2.waitKey(1) == ord("q"):
                break
    
    except Exception as e:
        print(f"Attendance tracking error: {e}")
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

# Main window
window = tk.Tk()
window.geometry("1600x700")
window.resizable(True, False)
window.title("🚀 Enhanced Face Recognition System")
window.configure(background="#333333")

# Background image
try:
    bg_image = Image.open("background_image2.jpg")
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = tk.Label(window, image=bg_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1.05)
except FileNotFoundError:
    print("Background image not found, using default background")

# Frames
frame1 = tk.Frame(window, bg="#f0f0f0", relief="ridge", bd=5)
frame1.place(relx=0.07, rely=0.20, relwidth=0.35, relheight=0.68)

frame2 = tk.Frame(window, bg="#f0f0f0", relief="ridge", bd=5)
frame2.place(relx=0.55, rely=0.20, relwidth=0.35, relheight=0.68)

# Title
gpu_info = f" (RTX 2050 Enabled)" if gpu_manager.is_gpu_available() else " (CPU Mode)"
message3 = tk.Label(window, text=f"🚀 Enhanced Face Recognition System{gpu_info}", 
                   fg="#ffffff", bg="#111111", width=70, height=2, font=("Helvetica", 22, " bold "))
message3.place(x=30, y=10)

# Date and clock
datef = tk.Label(window, text=day + "-" + mont[month] + "-" + year, fg="#ffffff", bg="#111111",
                width=20, font=("Helvetica", 16, " bold "))
datef.place(relx=0.37, rely=0.12)

clock = tk.Label(window, fg="#ffffff", bg="#111111", width=30, font=("Helvetica", 12, " bold "))
clock.place(relx=0.65, rely=0.12)
tick()

# Frame headers
head2 = tk.Label(frame2, text="           New Registrations           ", 
                fg="#000000", bg="#d9d9d9", font=("Helvetica", 16, " bold "))
head2.place(x=0, y=0, relwidth=1)

head1 = tk.Label(frame1, text="           Registered Profiles           ", 
                fg="#000000", bg="#d9d9d9", font=("Helvetica", 16, " bold "))
head1.place(x=0, y=0, relwidth=1)

# Input fields
lbl = tk.Label(frame2, text="Enter ID", fg="#111111", bg="#f0f0f0", font=("Helvetica", 14, " bold "))
lbl.place(x=10, y=50)

txt = tk.Entry(frame2, width=26, fg="#000000", font=("Helvetica", 14))
txt.place(x=150, y=50)

lbl2 = tk.Label(frame2, text="Enter Name", fg="#111111", bg="#f0f0f0", font=("Helvetica", 14, " bold "))
lbl2.place(x=10, y=100)

txt2 = tk.Entry(frame2, width=26, fg="#000000", font=("Helvetica", 14))
txt2.place(x=150, y=100)

# Status messages
message1 = tk.Label(frame2, text="1) Take Images  >>>  2) Save Profile", bg="#f0f0f0", fg="#111111", 
                   width=45, height=1, font=("Helvetica", 12, " bold "))
message1.place(x=10, y=150)

message = tk.Label(frame2, text="", bg="#f0f0f0", fg="#111111", width=45, height=1, 
                  font=("Helvetica", 12, " bold "))
message.place(x=10, y=200)

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

# Treeview attendance table
tv = ttk.Treeview(frame1, height=10, columns=("name", "date", "time"))
tv.column("#0", width=82)
tv.column("name", width=130)
tv.column("date", width=133)
tv.column("time", width=133)
tv.grid(row=2, column=0, padx=(10, 0), pady=(90, 10), columnspan=4)
tv.heading("#0", text="ID")
tv.heading("name", text="NAME")
tv.heading("date", text="DATE")
tv.heading("time", text="TIME")

# Scrollbars
scroll = ttk.Scrollbar(frame1, orient="vertical", command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 10), pady=(90, 10), sticky="ns")
tv.configure(yscrollcommand=scroll.set)

scroll_x = ttk.Scrollbar(frame1, orient="horizontal", command=tv.xview)
scroll_x.grid(row=3, column=0, pady=(0, 10), padx=(10, 0), sticky="ew", columnspan=4)
tv.configure(xscrollcommand=scroll_x.set)

# Buttons
clearButton = tk.Button(frame2, text="Clear", command=clear, fg="#111111", bg="#ffcc00", width=11, 
                       height=1, font=("Helvetica", 12, " bold "), relief="ridge")
clearButton.place(x=280, y=240)

clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="#111111", bg="#ffcc00", width=11, 
                        height=1, font=("Helvetica", 12, " bold "), relief="ridge")
clearButton2.place(x=280, y=280)

takeImg = tk.Button(frame2, text="🎥 Capture Faces", command=TakeImages, fg="#ffffff", bg="#336699", 
                   width=30, height=1, font=("Helvetica", 14, " bold "), relief="ridge")
takeImg.place(x=40, y=320)

trainImg = tk.Button(frame2, text="🧠 Train Model", command=psw, fg="#ffffff", bg="#336699", 
                    width=30, height=1, font=("Helvetica", 14, " bold "), relief="ridge")
trainImg.place(x=40, y=370)

trackImg = tk.Button(frame1, text="🚀 Start Attendance", command=TrackImages, fg="#ffffff", bg="#33cc33", 
                    width=20, height=1, font=("Helvetica", 14, " bold "), relief="ridge")
trackImg.place(x=60, y=45)

thresholdButton = tk.Button(frame1, text="⚙️ Adjust Threshold", command=adjust_threshold, fg="#ffffff", bg="#ff9900", 
                           width=20, height=1, font=("Helvetica", 12, " bold "), relief="ridge")
thresholdButton.place(x=60, y=370)

window.configure(menu=menubar)
window.mainloop()
