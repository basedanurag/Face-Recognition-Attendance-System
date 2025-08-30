# Face Recognition Attendance System

A modern attendance management system that uses facial recognition technology to automate the attendance tracking process. This system is built with Python and leverages GPU acceleration for improved performance.

## Features

- 🎯 Real-time face detection and recognition
- 💻 GPU-optimized performance for faster processing
- 👥 Easy student registration with photo capture
- 📊 Automatic attendance tracking and CSV export
- 📧 Email functionality for attendance reports
- 🔐 Password-protected administrative functions
- 🎨 Modern GUI interface with customizable themes

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (recommended for optimal performance)
- CUDA Toolkit
- cuDNN

## Required Libraries

```bash
opencv-contrib-python>=4.8.0
pillow>=10.1.0
pandas>=2.0.0
numpy>=1.26.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/basedanurag/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```

2. Install the required dependencies:
```bash
pip install -r requirements_gpu.txt
```

3. Run the GPU setup script:
```bash
python gpu_setup_enhanced.py
```

4. Launch the main application:
```bash
python main_enhanced_gpu_optimized.py
```

## Usage

### 1. Student Registration
- Click on "Take Images" to capture student photos
- Enter student ID and Name
- The system will capture multiple angles of the face
- Click "Save Profile" to store the data

### 2. Taking Attendance
- Click on "Take Attendance" to start the recognition process
- The system will automatically detect and recognize registered faces
- Attendance is saved with timestamp in CSV format

### 3. Managing Attendance Records
- Attendance records are stored in the `Attendance` folder
- CSV files are created with date stamps
- View attendance directly in the application
- Send attendance reports via email

### 4. Administrative Functions
- Change admin password
- Delete registration data
- Delete attendance records
- Remove registered images

## Project Structure

```
├── main_enhanced_gpu_optimized.py    # Main application file
├── gpu_setup_enhanced.py             # GPU configuration setup
├── requirements_gpu.txt              # Project dependencies
├── haarcascade_frontalface_default.xml # Face detection cascade file
├── background_image2.jpg             # GUI background
└── Folders
    ├── Attendance/                   # Attendance records
    ├── StudentDetails/               # Student information
    ├── TrainingImage/               # Training images
    └── TrainingImageLabel/          # Trained model data
```

## Features in Detail

1. **Face Detection**
   - Uses Haar Cascade Classifier
   - Real-time detection with GPU acceleration

2. **Face Recognition**
   - LBPH (Local Binary Pattern Histogram) Face Recognizer
   - Optimized for GPU processing

3. **Attendance Management**
   - Automatic CSV generation
   - Date and time tracking
   - Email integration for reports

4. **User Interface**
   - Modern Tkinter-based GUI
   - Intuitive controls
   - Real-time video feed
   - Attendance display table

## Security Features

- Password protection for administrative functions
- Secure storage of student data
- Protected training process

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV team for computer vision tools
- NVIDIA for GPU acceleration support
- Contributors and testers

## Support

For support, please open an issue in the GitHub repository or contact the maintainer at anuragsrivastava241529@gmail.com.

---
Created by Anurag Srivastava
Last Updated: August 2025
