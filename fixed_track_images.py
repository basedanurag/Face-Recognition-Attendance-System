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
