import os
# Ensure these paths are correct for your system
os.add_dll_directory(r'C:/opencv_build/bin/Release')
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin')
import cv2
import cv2.cuda as cv2_cuda
import mediapipe as mp
import numpy as np
import sys
from ultralytics import YOLOE
import torch
# Import necessary PyQt5 widgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
                             QProgressBar, QCheckBox, QTabWidget, QRadioButton,
                             QGroupBox, QLabel, QMessageBox, QSplashScreen, QDialog, QTextBrowser, QInputDialog)
from PyQt5.QtCore import (Qt, QSize, QTimer)
from PyQt5.QtGui import (QMovie, QPixmap, QIcon)
import time # For basic timing
import webbrowser
import re
import csv
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import markdown2
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# --- Constants ---
KERNEL = np.ones((9, 9), np.uint8)
YOLO_BATCH_SIZE = 48 # Adjust based on GPU memory (Consider reducing if OOM errors occur)

#checking date-time validity
def is_valid_date(date_str):
    """Check if the date matches YYYY/MM/DD format."""
    return re.match(r'^\d{4}/\d{2}/\d{2}$', date_str) is not None

def is_valid_time(time_str):
    """Check if the time matches HH:MM:SS format."""
    return re.match(r'^\d{2}:\d{2}:\d{2}$', time_str) is not None

# --- Driving Mode Detection Functions ---
# scale_rois (unchanged)
def scale_rois(rois, width, height):
    """Return ROI coordinates scaled to the current image size."""
    ref_width, ref_height = 944, 480
    area_scale = (width * height) / (ref_width * ref_height)
    scaled_rois = {}
    for roi_name, roi in rois.items():
        x_scale = width / ref_width
        y_scale = height / ref_height
        scaled_rois[roi_name] = {
            'x': int(roi['x'] * x_scale),
            'y': int(roi['y'] * y_scale),
            'width': int(roi['width'] * x_scale),
            'height': int(roi['height'] * y_scale),
            'min_area': int(roi['min_area'] * area_scale),
            'max_area': int(roi['max_area'] * area_scale),
            'circularity_min': roi['circularity_min'],
            'circularity_max': roi['circularity_max'],
            'aspect_ratio_min': roi['aspect_ratio_min'],
            'aspect_ratio_max': roi['aspect_ratio_max']
        }
    return scaled_rois

# check_green_steering_wheel (unchanged from previous fix)
def check_green_steering_wheel(gpu_hsv, roi, stream, morph_filter_close, morph_filter_dilate, debug_image=None, debug_mode=False, car_type='n'):
    """Checks a specific ROI on the GPU for green steering wheel like shapes, using streams."""
    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    # Use size() for GpuMat dimensions
    
    hsv_width, hsv_height = gpu_hsv.size()

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(hsv_width, x + w)
    y2 = min(hsv_height, y + h)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
      
      
    if clipped_w <= 0 or clipped_h <= 0:
        print(f"Invalid ROI after clipping: w={clipped_w}, h={clipped_h}")
        return False, 0

    try:
        gpu_roi = cv2.cuda.GpuMat(gpu_hsv, (x1, y1, clipped_w, clipped_h))
        # Define green color range
        if car_type == 's':
            green_lower = (0, 0, 200) #40,20,255
            green_upper = (115, 90, 255) #55,55,255
        
        elif car_type == '3':
            green_lower = (0,0,0)
            green_upper = (157,78,81)
        
        else:
            green_lower = (35, 40, 30)
            green_upper = (85, 255, 255)
        
        # Define white color range
        if car_type == 'v':
            white_lower = (0, 0, 200) #white_lower = (35, 40, 30)
            white_upper = (180, 30, 255) #white_upper = (85, 255, 255)
        
        elif car_type == '3':
            white_lower = (0,0,0)
            white_upper = (0,0,10)
            
        else:
            white_lower = (0, 0, 200)
            white_upper = (180, 30, 255)

        # Create and combine masks
        gpu_mask_green = cv2_cuda.inRange(gpu_roi, green_lower, green_upper, stream=stream)
        gpu_mask_white = cv2_cuda.inRange(gpu_roi, white_lower, white_upper, stream=stream)
        gpu_mask = cv2_cuda.bitwise_or(gpu_mask_green, gpu_mask_white, stream=stream)
        
        gpu_mask = morph_filter_close.apply(gpu_mask, stream=stream)
        gpu_mask = morph_filter_dilate.apply(gpu_mask, stream=stream)
        
        # countNonZero appears to return a scalar directly or a single-element GpuMat
        # If it returns GpuMat error, download it. If scalar, use directly after sync.
        
        count_gpu = cv2.cuda.countNonZero(gpu_mask, stream=stream) # Assuming returns scalar after sync
        mask_cpu = gpu_mask.download(stream=stream)

        stream.waitForCompletion()
        color_pixel_count = count_gpu # Assign after sync

        contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    except cv2.error as e:
        print(f"CUDA error in ROI processing: {e}")
        stream.waitForCompletion()
        return False, 0

    has_steering_wheel_shape = False
    circularity = 0.0  # Initialize with default values
    aspect_ratio = 0.0  # Initialize with default values
    area = 0.0
    if debug_mode:
        print(f"\nROI at ({x}, {y}): Found {len(contours)} contours, Green Pixels (GPU Count): {color_pixel_count}")

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < roi['min_area'] or area > roi['max_area']: continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        _, _, w_contour, h_contour = cv2.boundingRect(contour)
        aspect_ratio = float(w_contour) / h_contour if h_contour != 0 else 0
        if (roi['circularity_min'] < circularity < roi['circularity_max'] and
            roi['aspect_ratio_min'] < aspect_ratio < roi['aspect_ratio_max']):
            has_steering_wheel_shape = True
            break

    if debug_mode:
        print(f"\nROI at ({x}, {y}): Circularity - {circularity:.2f}, Aspect Ratio - {aspect_ratio:.2f}, State - {has_steering_wheel_shape}, Area - {area:.2f}")
        
    
    if debug_mode and debug_image is not None:
        try:
            print(f"Drawing ROI rectangle at ({x1}, {y1}) to ({x2}, {y2})")
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi_debug_view = debug_image[y1:y2, x1:x2]
            if mask_cpu is not None and roi_debug_view.shape[:2] == mask_cpu.shape[:2]:
                mask_rgb = cv2.cvtColor(mask_cpu, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(roi_debug_view, 0.7, mask_rgb, 0.3, 0)
                debug_image[y1:y2, x1:x2] = overlay
            for contour in contours:
                shifted_contour = contour + np.array([[x1, y1]])
                cv2.drawContours(debug_image, [shifted_contour], -1, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error in debug visualization: {e}")

    return has_steering_wheel_shape, color_pixel_count

# crop_gpu, crop_to_third_quadrant_gpu, crop_to_first_quadrant_gpu (unchanged)
def crop_gpu(gpu_image, x_start, y_start, x_end, y_end):
    # Use size() for GpuMat dimensions
    width, height = gpu_image.size()
    x_start_c = max(0, x_start)
    y_start_c = max(0, y_start)
    x_end_c = min(width, x_end)
    y_end_c = min(height, y_end)
    w = x_end_c - x_start_c
    h = y_end_c - y_start_c
    if w <= 0 or h <= 0:
        print(f"Warning: Invalid crop dimensions ({w}x{h}). Returning original image.")
        return gpu_image
    return cv2.cuda.GpuMat(gpu_image, (x_start_c, y_start_c, w, h))

def crop_to_third_quadrant_gpu(gpu_image):
    width, height = gpu_image.size()
    return crop_gpu(gpu_image, 0, height // 2, width // 2, height)

def crop_to_first_quadrant_gpu(gpu_image):
    width, height = gpu_image.size()
    return crop_gpu(gpu_image, width // 2, 0, width, height // 2)

# draw_grid (unchanged)
def draw_grid(image, grid_spacing=100):
    height, width = image.shape[:2]
    for x in range(0, width, grid_spacing): cv2.line(image, (x, 0), (x, height), (128, 128, 128), 1)
    for y in range(0, height, grid_spacing): cv2.line(image, (0, y), (width, y), (128, 128, 128), 1)


def process_image_for_mode(gpu_image_full, selected_rois, stream, morph_close, morph_dilate, debug_mode, car_type='n'):
    """Process the cropped third quadrant for mode detection and display debug visualization with HSV values on hover."""
    gpu_image_cropped = crop_to_third_quadrant_gpu(gpu_image_full)
    width, height = gpu_image_cropped.size()
    gpu_rgb = cv2_cuda.cvtColor(gpu_image_cropped, cv2.COLOR_BGR2RGB, stream=stream)
    gpu_hsv = cv2_cuda.cvtColor(gpu_rgb, cv2.COLOR_RGB2HSV, stream=stream)
    debug_image_cpu = None
    hsv_image_cpu = None
    debug_window_name = "Debug: Mode Detection (3rd Quadrant)"
    last_hsv_text = None  # Track last displayed HSV text to avoid flicker

    if debug_mode:
        try:
            debug_image_cpu = gpu_image_cropped.download(stream=stream)
            hsv_image_cpu = gpu_hsv.download(stream=stream)  # Download HSV image once
            print(f"Downloaded debug image: {debug_image_cpu.shape if debug_image_cpu is not None else 'None'}")
            print(f"Downloaded HSV image: {hsv_image_cpu.shape if hsv_image_cpu is not None else 'None'}")
        except cv2.error as e:
            print(f"Error downloading debug or HSV image: {e}")
            debug_image_cpu = None
            hsv_image_cpu = None

    stream.waitForCompletion()

    if debug_mode and debug_image_cpu is not None:
        draw_grid(debug_image_cpu, grid_spacing=50)
    else:
        if debug_mode:
            print("Debug image is None, skipping visualization")

    scaled_rois = scale_rois(selected_rois, width, height)
    if debug_mode:
        print(f"Scaled ROIs: {scaled_rois}")

    top_left_has_shape, _ = check_green_steering_wheel(
        gpu_hsv, scaled_rois['top_left'], stream, morph_close, morph_dilate, debug_image_cpu, debug_mode, car_type
    )
    bottom_center_has_shape, _ = check_green_steering_wheel(
        gpu_hsv, scaled_rois['bottom_center'], stream, morph_close, morph_dilate, debug_image_cpu, debug_mode, car_type
    )

    stream.waitForCompletion()

    if debug_mode and debug_image_cpu is not None and hsv_image_cpu is not None:
        try:
            # Define mouse callback for HSV value display on hover
            def mouse_callback(event, x, y, flags, param):
                nonlocal debug_image_cpu, hsv_image_cpu, last_hsv_text
                if event == cv2.EVENT_MOUSEMOVE:
                    # Create a fresh copy of the debug image to avoid accumulating text
                    display_image = debug_image_cpu.copy()
                    draw_grid(display_image, grid_spacing=50)  # Redraw grid
                    # Redraw ROIs and contours from check_green_steering_wheel
                    for roi_name in ['top_left', 'bottom_center']:
                        roi = scaled_rois[roi_name]
                        x1, y1 = roi['x'], roi['y']
                        x2, y2 = x1 + roi['width'], y1 + roi['height']
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Note: Mask and contours are drawn in check_green_steering_wheel, so they persist

                    if 0 <= x < hsv_image_cpu.shape[1] and 0 <= y < hsv_image_cpu.shape[0]:
                        hsv_value = hsv_image_cpu[y, x]
                        hsv_text = f"HSV: {hsv_value[0]}, {hsv_value[1]}, {hsv_value[2]}"
                        # Print to console to help with tuning
                        if hsv_text != last_hsv_text:
                            #print(f"Hover at ({x}, {y}): {hsv_text}")
                            last_hsv_text = hsv_text
                        # Display HSV text near cursor
                        text_pos = (x + 10, y - 10) if x < width - 100 and y > 20 else (x - 100, y + 20)
                        cv2.putText(display_image, hsv_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254, 1, 154), 1, cv2.LINE_AA)
                    cv2.imshow(debug_window_name, display_image)

            cv2.namedWindow(debug_window_name)
            cv2.setMouseCallback(debug_window_name, mouse_callback)
            cv2.imshow(debug_window_name, debug_image_cpu)  # Initial display
            cv2.waitKey(1)  # Ensure window updates
        except Exception as e:
            print(f"Error displaying debug window: {e}")

    mode = "Automated Mode" if top_left_has_shape or bottom_center_has_shape else "Manual Mode"
    if debug_mode:
        print(f"Mode Detection Results: TL={top_left_has_shape}, BC={bottom_center_has_shape} -> Mode={mode}")
    return mode

# --- Driver Activity Detection Functions ---
# process_mediapipe_frame (unchanged)
def process_mediapipe_frame(args):
    frame, mp_hands, mp_pose = args
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_result = mp_hands.process(rgb_frame)
    pose_result = mp_pose.process(rgb_frame)
    return hands_result, pose_result

# process_activity_batch 
def process_activity_batch(activity_batch_data, mp_hands, mp_pose, yolo, activities, debug_mode, mp_drawing):
    """Processes a batch of frames with phone position and hand-hold verification."""
    batch_results = []
    batch_frames_cpu = [data['frame_cpu'] for data in activity_batch_data]
    batch_indices = [data['frame_index'] for data in activity_batch_data]
    batch_times = [data['time'] for data in activity_batch_data]

    if not batch_frames_cpu: return []

    print(f"Processing activity batch of size {len(batch_frames_cpu)} (Indices: {batch_indices[0]}..{batch_indices[-1]})")
    height, width = batch_frames_cpu[0].shape[:2]
    
    # Process YOLO and MediaPipe in parallel
    yolo_results_batch = yolo.predict(batch_frames_cpu, conf=0.2, imgsz=(width, height), 
                             verbose=False, device='cuda', batch=YOLO_BATCH_SIZE)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        mp_inputs = [(frame, mp_hands, mp_pose) for frame in batch_frames_cpu]
        mp_results = list(executor.map(process_mediapipe_frame, mp_inputs))

    for i, (frame_cpu, yolo_results_frame, (hands_result, pose_result)) in enumerate(zip(
        batch_frames_cpu, yolo_results_batch, mp_results
    )):
        frame_index = batch_indices[i]
        current_time = batch_times[i]
        activity = "none"
        confidence = 0.0
        
        
        # Initialize phone and food/drink variables
        phone_to_mouth = float('inf')
        phone_to_ear = float('inf')
        food_to_mouth = float('inf')
        is_phone = False
        is_food_drink = False
        phone_box = None
        food_box = None
        phone_held = False
        food_held = False
        phone_conf = 0.0
        food_conf = 0.0

        if hands_result.multi_hand_landmarks and pose_result.pose_landmarks:
            try:
                # Get pose landmarks
                pose_landmarks = pose_result.pose_landmarks.landmark
                mouth = pose_landmarks[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT]
                ear = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR]

                # Check for phone and food/drink
                
                detected_objects = []
                for result in yolo_results_frame:  
                    for box in result.boxes:
                        obj_name = result.names[int(box.cls)] if hasattr(result, 'names') else "unknown"
                        obj_conf = box.conf.item()
                        detected_objects.append(obj_name)
                        if "cell phone" in obj_name.lower():
                            is_phone = True
                            phone_box = box.xyxy[0].cpu().numpy()
                            phone_conf = max(phone_conf, obj_conf)
                        elif any(food in obj_name.lower() for food in ["apple", "banana", "orange", "cake", "broccoli", "carrot", "hot dog", "pizza", "donut", "sandwich", "bowl", "cup", "bottle"]):
                            is_food_drink = True
                            food_box = box.xyxy[0].cpu().numpy()
                            food_conf = max(food_conf, obj_conf)

                # Phone position analysis
                
                if is_phone and phone_box is not None:
                    phone_center_x = (phone_box[0] + phone_box[2]) / 2 / width  # Normalized
                    phone_center_y = (phone_box[1] + phone_box[3]) / 2 / height
                    phone_to_mouth = np.sqrt((phone_center_x - mouth.x)**2 + (phone_center_y - mouth.y)**2)
                    phone_to_ear = np.sqrt((phone_center_x - ear.x)**2 + (phone_center_y - ear.y)**2)

                # Food/drink position analysis
                if is_food_drink and food_box is not None:
                    food_center_x = (food_box[0] + food_box[2]) / 2 / width  # Normalized
                    food_center_y = (food_box[1] + food_box[3]) / 2 / height
                    food_to_mouth = np.sqrt((food_center_x - mouth.x)**2 + (food_center_y - mouth.y)**2)
                
                # Hand analysis
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    wrist_to_mouth = np.sqrt((wrist.x - mouth.x)**2 + (wrist.y - mouth.y)**2)
                    wrist_to_ear = np.sqrt((wrist.x - ear.x)**2 + (wrist.y - ear.y)**2)
                    hand_points = [
                        (int(lm.x * width), int(lm.y * height))
                        for lm in hand_landmarks.landmark
                    ]

                    # Phone-hold verification
                    
                    if is_phone and phone_box is not None:
                        phone_held = any(
                            (phone_box[0] <= x <= phone_box[2]) and 
                            (phone_box[1] <= y <= phone_box[3])
                            for x, y in hand_points
                        )

                    # Food/drink-hold verification
                    if is_food_drink and food_box is not None:
                        food_held = any(
                            (food_box[0] <= x <= food_box[2]) and 
                            (food_box[1] <= y <= food_box[3])
                            for x, y in hand_points
                        )
                    
                    # Activity priority logic
                    if is_phone or phone_held:
                        if phone_held and (phone_to_ear > 0.3 or phone_to_mouth > 0.2):
                            activity = "Browse the phone"
                            confidence = phone_conf
                            break
                        elif phone_to_ear < 0.3 or phone_to_mouth < 0.2:
                            activity = "talking on phone"
                            confidence = phone_conf
                            break
                    elif food_to_mouth < 0.4: #elif is_food_drink and food_held and food_to_mouth < 0.4: <<this is a better logic. the one currently used is for low quality video>> removed (food_held and (wrist_to_mouth<0.15))
                        activity = "consuming"
                        confidence = food_conf
                        break

            except Exception as e:
                print(f"Error in frame {frame_index}: {str(e)}")

        # Debug visualization
        if debug_mode:
            debug_frame = frame_cpu.copy()
            
            # Draw pose and hands
            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    debug_frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )
            if hands_result.multi_hand_landmarks:
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        debug_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
            
            # Draw YOLO boxes and phone distances
            for result in yolo_results_frame:  # Adjust for YOLOE output
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{result.names[int(box.cls)]} {box.conf.item():.2f}"
                    cv2.putText(debug_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    if "cell phone" in result.names[int(box.cls)].lower() and pose_result.pose_landmarks:
                        mouth = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT]
                        ear = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
                        phone_center = ((x1+x2)//2, (y1+y2)//2)
                        mouth_px = (int(mouth.x * width), int(mouth.y * height))
                        ear_px = (int(ear.x * width), int(ear.y * height))
                        cv2.line(debug_frame, phone_center, mouth_px, (255,0,0), 2)# Phone-to-mouth
                        cv2.line(debug_frame, phone_center, ear_px, (0,255,255), 2)# Phone-to-ear
                    elif any(food in result.names[int(box.cls)].lower() for food in ["apple", "banana", "orange", "cake", "broccoli", "carrot", "hot dog", "pizza", "donut", "sandwich", "bowl", "cup", "bottle"]) and pose_result.pose_landmarks:
                        mouth = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT]
                        food_center = ((x1+x2)//2, (y1+y2)//2)
                        mouth_px = (int(mouth.x * width), int(mouth.y * height))
                        cv2.line(debug_frame, food_center, mouth_px, (255,165,0), 2) # Food-to-mouth, orange line
            
            cv2.putText(debug_frame, f"Activity: {activity} (Conf: {confidence:.2f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Activity Debug", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        batch_results.append({
            "frame_index": frame_index,
            "time": current_time,
            "activity": activity,
            "consuming_buffer": 0,
            "phone_to_mouth": float(phone_to_mouth),
            "phone_to_ear": float(phone_to_ear),
            "confidence": confidence 
        })

    return batch_results

# --- Combined Processing Function (Unchanged from previous fix) ---
def process_video_combined(video_path, activities, selected_rois, progress_callback, debug_mode=False, car_type='n'):
    # (Setup, loop, cleanup - code omitted for brevity, same structure as before)
    MODE_CHECK_INTERVAL_SECONDS = 10
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Could not open video at {video_path}")
    mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7, model_complexity=0)
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
    mp_drawing = mp.solutions.drawing_utils
    print("Loading YOLOE model..."); yolo = YOLOE("yoloe-v8m-seg"); print("YOLOE model loaded.")
    # Set prompts based on selected activities
    activities = ["cell phone", "apple", "banana", "orange", "cake", "broccoli", "carrot", "hot dog", "pizza", "donut", "sandwich", "bowl", "cup", "bottle"]
    yolo.set_classes(activities, yolo.get_text_pe(activities))
    fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps if fps > 0 else 0
    print(f"Video FPS: {fps}, Total Frames: {total_frames}, Total Seconds: {total_seconds:.2f}")
    try:
        print(f"CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}"); cv2.cuda.setDevice(0)
        print(f"Using CUDA Device: {cv2.cuda.getDevice()}")
        stream_main = cv2.cuda.Stream()
        morph_filter_close = cv2_cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8U, KERNEL)
        morph_filter_dilate = cv2_cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8U, KERNEL, iterations=1)
        gpu_frame_full = cv2_cuda.GpuMat()
    except cv2.error as e:
        print(f"CUDA Initialization Error: {e}. Check OpenCV build and CUDA toolkit.")
        mp_hands.close(); mp_pose.close(); cap.release(); return [], [], [], []
    activity_frame_interval = int(fps) if fps > 0 else 1
    mode_frame_interval = int(fps * MODE_CHECK_INTERVAL_SECONDS) if fps > 0 else 10
    activity_intervals = []; current_activity = "none"; activity_start_time = 0.0; last_consuming_buffer = 0
    all_activity_results = []
    mode_intervals = []; current_mode = None; mode_start_time = 0.0
    activity_batch_queue = deque()
    frame_count = 0; start_time = time.time()
    while cap.isOpened():
        ret, frame_cpu = cap.read()
        if not ret: print("End of video stream."); break
        current_time_sec = frame_count / fps if fps > 0 else 0
        progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
        progress_callback(progress)
        gpu_frame_full.upload(frame_cpu, stream=stream_main)
        if frame_count % mode_frame_interval == 0:
            detected_mode = process_image_for_mode(
                gpu_frame_full, selected_rois, stream_main, morph_filter_close, morph_filter_dilate, debug_mode, car_type
            )
            stream_main.waitForCompletion()
            if current_mode is None:
                 current_mode = detected_mode; mode_start_time = 0.0
                 print(f"Initial Mode Detected: {current_mode} at {current_time_sec:.2f}s")
            elif detected_mode != current_mode:
                print(f"Mode Change Detected: {current_mode} -> {detected_mode} at {current_time_sec:.2f}s")
                mode_intervals.append({"mode": current_mode, "start": mode_start_time, "end": current_time_sec})
                current_mode = detected_mode; mode_start_time = current_time_sec
            if debug_mode:
                debug_frame_cpu_full = frame_cpu.copy()
                cv2.putText(debug_frame_cpu_full, f"Mode: {current_mode} (Detect @ {current_time_sec:.1f}s)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Debug: Main Frame with Mode", debug_frame_cpu_full)
        if frame_count % activity_frame_interval == 0:
            gpu_frame_activity = crop_to_first_quadrant_gpu(gpu_frame_full)
            frame_cpu = gpu_frame_activity.download(stream=stream_main)
            stream_main.waitForCompletion()
            activity_batch_queue.append({"frame_cpu": frame_cpu, "frame_index": frame_count, "time": current_time_sec, "consuming_buffer": last_consuming_buffer})
        is_last_iteration = (frame_count + 1) >= total_frames
        if len(activity_batch_queue) >= YOLO_BATCH_SIZE or (is_last_iteration and activity_batch_queue):
             batch_to_process = list(activity_batch_queue); activity_batch_queue.clear()
             activity_results_batch = process_activity_batch(batch_to_process, mp_hands, mp_pose, yolo, activities, debug_mode, mp_drawing)
             all_activity_results.extend(activity_results_batch)
             for result in activity_results_batch:
                 detected_activity = result["activity"]; frame_time = result["time"]; last_consuming_buffer = result["consuming_buffer"]
                 if detected_activity != current_activity:
                     if current_activity != "none": activity_intervals.append({"activity": current_activity, "start": activity_start_time, "end": frame_time})
                     if detected_activity != "none": current_activity = detected_activity; activity_start_time = frame_time; current_confidence = result["confidence"]
                     else: current_activity = "none"; current_confidence = 0.0
        if debug_mode:
            if cv2.waitKey(1) & 0xFF == ord('q'): print("'q' pressed, exiting loop."); break
        frame_count += 1
    end_time = time.time()
    print(f"\nVideo processing finished. Total time: {end_time - start_time:.2f} seconds."); print(f"Processed {frame_count} frames.")
    final_time = total_seconds if total_seconds > 0 else current_time_sec
    if current_activity != "none": activity_intervals.append({ "activity": current_activity, "start": activity_start_time, "end": final_time})
    if current_mode is not None: mode_intervals.append({ "mode": current_mode, "start": mode_start_time, "end": final_time })
    else: print("Warning: Video too short for any mode detection intervals.")
    cap.release(); mp_hands.close(); mp_pose.close()
    if debug_mode: cv2.destroyAllWindows()
    del gpu_frame_full, morph_filter_close, morph_filter_dilate; print("Resources released.")
    # Aggregate results (same as before)
    activity_results_agg = []
    activity_summary = {}
    for interval in activity_intervals:
        act = interval["activity"]; start = interval["start"]; end = interval["end"]
        duration = max(0.0, end - start)
        interval_confidence = 0.0
        for result in all_activity_results:  # Assuming activity_results_batch is accessible
            if result["time"] >= start and result["time"] <= end and result["activity"] == act:
                interval_confidence = max(interval_confidence, result["confidence"])
        if duration > 0.1:
            activity_summary[act] = activity_summary.get(act, 0) + duration
            activity_results_agg.append((video_path, act, round(start, 2), round(end, 2), round(duration, 2), round(interval_confidence, 2)))
    automated_duration = sum(max(0.0, interval["end"] - interval["start"]) for interval in mode_intervals if interval["mode"] == "Automated Mode")
    manual_duration = sum(max(0.0, interval["end"] - interval["start"]) for interval in mode_intervals if interval["mode"] == "Manual Mode")
    mode_results_agg = [(video_path, "Automated Mode", round(automated_duration, 2)), (video_path, "Manual Mode", round(manual_duration, 2))]
    print(f"\nActivity Intervals: {activity_intervals}"); print(f"Mode Intervals: {mode_intervals}")
    progress_callback(100)
    return activity_results_agg, activity_intervals, mode_intervals, mode_results_agg

# --- Correlation Function (Unchanged) ---
def correlate_activities_with_modes(activity_intervals, mode_intervals):
    # (Code omitted for brevity, same as before)
    activity_mode_durations = {}
    all_activities = set(interval["activity"] for interval in activity_intervals)
    for activity in all_activities:
         if activity not in activity_mode_durations: activity_mode_durations[activity] = {"Automated Mode": 0.0, "Manual Mode": 0.0}
    for activity_interval in activity_intervals:
        activity = activity_interval["activity"]; act_start = activity_interval["start"]; act_end = activity_interval["end"]
        for mode_interval in mode_intervals:
            mode = mode_interval["mode"]; mode_start = mode_interval["start"]; mode_end = mode_interval["end"]
            overlap_start = max(act_start, mode_start); overlap_end = min(act_end, mode_end)
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                activity_mode_durations[activity][mode] += overlap_duration
    for activity in activity_mode_durations:
        for mode in activity_mode_durations[activity]: activity_mode_durations[activity][mode] = round(activity_mode_durations[activity][mode], 2)
    return activity_mode_durations


# --- GUI Class (Modified for Radio Button Car Selection) ---
class VideoAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("logo.png"))
        self.setWindowTitle("Driver Activity and Mode Analyzer")
        self.setGeometry(100, 100, 900, 750) # Increased height slightly for car selection
        
        self.files = []
        self.results_cache = {}
        self.video_rois = {}  # Dictionary to store ROIs per video {video_path: rois}
        self.results_cache = {}

        # --- Main Layout ---
        main_layout = QVBoxLayout()
        logo_container = QHBoxLayout()
        logo_label = QLabel()
        pixmap = QPixmap("banner.png")
        pixmap = pixmap.scaledToWidth(250, Qt.SmoothTransformation)  # Resize if needed
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignLeft)
        logo_container.addWidget(logo_label, alignment=Qt.AlignLeft)
        
        # --- Upload Button ---
        self.upload_btn = QPushButton("Upload Video(s)")
        self.upload_btn.clicked.connect(self.upload_files)
        logo_container.addWidget(self.upload_btn, alignment=Qt.AlignLeft)
        main_layout.addLayout(logo_container)
        
        # --- Open Help button ---
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.open_help_dialog)
        logo_container.addWidget(self.help_btn, alignment=Qt.AlignLeft)
        
        # --- Website Button ---
        self.website_btn = QPushButton("Visit HS Lab")
        self.website_btn.clicked.connect(lambda: webbrowser.open("https://hslab.org"))
        logo_container.addWidget(self.website_btn, alignment=Qt.AlignLeft)

        # --- Top Controls Layout (Upload, Activities, Car Type) ---
        top_controls_layout = QHBoxLayout() # Horizontal layout for top controls

        
        # Activities Checkboxes Group
        activities_group = QGroupBox("Select Activities to Detect")
        activities_layout = QVBoxLayout()
        self.talking_checkbox = QCheckBox("Talking on Phone"); self.talking_checkbox.setChecked(True)
        self.consuming_checkbox = QCheckBox("Consuming (Food/Drink)"); self.consuming_checkbox.setChecked(True)
        
        self.Browse_checkbox = QCheckBox("Browse the phone"); self.Browse_checkbox.setChecked(True)
        activities_layout.addWidget(self.talking_checkbox)
        activities_layout.addWidget(self.consuming_checkbox)
        activities_layout.addWidget(self.Browse_checkbox) # Corrected widget added
        activities_group.setLayout(activities_layout)
        top_controls_layout.addWidget(activities_group)

        # Car Type Radio Buttons Group
        self.rois_dict = {
            # User provided ROIs - ensure keys match radio button object names below
            'c': {'label': 'Cadillac', 'rois': {
                 'top_left': {'x': 260, 'y': 260, 'width': 140, 'height': 140, 'min_area': 400, 'max_area': 1200, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.0},
                 'bottom_center': {'x': 800, 'y': 400, 'width': 600, 'height': 480, 'min_area': 2800, 'max_area': 8000, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.0}}}, # Corrected original ROIs from user
            'n': {'label': 'Nissan', 'rois': {
                 'top_left': {'x': 520, 'y': 120, 'width': 120, 'height': 100, 'min_area': 200, 'max_area': 6000, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.5},
                 'bottom_center': {'x': 520, 'y': 120, 'width': 120, 'height': 100, 'min_area': 1000, 'max_area': 6000, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.5}}},
            's': {'label': 'Tesla S', 'rois': {
                 'top_left': {'x': 495, 'y': 40, 'width': 45, 'height': 45, 'min_area': 200, 'max_area': 6000, 'circularity_min': 0.25, 'circularity_max': 1.35, 'aspect_ratio_min': 0.55, 'aspect_ratio_max': 1.9},
                 'bottom_center': {'x': 495, 'y': 40, 'width': 45, 'height': 45, 'min_area': 760, 'max_area': 6000, 'circularity_min': 0.25, 'circularity_max': 1.35, 'aspect_ratio_min': 0.55, 'aspect_ratio_max': 2.5}}},
            '3': {'label': 'Tesla 3', 'rois': {
                 'top_left': {'x': 944, 'y': 180, 'width': 80, 'height': 80, 'min_area': 200, 'max_area': 800, 'circularity_min': 0.35, 'circularity_max': 1.25, 'aspect_ratio_min': 0.65, 'aspect_ratio_max': 1.7},
                 'bottom_center': {'x': 944, 'y': 280, 'width': 200, 'height': 160, 'min_area': 1200, 'max_area': 6000, 'circularity_min': 0.35, 'circularity_max': 1.25, 'aspect_ratio_min': 0.65, 'aspect_ratio_max': 1.7}}},
            'v': {'label': 'Volvo', 'rois': {
                 'top_left': {'x': 944, 'y': 180, 'width': 80, 'height': 80, 'min_area': 200, 'max_area': 800, 'circularity_min': 0.4, 'circularity_max': 1.2, 'aspect_ratio_min': 0.7, 'aspect_ratio_max': 1.6},
                 'bottom_center': {'x': 630, 'y': 360, 'width': 360, 'height': 360, 'min_area': 7500, 'max_area': 25000, 'circularity_min': -1, 'circularity_max': 1.2, 'aspect_ratio_min': -1, 'aspect_ratio_max': 2.1}}},
        }
        car_group = QGroupBox("Select Car Type (for Mode Detection ROIs)")
        car_layout = QVBoxLayout()
        self.car_radio_buttons = {} # Store radio buttons keyed by car code
        
        default_car = 'n' # Set Nissan as default
        for key, data in self.rois_dict.items():  # From here till the next 8 lines (until self.car_radio_buttons['s'].setEnabled(False) is to disable the button
            radio_button = QRadioButton(data['label'])  
            self.car_radio_buttons[key] = radio_button
            
            if key == default_car:
                radio_button.setChecked(True) # Set default selection
            car_layout.addWidget(radio_button)
        self.car_radio_buttons['3'].setEnabled(False)
        
        car_group.setLayout(car_layout)
        top_controls_layout.addWidget(car_group)

        main_layout.addLayout(top_controls_layout) # Add the top controls HBox to main layout

        # --- Debug Checkbox ---
        self.debug_checkbox = QCheckBox("Enable Debug Mode (Visualize Processing)")
        main_layout.addWidget(self.debug_checkbox)

        # --- Define ROIs Button ---
        self.define_roi_btn = QPushButton("Define ROIs (Debug Mode)")
        self.define_roi_btn.clicked.connect(self.define_rois)
        self.define_roi_btn.setEnabled(False)  # Disabled until debug mode and car type selected
        
        
        
        
        # --- Analyze Button ---
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.define_roi_btn)
        self.analyze_btn = QPushButton("Analyze Video(s)")
        self.analyze_btn.clicked.connect(self.analyze_video)
        buttons_layout.addWidget(self.analyze_btn)
        self.export_btn = QPushButton("Export Summary as CSV")
        self.export_btn.clicked.connect(self.export_summary)
        self.export_btn.setEnabled(False)  # Disabled until analysis is complete
        buttons_layout.addWidget(self.export_btn)
        main_layout.addLayout(buttons_layout)
        
        

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        
        # --- Results Tabs ---
        self.tabs = QTabWidget()
        self.activity_table = QTableWidget(0, 6)
        self.activity_table.setHorizontalHeaderLabels(["File", "Activity", "Start Time (s)", "End Time (s)", "Duration (s)", "Confidence"])
        self.tabs.addTab(self.activity_table, "Activity Segments")
        #self.correlation_table = QTableWidget(0, 3)
        #self.correlation_table.setHorizontalHeaderLabels(["Activity", "Mode", "Duration (s)"])
        #self.tabs.addTab(self.correlation_table, "Activity by Mode")
        self.mode_change_table = QTableWidget(0, 3)
        self.mode_change_table.setHorizontalHeaderLabels(["File", "Mode", "Timestamp (s)"])
        self.tabs.addTab(self.mode_change_table, "Mode Changes")
        self.mode_table = QTableWidget(0, 5)
        self.mode_table.setHorizontalHeaderLabels(["File", "Mode", "Total Duration (s)", "Date", "Time"])
        self.tabs.addTab(self.mode_table, "Mode Durations")
        self.activity_mode_table = QTableWidget(0, 8)
        self.activity_mode_table.setHorizontalHeaderLabels([
            "File", "Mode", "Activity", "Activity Duration (s)", "Mode Duration (s)", "Percentage of Time (%)", "Date", "Time"
        ])
        self.tabs.addTab(self.activity_mode_table, "Activity Mode Breakdown")
        main_layout.addWidget(self.tabs)

        # --- Set Central Widget ---
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.files = []
        self.results_cache = {} # Cache results per file
        
        # --- ROI Drawing State ---
        self.drawing_rois = False
        self.new_rois = None
        self.roi_drawing_window = "Draw ROIs (Drag to select, Press 'c' to confirm, 'r' to reset)"
        
        # Connect debug checkbox and car radio buttons to update ROI button state
        self.debug_checkbox.stateChanged.connect(self.update_roi_button_state)
        for radio_button in self.car_radio_buttons.values():
            radio_button.toggled.connect(self.update_roi_button_state)

    def define_rois(self):
        """Prompt the user to define ROIs for each selected video."""
        if not self.files:
            print("No video files selected!")
            return
        if not self.debug_checkbox.isChecked():
            print("Debug mode must be enabled to define ROIs!")
            return

        # Get selected car type
        selected_car_key = None
        for key, radio_button in self.car_radio_buttons.items():
            if radio_button.isChecked():
                selected_car_key = key
                break
        if selected_car_key is None:
            print("No car type selected!")
            return

        self.video_rois.clear()  # Clear previous ROIs
        print(f"Cleared self.video_rois. Starting ROI definition for {len(self.files)} videos.")
        total_frames_dict = {}  # Store total frames for each video

        # Normalize file paths
        self.files = [os.path.normpath(f) for f in self.files]
        
        # Get total frames for each video to validate frame numbers
        for video_path in self.files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue
            total_frames_dict[video_path] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        # Prompt for frame number for each video
        for video_path in self.files:
            total_frames = total_frames_dict.get(video_path, 0)
            if total_frames == 0:
                print(f"Skipping {video_path}: Could not read video.")
                continue

            frame_number = 0
            # Draw ROIs for this video
            print(f"Defining ROIs for {video_path} at frame {frame_number}")
            while True:
                frame_number, ok = QInputDialog.getInt(
                    self,
                    f"Select Frame for {os.path.basename(video_path)}",
                    f"Enter frame number (0 to {total_frames - 1}):",
                    value=0,
                    min=0,
                    max=total_frames - 1
                )
                if not ok:
                    print(f"Frame selection canceled for {video_path}. Using default ROIs for car type {selected_car_key}.")
                    self.video_rois[video_path] = self.rois_dict[selected_car_key]['rois']
                    break

                new_rois = self.draw_roi_interactively(video_path, selected_car_key, frame_number)

                if new_rois:
                    confirm = QMessageBox.question(
                        self,
                        "Confirm ROIs",
                        f"Are you sure you want to save these ROIs for {os.path.basename(video_path)}?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if confirm == QMessageBox.Yes:
                        self.video_rois[video_path] = new_rois
                        print(f"Stored ROIs for {video_path}: {new_rois}")
                        break
                    else:
                        print("User declined. Asking for frame number again.")
                        continue
                elif new_rois is None:
                    print(f"ROI definition canceled for {video_path}. Using default ROIs for car type {selected_car_key}.")
                    self.video_rois[video_path] = self.rois_dict[selected_car_key]['rois']
                    break



        if self.video_rois:
            print(f"ROIs defined for {len(self.video_rois)} videos.")
        else:
            print("No ROIs defined. Analysis will use default ROIs for selected car type.")

    def draw_roi_interactively(self, video_path, car_key, frame_number=0):
        """Display a video frame and allow the user to draw ROIs with the mouse."""
        global drawing, start_point, end_point, rois_drawn, frame_copy
        drawing = False
        start_point = (-1, -1)
        end_point = (-1, -1)
        rois_drawn = []  # List to store (x, y, w, h) for top_left and bottom_center
        frame_copy = None

        def mouse_callback(event, x, y, flags, param):
            global drawing, start_point, end_point, frame_copy
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                end_point = (x, y)
                temp_frame = frame_copy.copy()
                cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)
                for roi in rois_drawn:
                    cv2.rectangle(temp_frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)
                cv2.imshow(self.roi_drawing_window, temp_frame)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                x = min(start_point[0], end_point[0])
                y = min(start_point[1], end_point[1])
                w = abs(end_point[0] - start_point[0])
                h = abs(end_point[1] - start_point[1])
                if w > 10 and h > 10:  # Minimum size to avoid accidental clicks
                    rois_drawn.append((x, y, w, h))
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.imshow(self.roi_drawing_window, frame_copy)

        # Open video and read first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Set to the specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Error: Could not read frame {frame_number} from {video_path}")
            return None

        
        # Crop to third quadrant (bottom-left)
        height, width = frame.shape[:2]
        frame_cropped = frame[height//2:height, 0:width//2]
        if frame_cropped.size == 0:
            print(f"Error: Cropped frame is empty (height={height}, width={width})")
            return None
        
        frame_copy = frame_cropped.copy()
        cv2.namedWindow(self.roi_drawing_window)
        cv2.setMouseCallback(self.roi_drawing_window, mouse_callback)

        print(f"Draw ROIs on frame {frame_number}: Click and drag to draw rectangles (max 2: top_left, bottom_center).")
        print("Press 'c' to confirm, 'r' to reset, 'q' to cancel.")

        while True:
            display_frame = frame_copy.copy()
            if drawing:
                cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)
            for i, roi in enumerate(rois_drawn):
                label = "top_left" if i == 0 else "bottom_center"
                cv2.rectangle(display_frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)
                cv2.putText(display_frame, label, (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(display_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.roi_drawing_window, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(rois_drawn) == 2:
                break  # Exit ROI drawing, let define_rois() handle confirmation

            elif key == ord('r'):  # Reset ROIs
                rois_drawn = []
                frame_copy = frame_cropped.copy()
            elif key == ord('q'):  # Cancel
                cv2.destroyWindow(self.roi_drawing_window)
                cv2.destroyAllWindows()
                return None

        cv2.destroyWindow(self.roi_drawing_window)
        cv2.destroyAllWindows()

        # Create ROI dictionary with default parameters
        if len(rois_drawn) != 2:
            print("Error: Exactly 2 ROIs (top_left, bottom_center) required.")
            return None

        # Scale drawn coordinates to reference resolution (944x480)
        crop_width, crop_height = frame_cropped.shape[1], frame_cropped.shape[0]
        ref_width, ref_height = 944, 480
        x_scale = ref_width / crop_width
        y_scale = ref_height / crop_height
        
        # Use default parameters from existing rois_dict for the car type
        default_top_left = self.rois_dict[car_key]['rois']['top_left']
        default_bottom_center = self.rois_dict[car_key]['rois']['bottom_center']  # New (correct)

        new_rois = {
            'top_left': {
                'x': int(rois_drawn[0][0] * x_scale),
                'y': int(rois_drawn[0][1] * y_scale),
                'width': int(rois_drawn[0][2] * x_scale),
                'height': int(rois_drawn[0][3] * y_scale),
                'min_area': default_top_left['min_area'],  # Keep top_left constraints
                'max_area': default_top_left['max_area'],
                'circularity_min': default_top_left['circularity_min'],
                'circularity_max': default_top_left['circularity_max'],
                'aspect_ratio_min': default_top_left['aspect_ratio_min'],
                'aspect_ratio_max': default_top_left['aspect_ratio_max']
            },
            'bottom_center': {
                'x': int(rois_drawn[1][0] * x_scale),
                'y': int(rois_drawn[1][1] * y_scale),
                'width': int(rois_drawn[1][2] * x_scale),
                'height': int(rois_drawn[1][3] * y_scale),
                'min_area': default_bottom_center['min_area'],  # Keep bottom_center constraints
                'max_area': default_bottom_center['max_area'],
                'circularity_min': default_bottom_center['circularity_min'],
                'circularity_max': default_bottom_center['circularity_max'],
                'aspect_ratio_min': default_bottom_center['aspect_ratio_min'],
                'aspect_ratio_max': default_bottom_center['aspect_ratio_max']
            }
        }
        print(f"Returning ROIs for {video_path}: {new_rois}")
        return new_rois
    
    def update_roi_button_state(self):
        """Enable Define ROIs button only when debug mode is checked and a car type is selected."""
        debug_enabled = self.debug_checkbox.isChecked()
        car_selected = any(radio_button.isChecked() for radio_button in self.car_radio_buttons.values())
        self.define_roi_btn.setEnabled(debug_enabled and car_selected)
        
    
    def open_help_dialog(self):
        doc_path = os.path.join(os.getcwd(), "documentation.md")
        if not os.path.exists(doc_path):
            print("Help file not found.")
            return

        with open(doc_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert markdown to HTML
        html = markdown2.markdown(md_content)

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Help / Documentation")
        dialog.resize(600, 700)

        layout = QVBoxLayout()
        text_browser = QTextBrowser()
        text_browser.setHtml(html)
        layout.addWidget(text_browser)

        dialog.setLayout(layout)
        dialog.exec_()



    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov)")
        if files:
            self.files = files; print("Selected files:", self.files)
            self.clear_results_display(); self.results_cache = {}
            self.upload_btn.setText(f"{len(self.files)} Video(s) Selected")

    def update_progress(self, value):
        self.progress_bar.setValue(value); QApplication.processEvents()

    def clear_results_display(self):
         self.activity_table.setRowCount(0); self.mode_table.setRowCount(0)
         self.mode_change_table.setRowCount(0); self.progress_bar.setValue(0); self.progress_bar.setFormat("%p%")

    def export_summary(self):
        #if self.activity_mode_table.rowCount() == 0:
        #    print("No data to export!")
        #    return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Summary as CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                headers = [
                    "File", "Mode", "Activity", "Activity Duration (s)",
                    "Mode Duration (s)", "Percentage of Time (%)", "Date", "Time"
                ]
                writer.writerow(headers)
                for row in range(self.activity_mode_table.rowCount()):
                    row_data = []
                    for col in range(self.activity_mode_table.columnCount()):
                        item = self.activity_mode_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            # Export Activity Segments with Confidence
            activity_csv_path = file_path.replace('.csv', '_activity_segments.csv')
            with open(activity_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                headers = ["File", "Activity", "Start Time (s)", "End Time (s)", "Duration (s)", "Confidence"]
                writer.writerow(headers)
                for row in range(self.activity_table.rowCount()):
                    row_data = []
                    for col in range(self.activity_table.columnCount()):
                        item = self.activity_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            # Export Mode Durations
            mode_csv_path = file_path.replace('.csv', '_mode_durations.csv')
            with open(mode_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                headers = ["File", "Mode", "Total Duration (s)", "Date", "Time"]
                writer.writerow(headers)
                for row in range(self.mode_table.rowCount()):
                    row_data = []
                    for col in range(self.mode_table.columnCount()):
                        item = self.mode_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            # Export Mode Changes
            mode_changes_csv_path = file_path.replace('.csv', '_mode_changes.csv')
            with open(mode_changes_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                headers = ["File", "Mode", "Timestamp (s)"]
                writer.writerow(headers)
                for row in range(self.mode_change_table.rowCount()):
                    row_data = []
                    for col in range(self.mode_change_table.columnCount()):
                        item = self.mode_change_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
                    
            print(f"Summary exported to {file_path}")
            print(f"Activity Segments exported to {activity_csv_path}")
            print(f"Mode Durations exported to {mode_csv_path}")
            print(f"Mode Changes exported to {mode_changes_csv_path}")
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
    
    def analyze_video(self):
       
        if not self.files: print("No files selected!"); return
        activities = []
        if self.talking_checkbox.isChecked(): activities.append("talking on phone")
        if self.consuming_checkbox.isChecked(): activities.append("consuming")
        # Corrected check for Browse checkbox
        if self.Browse_checkbox.isChecked(): activities.append("Browse the phone")
        if not activities: print("No activities selected for detection!"); return

        # --- Determine selected car type ---
        selected_car_key = None
        for key, radio_button in self.car_radio_buttons.items():
            if radio_button.isChecked():
                selected_car_key = key
                break
        # Fallback to default if none selected (shouldn't happen with radio buttons)
        if selected_car_key is None:
            print("Warning: No car type selected, defaulting to Nissan.")
            selected_car_key = 'n'

        
        # --- End car type determination ---

        debug_mode = self.debug_checkbox.isChecked()
        print(f"Starting analysis with activities: {activities}, Debug mode: {debug_mode}")
        print(f"Available ROIs in self.video_rois: {self.video_rois}")  # Debug print
        self.clear_results_display(); self.analyze_btn.setEnabled(False); QApplication.processEvents()
        all_activity_results_agg = [] 
        all_correlation_results = [] 
        all_mode_results_agg = []
        all_activity_mode_results = []
        all_mode_changes = []
        total_files = len(self.files)
        for idx, file in enumerate(self.files):
            file = os.path.normpath(file)  # Normalize path
            print(f"\n--- Processing file {idx+1}/{total_files}: {os.path.basename(file)} ---")
            self.progress_bar.setFormat(f"Processing {os.path.basename(file)} - %p%")
            self.update_progress(0)
            
            
            # --- Improved Date/Time Extraction ---
            date, time = "Unknown", "Unknown"
            cap = cv2.VideoCapture(file)
            if not cap.isOpened():
                print(f"Error: Could not open video {file}")
                continue
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                height, width = frame.shape[:2]
                print(f"Video resolution: {width}x{height}")
                
                # Define OCR regions based on resolution
                if width == 1888 and height == 960:
                    # For 1888x960 resolution
                    quadrants = [
                        frame[height//2:height, 0:width//2],  # Bottom-left
                        frame[0:height//2, width//2:width],   # Top-right
                        #frame[height//2:height, width//2:width],  # Bottom-right
                        frame[0:height//2, 0:width//2],       # Top-left
                    ]
                    ocr_top, ocr_bottom, ocr_left, ocr_right = 10, 50, 20, 320
                elif width == 2560 and height == 1440:
                    # For 2560x1440 resolution
                    quadrants = [
                        frame[height//2:height, 0:width//2],  # Bottom-left
                        frame[0:height//2, width//2:width],   # Top-right
                        frame[0:height//2, 0:width//2],       # Top-left
                    ]
                    ocr_top, ocr_bottom, ocr_left, ocr_right = 20, 75, 30, 450
                else:
                    # For other resolutions, try common positions with relative coordinates
                    quadrants = [
                        frame[int(height*0.7):height, 0:width//2],  # Bottom-left
                        frame[0:height//2, width//2:width],        # Top-right
                        #frame[height//2:height, width//2:width],   # Bottom-right
                        frame[0:height//2, 0:width//2],            # Top-left
                    ]
                    # Use relative coordinates for OCR region
                    ocr_top = int(height * 0.02)
                    ocr_bottom = int(height * 0.06)
                    ocr_left = int(width * 0.02)
                    ocr_right = int(width * 0.25)
                
                for quadrant in quadrants:
                    try:
                        ocr_region = quadrant[ocr_top:ocr_bottom, ocr_left:ocr_right]
                        gray = cv2.cvtColor(ocr_region, cv2.COLOR_BGR2GRAY)
                        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        text = pytesseract.image_to_string(gray, config='--psm 6').strip()
                        
                        if not text:
                            continue
                        
                        # Cleanse and validate
                        parts = text.split(maxsplit=1)
                        if len(parts) >= 2:
                            date_candidate = re.sub(r'[^0-9/]', '', parts[0])
                            time_candidate = re.sub(r'[^0-9:]', '', parts[1])
                            
                            if is_valid_date(date_candidate) and is_valid_time(time_candidate):
                                date, time = date_candidate, time_candidate
                                break
                                
                    except Exception as e:
                        print(f"OCR error in quadrant: {e}")

            print(f"Extracted Date: {date}, Time: {time} from {os.path.basename(file)}")
            self.results_cache[file] = {"date": date, "time": time}
            
            # Select ROIs for this video
            selected_rois = self.video_rois.get(file, self.rois_dict[selected_car_key]['rois'])
            print(f"Using ROIs for {os.path.basename(file)}: {selected_rois}")
            
            
            # Check cache only if ROI selection hasn't changed (or store ROI choice in cache key)
            # Simple approach: ignore cache if ROIs might have changed. For full caching, include selected_car_key in cache key.
            # if file in self.results_cache: ...
            try:
                # Pass the dynamically selected ROIs
                activity_results_agg, activity_intervals, mode_intervals, mode_results_agg = process_video_combined(
                    file, activities, selected_rois, lambda p: self.update_progress(p), debug_mode, selected_car_key
                )
                # Consider adding selected_car_key to cache key if caching is desired across ROI changes
                # self.results_cache[(file, selected_car_key)] = (activity_results_agg, activity_intervals, mode_intervals, mode_results_agg)
            except Exception as e:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"ERROR processing file {file}: {e}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                import traceback; traceback.print_exc()
                activity_results_agg, activity_intervals, mode_intervals, mode_results_agg = [], [], [], []
            all_activity_results_agg.extend(activity_results_agg)
            all_mode_results_agg.extend(mode_results_agg)
            for interval in mode_intervals:
                all_mode_changes.append((file, interval["mode"], round(interval["start"], 2)))
            correlation_results = correlate_activities_with_modes(activity_intervals, mode_intervals)
            for activity, modes in correlation_results.items():
                 for mode, duration in modes.items():
                     if duration > 0: 
                        all_correlation_results.append((activity, mode, duration))
                        # Find mode duration for this file and mode
                        mode_duration = next(
                            (md[2] for md in mode_results_agg if md[0] == file and md[1] == mode),
                            0.0
                        )
                        if mode_duration > 0:
                            percentage = (duration / mode_duration) * 100
                            all_activity_mode_results.append((
                                os.path.basename(file), mode, activity,
                                round(duration, 2), round(mode_duration, 2), round(percentage, 2),
                                self.results_cache[file]["date"], self.results_cache[file]["time"]
                            ))
            QApplication.processEvents()
        print("\n--- Aggregating and Displaying Results ---")
        # Display results (code omitted for brevity, same as before)
        self.activity_table.setRowCount(len(all_activity_results_agg))
        for i, (file, activity, start, end, duration, confidence) in enumerate(all_activity_results_agg):
             self.activity_table.setItem(i, 0, QTableWidgetItem(os.path.basename(file))); self.activity_table.setItem(i, 1, QTableWidgetItem(activity))
             self.activity_table.setItem(i, 2, QTableWidgetItem(str(start))); self.activity_table.setItem(i, 3, QTableWidgetItem(str(end))); self.activity_table.setItem(i, 4, QTableWidgetItem(str(duration)))
             self.activity_table.setItem(i, 5, QTableWidgetItem(f"{confidence:.2f}"))
        self.activity_table.resizeColumnsToContents()
        
        grouped_modes = {}; i = 0
        for file, mode, dur in all_mode_results_agg: grouped_modes[(file, mode)] = grouped_modes.get((file, mode), 0) + dur
        self.mode_table.setRowCount(len(grouped_modes))
        for (file, mode), duration in grouped_modes.items():
              self.mode_table.setItem(i, 0, QTableWidgetItem(os.path.basename(file))); self.mode_table.setItem(i, 1, QTableWidgetItem(mode)); self.mode_table.setItem(i, 2, QTableWidgetItem(f"{duration:.2f}"))
              date = self.results_cache.get(file, {}).get("date", "Unknown")
              time = self.results_cache.get(file, {}).get("time", "Unknown")
              self.mode_table.setItem(i, 3, QTableWidgetItem(date)); self.mode_table.setItem(i, 4, QTableWidgetItem(time)); i+=1
        self.mode_table.resizeColumnsToContents()
        self.mode_change_table.setRowCount(len(all_mode_changes))
        for i, (file, mode, timestamp) in enumerate(all_mode_changes):
            self.mode_change_table.setItem(i, 0, QTableWidgetItem(os.path.basename(file)))
            self.mode_change_table.setItem(i, 1, QTableWidgetItem(mode))
            self.mode_change_table.setItem(i, 2, QTableWidgetItem(f"{timestamp:.2f}"))
        self.mode_change_table.resizeColumnsToContents()
        self.activity_mode_table.setRowCount(len(all_activity_mode_results))
        for i, (file, mode, activity, act_duration, mode_duration, percentage, date, time) in enumerate(all_activity_mode_results):
            self.activity_mode_table.setItem(i, 0, QTableWidgetItem(file))
            self.activity_mode_table.setItem(i, 1, QTableWidgetItem(mode))
            self.activity_mode_table.setItem(i, 2, QTableWidgetItem(activity))
            self.activity_mode_table.setItem(i, 3, QTableWidgetItem(f"{act_duration:.2f}"))
            self.activity_mode_table.setItem(i, 4, QTableWidgetItem(f"{mode_duration:.2f}"))
            self.activity_mode_table.setItem(i, 5, QTableWidgetItem(f"{percentage:.2f}"))
            self.activity_mode_table.setItem(i, 6, QTableWidgetItem(date))
            self.activity_mode_table.setItem(i, 7, QTableWidgetItem(time))
        self.activity_mode_table.resizeColumnsToContents()
        self.progress_bar.setFormat("Analysis Complete")
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        print("Analysis complete. Results displayed.")


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Start splash screen
    splash = QWidget()
    splash.setWindowFlags(Qt.FramelessWindowHint | Qt.SplashScreen)
    splash.setAttribute(Qt.WA_TranslucentBackground)
    splash.setStyleSheet("background-color: black;")
    layout = QVBoxLayout(splash)
    gif_label = QLabel()
    gif_label.setAlignment(Qt.AlignCenter)  # Make sure it centers in the layout
    gif_movie = QMovie("splash.gif")
    gif_movie.setScaledSize(QSize(240, 240))
    gif_label.setMovie(gif_movie)
    
    layout.addWidget(gif_label)
        
    gif_movie.start()
    splash.show()
    
    
    # CUDA Check (same as before)
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() == 0: print("! WARNING: No CUDA devices found by OpenCV !")
        else: print(f"Found {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)."); cv2.cuda.printShortCudaDeviceInfo(cv2.cuda.getDevice())
    except Exception as e: print(f"Error during CUDA check: {e}.")

    
    window = VideoAnalyzer()
    
    QTimer.singleShot(5000, lambda: (
        splash.close(),
        window.show()
    ))
    
    sys.exit(app.exec_())