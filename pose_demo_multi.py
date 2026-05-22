# -*- coding: utf-8 -*-
"""
Created April 2026

@author: RCourter
"""
# %% Import and setup
import cv2
import time
import os
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


# %% Define functions
# === Resource helper ===
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# === Logo overlay ===
def overlay_logo(frame, logo, position="bottom-left", scale=0.2, margin=20):
    if logo is None:
        return frame

    h, w, _ = frame.shape
    lh, lw = logo.shape[:2]

    target_width = int(w * scale)
    aspect_ratio = lw / lh
    target_height = int(target_width / aspect_ratio)
    logo_resized = cv2.resize(logo, (target_width, target_height))

    lh, lw = logo_resized.shape[:2]

    if position == "bottom-left":
        x, y = margin, h - lh - margin
    elif position == "bottom-right":
        x, y = w - lw - margin, h - lh - margin
    else:
        x, y = margin, margin

    if logo_resized.shape[2] == 4:
        alpha = logo_resized[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+lh, x:x+lw, c] = (
                alpha * logo_resized[:, :, c] +
                (1 - alpha) * frame[y:y+lh, x:x+lw, c]
            )
    else:
        frame[y:y+lh, x:x+lw] = logo_resized

    return frame

# === Tracking helpers ===
def get_centroid(landmarks, w, h):

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    cx = int((left_hip.x + right_hip.x) / 2 * w)
    cy = int((left_hip.y + right_hip.y) / 2 * h)

    return (cx, cy)

def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

# === Async callback ===
latest_result = None
def result_callback(result, output_image, timestamp_ms):
    global latest_result, processing
    latest_result = result
    processing = False

# %% Initialization
# === Directories ===
save_dir = os.path.join(os.getcwd(), "screenshots")
video_dir = os.path.join(os.getcwd(), "videos")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

model_path = resource_path("pose_landmarker_lite.task")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Could not find model file:\n{model_path}"
    )
    
BaseOptions = python.BaseOptions

# === Mediapipe setup ===
options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=3,
    result_callback=result_callback
)

detector = vision.PoseLandmarker.create_from_options(options)

# === Tracking state ===
prev_centroids = []
prev_ids = []
next_id = 0
missed_frames = 0
MAX_MISSES = 10
processing = False

COLORS = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255)]

POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),(23,24),
    (11,23),(12,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

# === Load logo ===
logo = None
logo_path = resource_path("logo.png")
if os.path.exists(logo_path):
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

# === Video setup ===
cap = cv2.VideoCapture(0) #1
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === State ===
frame_timestamp = 0
flash_end_time = 0

# Recording
recording = False
record_frames_remaining = 0
out = None
last_valid_result = None

# === Window ===
cv2.namedWindow("Pose", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# %% Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

    # === Timestamp (stable) ===
    frame_timestamp += 33

    #detector.detect_async(mp_image, frame_timestamp)
    if not processing:
        detector.detect_async(mp_image, frame_timestamp)
        processing = True

    # === Get latest result ===
    if latest_result is not None:
        last_valid_result = latest_result
        latest_result = None

    result = last_valid_result

    # === Tracking ===
    current_centroids = []
    assignments = []
    
    if result and result.pose_landmarks:
        missed_frames = 0
    
        # --- Step 1: compute centroids ---
        for person in result.pose_landmarks:
            current_centroids.append(get_centroid(person, w, h))
    
        # --- sanity check ---
        if len(prev_centroids) != len(prev_ids):
            prev_centroids, prev_ids = [], []
    
        used_prev = set()
    
        # --- Step 2: match to previous ---
        for c in current_centroids:
            best_idx, best_dist = -1, float("inf")
    
            for i, pc in enumerate(prev_centroids):
                if i in used_prev:
                    continue
    
                d = distance(c, pc)
                if d < best_dist:
                    best_dist, best_idx = d, i
    
            if best_idx != -1 and best_dist < 120:
                assignments.append(prev_ids[best_idx])
                used_prev.add(best_idx)
            else:
                assignments.append(None)
    
        # --- Step 3: assign new IDs ---
        for i in range(len(assignments)):
            if assignments[i] is None:
                assignments[i] = next_id
                next_id += 1
    
        # --- Step 4: smoothing (ID-aware) ---
        for i in range(len(current_centroids)):
            pid = assignments[i]
    
            if pid in prev_ids:
                prev_index = prev_ids.index(pid)
                prev_c = prev_centroids[prev_index]
                new_c = current_centroids[i]
    
                alpha = 0.7
                current_centroids[i] = (
                    int(alpha * prev_c[0] + (1 - alpha) * new_c[0]),
                    int(alpha * prev_c[1] + (1 - alpha) * new_c[1])
                )
    
        # --- Step 5: draw ---
        for i, person in enumerate(result.pose_landmarks):
            pid = assignments[i]
            color = COLORS[pid % len(COLORS)]
    
            for lm in person:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, color, -1)
    
            for start, end in POSE_CONNECTIONS:
                try:
                    p1, p2 = person[start], person[end]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 3)
                except:
                    pass
    
            cx, cy = current_centroids[i]
            # cv2.putText(frame, f"ID {pid}", (cx, cy-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
        prev_centroids = current_centroids.copy()
        prev_ids = assignments.copy()
    
    else:
        missed_frames += 1
    
        if missed_frames > MAX_MISSES:
            prev_centroids, prev_ids = [], []
            missed_frames = 0

    # === Logo ===
    frame = overlay_logo(frame, logo, "bottom-left", 0.25)

    # === Flash ===
    if time.time() < flash_end_time:
        overlay = frame.copy()
        overlay[:] = (255,255,255)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # === Recording ===
    if recording and out is not None:
        out.write(frame)
        record_frames_remaining -= 1

        cv2.circle(frame, (50,50), 15, (0,0,255), -1)
        cv2.putText(frame, "REC", (75,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if record_frames_remaining <= 0:
            recording = False
            out.release()
            print("Recording saved")

    # === Show ===
    cv2.imshow("Pose", frame)

    # === Keyboard ===
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == 32:  # screenshot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(save_dir, f"screenshot_{timestamp}.png")
        cv2.imwrite(path, frame)
        print("Saved:", path)
        flash_end_time = time.time() + 0.2

    elif key == ord('r') and not recording:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(video_dir, f"clip_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))

        recording = True
        record_frames_remaining = 5 * fps

        print("Recording started:", path)

cap.release()
cv2.destroyAllWindows()