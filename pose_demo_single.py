# -*- coding: utf-8 -*-
"""
Created July 2025

@author: RCourter
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import os
import sys

# =========================================================
# === Define custom functions
# =========================================================

def resource_path(relative_path):
    """
    Get absolute path to resource.
    Works for development and PyInstaller bundle.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def overlay_logo(frame, logo, position="bottom-left", scale=0.15, margin=20):
    """
    Overlay logo image onto frame.
    """

    if logo is None:
        return frame

    h, w, _ = frame.shape
    lh, lw = logo.shape[:2]

    # Resize while preserving aspect ratio
    target_width = int(w * scale)
    aspect_ratio = lw / lh
    target_height = int(target_width / aspect_ratio)

    logo_resized = cv2.resize(
        logo,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    lh, lw = logo_resized.shape[:2]

    # Positioning
    if position == "bottom-left":
        x, y = margin, h - lh - margin

    elif position == "bottom-right":
        x, y = w - lw - margin, h - lh - margin

    elif position == "top-left":
        x, y = margin, margin

    elif position == "top-right":
        x, y = w - lw - margin, margin

    else:
        x, y = margin, margin

    # Alpha blending if transparency exists
    if logo_resized.shape[2] == 4:

        alpha = logo_resized[:, :, 3] / 255.0

        for c in range(3):
            frame[y:y+lh, x:x+lw, c] = (
                alpha * logo_resized[:, :, c]
                + (1 - alpha) * frame[y:y+lh, x:x+lw, c]
            )

    else:
        frame[y:y+lh, x:x+lw] = logo_resized

    return frame


# def calculate_angle(a, b, c):
#     """
#     Calculates angle between 3 landmarks.
#     """

#     a = [a.x, a.y]
#     b = [b.x, b.y]
#     c = [c.x, c.y]

#     ba = [a[0] - b[0], a[1] - b[1]]
#     bc = [c[0] - b[0], c[1] - b[1]]

#     cosine_angle = (
#         (ba[0] * bc[0] + ba[1] * bc[1]) /
#         (
#             math.sqrt(ba[0]**2 + ba[1]**2)
#             * math.sqrt(bc[0]**2 + bc[1]**2)
#             + 1e-6
#         )
#     )

#     angle = math.degrees(math.acos(cosine_angle))

#     return int(angle)


# =========================================================
# === Pose skeleton connections
# =========================================================

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32)
]


# =========================================================
# === Directories
# =========================================================

save_dir = os.path.join(os.getcwd(), "screenshots")
video_dir = os.path.join(os.getcwd(), "videos")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)


# =========================================================
# === Load logo
# =========================================================

logo = None

logo_path = resource_path("logo.png")

if not os.path.exists(logo_path):
    print("Logo not found:", logo_path)

else:
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    if logo is None:
        print("Could not load logo.")


# =========================================================
# === MediaPipe Tasks setup
# =========================================================

model_path = resource_path("pose_landmarker_lite.task")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Could not find model file:\n{model_path}"
    )

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = PoseLandmarker.create_from_options(options)


# =========================================================
# === Video recording state
# =========================================================

recording = False
out = None


# =========================================================
# === Fullscreen kiosk window
# =========================================================

cv2.namedWindow("Pose", cv2.WND_PROP_FULLSCREEN)

cv2.setWindowProperty(
    "Pose",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# =========================================================
# === Webcam setup
# =========================================================

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps <= 0:
    fps = 30

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# =========================================================
# === Main loop
# =========================================================

flash_end_time = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # =========================================
    # Convert image for MediaPipe
    # =========================================

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(time.time() * 1000)

    results = pose.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # =========================================
    # Draw pose skeleton
    # =========================================

    if results.pose_landmarks:

        for pose_landmarks in results.pose_landmarks:

            h, w, _ = frame.shape

            # Draw connections
            for connection in POSE_CONNECTIONS:

                start_idx = connection[0]
                end_idx = connection[1]

                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]

                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    4
                )

            # Draw landmarks
            for landmark in pose_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 255),
                    -1
                )

            # =========================================
            # Example realtime angle calculation
            # =========================================

            # try:
            #     # Frame flipped:
            #     # use LEFT arm for user's RIGHT arm

            #     r_shoulder = pose_landmarks[11]
            #     r_elbow = pose_landmarks[13]
            #     r_wrist = pose_landmarks[15]

            #     r_elbow_angle = calculate_angle(
            #         r_shoulder,
            #         r_elbow,
            #         r_wrist
            #     )

            #     cv2.putText(
            #         frame,
            #         f"Right Elbow: {r_elbow_angle} deg",
            #         (10, 40),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         1,
            #         (0, 255, 0),
            #         2
            #     )

            # except Exception:
            #     pass

    # =========================================
    # Overlay logo
    # =========================================

    if logo is not None:
        frame = overlay_logo(
            frame,
            logo,
            position="bottom-left",
            scale=0.25
        )

    # =========================================
    # Flash effect
    # =========================================

    if time.time() < flash_end_time:

        overlay = frame.copy()
        overlay[:] = (255, 255, 255)

        alpha = 0.4

        frame = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0
        )

    # =========================================
    # Recording indicator
    # =========================================

    if recording:

        cv2.circle(
            frame,
            (50, 50),
            15,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "REC",
            (75, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # =========================================
    # Display
    # =========================================

    cv2.imshow("Pose", frame)

    # =========================================
    # Keyboard controls
    # =========================================

    key = cv2.waitKey(5) & 0xFF

    # ESC
    if key == 27:
        break

    # SPACEBAR = screenshot
    elif key == 32:

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        filepath = os.path.join(
            save_dir,
            f"screenshot_{timestamp}.png"
        )

        cv2.imwrite(filepath, frame)

        print(f"Saved screenshot: {filepath}")

        flash_end_time = time.time() + 0.2

    # R = record 5s clip
    elif key == ord('r'):

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        video_filename = os.path.join(
            video_dir,
            f"clip_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(
            video_filename,
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        print(f"Recording clip: {video_filename}")

        recording = True

        for i in range(5 * fps):

            ret, rec_frame = cap.read()

            if not ret:
                break

            rec_frame = cv2.flip(rec_frame, 1)

            # Process pose
            rgb_rec = cv2.cvtColor(
                rec_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_rec_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_rec
            )

            timestamp_ms = int(time.time() * 1000)

            rec_results = pose.detect_for_video(
                mp_rec_image,
                timestamp_ms
            )

            # Draw pose
            if rec_results.pose_landmarks:

                for pose_landmarks in rec_results.pose_landmarks:

                    h, w, _ = rec_frame.shape

                    for connection in POSE_CONNECTIONS:

                        start = pose_landmarks[connection[0]]
                        end = pose_landmarks[connection[1]]

                        x1 = int(start.x * w)
                        y1 = int(start.y * h)

                        x2 = int(end.x * w)
                        y2 = int(end.y * h)

                        cv2.line(
                            rec_frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 0),
                            4
                        )

                    for landmark in pose_landmarks:

                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        cv2.circle(
                            rec_frame,
                            (x, y),
                            5,
                            (255, 0, 255),
                            -1
                        )

            # Add logo
            if logo is not None:

                rec_frame = overlay_logo(
                    rec_frame,
                    logo,
                    position="bottom-left",
                    scale=0.25
                )

            # REC indicator
            cv2.circle(
                rec_frame,
                (50, 50),
                15,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                rec_frame,
                "REC",
                (75, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            out.write(rec_frame)

            cv2.imshow("Pose", rec_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        recording = False

        out.release()

        print(f"Saved clip: {video_filename}")


# =========================================================
# === Cleanup
# =========================================================

pose.close()

cap.release()

cv2.destroyAllWindows()