# Markerless-Mediapipe-Demo

A real-time motion capture demonstration application using [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) and OpenCV.
This project allows users to stand in front of a webcam or connected camera and view real-time pose tracking overlays on-screen. Additional features include:
+ Real-time pose skeleton overlay
+ Single-person and multi-person tracking modes
+ Screenshot capture
+ Automatic 5-second video recording
+ Logo overlay support
+ Kiosk-mode friendly display
+ Executable packaging support via PyInstaller

## Demo Modes
| **Script** | **Description** |
|------------|-----------------|
| `pose_demo_single.py`   | Optimized single-person pose tracking using MediaPipe Pose    |
| `pose_demo_multi.py` | Multi-person tracking (up to 3 people) using MediaPipe Pose Landmarker |

## Requirements
### Software
+ Python 3.11.13
+ Mediapipe 0.10.35
+ Conda (Miniconda or Anaconda recommended)
+ Webcam or USB camera

## Installation
### 1. Clone the repository
```
git clone https://github.com/ShrinersMAC/Markerless-Mediapipe-Demo`
cd ~/Markerless-Mediapipe-Demo`
```

### 2. Create the Conda environment (*recommended*)
The repository includes an environment.yaml file for setting up the recommended conda environment, which includes [a required version of Python (3.11.13)](https://www.python.org/downloads/release/python-31113/).

If needed, [Anaconda can be downloaded for free here](https://www.anaconda.com/download/success?reg=skipped).

Open an Anaconda Command Prompt. From the project root directory, create the environment using:

`conda env create -f environment.yaml`

Activate the environment:

`conda activate mediapipe-demo`

You can verify the Python version with:

`python --version`

Expected output:

`python 3.11.13`

If not using a conda environment, the primary requirements are:
+ Exact Python version (python=3.11.13)
+ Exact Mediapipe version (mediapipe==0.10.35): `pip install mediapipe==0.10.35`
+ Install missing dependencies as needed: `pip install cv2 time os sys`

## Running the Application
### Single-Person Tracking

`python pose_demo_single.py`

### Multi-Person Tracking

`python pose_demo_multi.py`

### Controls
| **Key** | **Action** | **Note** |
|------------|-----------------|---------|
| `Spacebar`   | Capture screenshot    | Creates and/or saves to `screenshots/` |
| `R` | Record 5-second video clip | Creates and/or saves to `videos/` |
| `Esc` | Exit application |   |

### Logo Overlay

To display a logo overlay:

Add a file named:
\logo.png
Place it in the project root directory.

The logo will automatically appear in the application window.

Example root directory:\
project_folder/\
│\
├── pose_demo_single.py\
├── pose_demo_multi.py\
├── pose_landmarker.task\
├── logo.png\
├── environment.yaml\
├── screenshots/\
└── videos/

## Packaging as an Executable (.exe)
This project can be packaged into a standalone Windows executable using PyInstaller

### 1. Install PyInstaller
`conda install conda-forge::pyinstaller` 
or
`pip install pyinstaller`

### 2. Build executable
**Single-Person**
`pyinstaller --onefile --noconsole
--add-data "logo.png;."
pose_demo_single.py`

**Multi-Person**
`pyinstaller --onefile --noconsole
--add-data "logo.png;."
--add-data "pose_landmarker.task;."
pose_demo_multi.py`

## Notes on Performance
### Single-Person Version
The single-person implementation is:
+ Faster
+ Lower latency
+ More stable on lower-end hardware

Recommended for:
+ Kiosk installations
+ Sports demonstrations
+ Interactive displays
+ Live events

### Multi-Person Version
The multi-person implementation:
+ Tracks up to 3 people simultaneously
+ Uses stable color assignment for each person
+ Requires more compute resources

A dedicated GPU is recommended for best performance.

## Troubleshooting
### Camera does not open
Try changing:

`cap = cv2.VideoCapture(0)`

to:

`cap = cv2.VideoCapture(1)`

if multiple cameras are connected.

### Controls (`Spacebar`, `R`, or `Esc`) not working

Ensure window displaying the demo is the active window by clicking on the video feed

### Executable cannot find model or logo

Ensure:

+ resource_path() is used
+ files are included with --add-data

### Multi-person model errors

Verify that:

`pose_landmarker.task`

exists in the project directory.

## Acknowledgments

Built using:
+ MediaPipe (developed by Google)
+ OpenCV
+ PyInstaller
