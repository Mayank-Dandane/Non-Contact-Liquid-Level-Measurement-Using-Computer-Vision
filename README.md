Project Summary — Non-Contact Liquid Level Measurement Using Computer Vision

What We Built
A real-time, non-contact liquid level measurement system using a standard USB webcam and Python-based image processing. The system detects the water surface in any transparent container and outputs the measurement in millimeters — without any physical sensor touching the liquid.

Tech Stack

Language: Python 3
Libraries: OpenCV (cv2), NumPy, Matplotlib, SciPy
Hardware: USB webcam (720p), Windows PC
Backend: DirectShow (CAP_DSHOW) for reliable Windows camera access


System Architecture — 3 Core Modules
1. calibrate.py — Calibration Engine

Opens webcam feed with mouse callback registration
User clicks on known reference marks on the container (100ml to 900ml markings)
Each click captures the pixel Y-coordinate mapped to a known real-world height in mm
Runs NumPy linear regression (np.polyfit) on the pixel-Y vs real-mm dataset
Computes R² (coefficient of determination) to evaluate calibration quality
Achieved R² = 0.9973 — confirming near-perfect linearity
Saves slope, intercept, R², and all calibration points to calibration.json

2. main.py — Real-Time Detection Pipeline
Every video frame goes through this pipeline:
Raw BGR frame
    ↓ cv2.cvtColor → Grayscale
    ↓ cv2.GaussianBlur (7×7 kernel) → Noise suppression
    ↓ cv2.Canny (threshold 20–60) → Edge map
    ↓ cv2.dilate (3×3 kernel) → Edge strengthening
    ↓ np.sum(edges, axis=1) → Row-wise edge strength array
    ↓ np.convolve (5-point moving average) → Smoothed row sums
    ↓ np.argmax → Strongest horizontal edge = water surface row
    ↓ Proportional formula → level_mm
Proportional conversion formula:
level_mm = (1 - top_row_local / ROI_height_px) × real_ROI_height_mm
Empty bottle detection:

If max(row_sums_smooth) < EMPTY_THRESHOLD (80) → no valid water surface exists
Green line locks to bottom of ROI, display shows "EMPTY / 0.0 mm"
Prevents erratic readings from noise and container markings when bottle is empty

ROI system:

User drags mouse to define a rectangular Region of Interest inside the container
All processing is confined strictly within this ROI — ignores everything outside
User presses H to input the real physical height of the ROI in mm — makes the system container-agnostic

Key controls:

L — logs current reading with timestamp to level_log.csv
E — toggles edge view to visualize what Canny is detecting
R — resets ROI
H — sets real-world ROI height in mm

3. analysis.py — Performance Analysis
Reads calibration.json and level_log.csv and generates a 4-panel Matplotlib report:

Calibration curve — pixel Y vs actual mm scatter plot with regression line and R² annotation
Level vs time — time-series plot of all logged readings from CSV
Error bar chart — absolute % error at each test level (green = ≤2mm, red = >2mm)
Accuracy table — actual vs measured vs error vs % error for all test points
Prints resolution (mm/pixel), max error, mean error, and standard deviation to terminal


Key Instrumentation Parameters Achieved
ParameterValueCalibration R²0.9973Resolution~0.41 mm/pixelDetection methodCanny edge detectionFrame rate~10 FPSMeasurement unitMillimeters (mm)Calibration typeLinear regression + proportional ROI

Key Technical Decisions Made
Why Canny over HSV color thresholding:
Initially attempted HSV-based water detection — failed because the bottle was opaque-looking under camera lighting and background noise was too high. Switched to Canny edge detection which detects the water surface purely based on intensity gradient — works regardless of water color or container color.
Why proportional ROI mode over calibration.json:
The calibration.json approach is container-specific — recalibration needed every time you change containers. The proportional ROI mode makes the system fully generic — draw ROI anywhere, enter its real height once, system works for any container.
Why row-wise argmax over topmost edge:
Initially used the topmost detected edge (first non-zero row). This caused false detections from bottle neck text and markings at the top of the ROI. Switching to argmax of smoothed row sums selects the strongest horizontal edge — which is always the water surface since it produces the most continuous, pixel-dense horizontal line in the ROI.
Why EMPTY_THRESHOLD:
Without it, an empty bottle still has edges from container walls, printed markings, and noise — causing the green line to jump randomly. The threshold gates detection: if no row has sufficient edge strength, the system locks to 0mm and displays EMPTY.

Output Files Generated
FileContentscalibration.jsonslope, intercept, R², calibration pointslevel_log.csvtimestamp + level_mm for each logged readinganalysis_report.png4-panel performance analysis figure

What Makes It Non-Contact
The webcam is the only sensor — positioned 20–40cm from the container, at mid-height. Nothing touches the liquid at any point. The entire measurement chain is optical → digital → computational, making it safe for corrosive, sterile, or hazardous liquids where physical sensors would fail.