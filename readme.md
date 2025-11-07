## AI Video Annotation Service — Eye State & Posture Detection
# Overview

This project implements an AI-powered video annotation service that analyzes a person’s eye state and posture frame-by-frame from a pre-recorded laptop-camera video.
It was built as part of the Wellness at Work (WAW) AI App Development Engineer technical challenge.

The service exposes a single FastAPI endpoint /annotate which accepts a video file (.mp4 or .avi), processes it locally using lightweight computer-vision logic, and returns a JSON object containing eye-state and posture labels for every frame.

## Approach and Reasoning
# Research and Model Exploration

Before deciding on the final approach, I explored multiple ways to solve this problem:

Open-Source VLMs (Qwen-VL, LLaVA, etc.)
I initially tested some open-source vision-language models for posture and eye detection.
However, their latency was very high, and GPU resource requirements made them unsuitable for real-time or demo-ready deployment on local hardware.

CNN-based Posture Detection
I initially considered training a lightweight CNN using a posture dataset. However, creating a robust dataset and labeling it properly would have required substantial time and GPU-based computing resources, which I didn’t have access to. Given the tight development window of just 5–7 days, I decided to pivot to a more practical, explainable approach using mathematical landmark detection with Mediapipe and cvzone. This allowed me to build a working system quickly while keeping it interpretable and efficient.

# Final Choice

Eye detection: robust geometric approach (Eye Aspect Ratio) — accurate, fast, and reliable.

Posture detection: landmark-based geometric analysis using the nose and shoulder positions. While estimating posture from a front-facing view is inherently difficult without a CNN, I implemented several safety nets and calibration mechanisms to make it as consistent as possible.

##  Detection Logic — Eye State (EAR-Based)

The eye detection and state classification system is built using the Eye Aspect Ratio (EAR) method combined with Mediapipe’s FaceMesh for accurate landmark tracking.
This approach is lightweight, interpretable, and does not depend on heavy CNN models — making it ideal for fast, CPU-only inference.

# Core Idea

The Eye Aspect Ratio (EAR) measures the ratio between the vertical and horizontal distances of specific eye landmarks.
When the eyes close, the vertical distance shrinks significantly, reducing the EAR value.
By monitoring how this ratio changes over time, the model can reliably determine whether the eyes are open or closed.

# Step-by-Step Logic
1. Initialization
When the EyeBlinkDetector class is initialized, it either:
Loads a video from the given path, or
Opens a live webcam stream (if no path is provided).
It then initializes:
A FaceMeshDetector (from cvzone) for facial landmark tracking.
Separate landmark ID lists for both the left and right eyes.
These IDs correspond to specific points around the eyelids and eye corners.
Smoothing buffers to stabilize the detection results over short time windows.
Default thresholds (which will later be replaced by calibration values).
Calibration-related parameters like number of frames to collect, percentile cutoffs, and smoothing window size.
This setup ensures that the system adapts dynamically to each user and lighting condition.

2. EAR Calculation
The function calc_EAR() takes the detected face landmarks as input and performs the following:
Extracts six key points around each eye — upper, lower, left, and right corners.
Calculates two vertical distances per eye (upper-lower eyelid pairs) and one horizontal distance (between eye corners).
Computes the average vertical distance for stability.
Finally, calculates the EAR as:
𝐸𝐴𝑅 = (Vertical Distance / Horizontal Distance) × 100
The scaling factor (×100) just makes the values easier to interpret numerically.
If horizontal distance is zero (which can happen with noisy landmarks), EAR is set to 0 to avoid division errors.

3. Calibration Phase
To handle differences in face size, lighting, or camera angles, the detector performs auto-calibration during the first 150 valid frames.
For each frame with a clearly detected face:
It collects the left and right EAR values in separate lists.
After enough samples (150 frames by default), it computes two percentiles:
20th percentile → roughly corresponds to “eyes nearly closed”
80th percentile → roughly corresponds to “eyes fully open”
By averaging these two values, we get custom thresholds for that individual’s eye geometry.
This percentile-based calibration makes the detector robust to noise, outliers and avoids misclassification caused by temporary blinks or head movements.
If too few reliable samples are collected, default thresholds (19, 20) are used as a fallback.

4. Real-Time Detection and Smoothing
Once calibrated, the system continuously reads frames and:
Detects face landmarks.
Computes the left and right EAR values.
Maintains a rolling buffer (window size = 4) to smooth out sudden changes.
This short smoothing window eliminates flickering predictions due to momentary noise or face-mesh misalignment.
The smoothed EAR values are compared with the dynamic thresholds:
If both eyes’ EARs fall below their respective thresholds → the frame is labeled as “Closed”.
Otherwise → labeled as “Open”.

5. Handling Edge Cases
If no face or eyes are detected in a frame, the detector:
Clears the smoothing buffers to avoid stale data.
Skips labeling that frame (sets eye_state = None).
This prevents false detections when the user looks away or moves out of frame.

6. Frame-Wise Logging
For every frame, the detector logs a simple JSON entry:
"12": {
  "eye_state": "Closed"
}
All frame-level results are stored under self.frame_logs, along with the total frame count and video filename.
This structured output makes it easy to merge later with the posture results during the annotation step.

7. Runtime & Performance
The entire process runs in real time on a CPU, with average processing times of a few milliseconds per frame.
Since no neural network inference is required, the detector is fast, stable, and interpretable.

## Detection Logic — Posture (Landmark-Based Ratio Analysis)
The posture detection module estimates whether a person’s upper body is straight or hunched based on geometric relationships between body landmarks.
Instead of relying on a deep neural network, it uses Mediapipe’s Pose landmark model and interprets posture using simple normalized ratios.
This makes it fast, explainable, and suitable for lightweight local inference.

## Core Idea

The detector measures the vertical distance between the nose and shoulder midpoint, normalized by the shoulder width.
This creates a dimensionless ratio that describes how upright the upper body is:

Posture Ratio = ((Shoulder Midpoint_𝑦 −Nose_𝑦) / Shoulder Width) × 100
A larger ratio indicates a straighter posture (nose higher above shoulders),
while a smaller ratio suggests the person is slouched forward.

# Step-by-Step Logic
1. Initialization

When the PostureDetector class is initialized, it:
Loads either a video file or opens a webcam feed.
Sets up a PoseDetector from cvzone, fine-tuned for this use case:
staticMode = True → forces re-detection each frame (ideal for pre-recorded videos).
modelComplexity = 2 → medium model, balancing accuracy and performance.
smoothLandmarks = True → helps reduce jitter in landmark positions.
Detection and tracking confidences are lowered (0.3) to increase robustness under variable lighting or camera quality.
Initializes smoothing buffers, posture thresholds, and calibration parameters.
The class is designed to handle wide variations in distance, lighting, and user orientation gracefully.

2. Ratio Calculation
The calc_ratio() function forms the backbone of the detector.
It extracts three key points from each frame:
Nose (ID 0), Left Shoulder (ID 11), Right Shoulder (ID 12)
Using these:
It computes the midpoint between the shoulders to get the body’s central reference.
Calculates shoulder width (Euclidean distance) to normalize the measurement.
Finds the vertical offset between the nose and shoulder midpoint.
Then computes the normalized vertical ratio = (vertical offset / shoulder width) × 100.
This normalized ratio gives a consistent scale across different body types and distances from the camera.
The function also filters out unreliable frames:
Frames with too small a shoulder width (person too far away) are skipped.
Frames with excessive head turning (nose shifted sideways > 55% of shoulder width) are ignored.
This prevents incorrect classification when the person is not facing the camera.

3. Calibration Phase
Since every person’s proportions and camera angles differ, the detector auto-calibrates during the first 150 reliable frames.
For each valid frame:
It records the computed posture ratio.
Once enough samples are gathered, it calculates:
20th percentile (represents more “hunched” frames)
80th percentile (represents more “straight” frames)
The average of these two becomes the posture threshold for that individual.
This percentile-based calibration ensures that outliers (like extreme leaning or movement) don’t distort the threshold.
If too few reliable frames are detected, the detector falls back to a default threshold of 18.0.

4. Real-Time Smoothing and Detection
After calibration:
Each new frame’s ratio is added to a short smoothing buffer (window size = 4).
The moving average of this buffer is computed to reduce noise and jitter from landmark shifts.
The averaged ratio is then compared against the threshold:
If avg_ratio >= threshold → posture is “Straight”
If avg_ratio < threshold → posture is “Hunched”
This simple rule-based classification works surprisingly well for front-facing seated postures in webcam setups.

5. Robustness Checks
Several safety mechanisms are built in:
If landmarks aren’t detected → posture for that frame is skipped (None).
When frames are unreliable (too far, side-turned), the smoothing buffer is cleared.
Calibration is only computed if enough good samples exist.
This ensures that momentary detection failures don’t cause unstable or flickering predictions.

6. Frame-Wise Logging
For every frame processed, the system logs a JSON entry of the form:
"24": {
  "posture": "Hunched"
}
All logs are stored under self.frame_logs along with metadata like total frames and video filename.
This structured logging is used later to merge posture and eye results together for unified annotation output.

7. Runtime & Performance
The entire process runs in real time on a CPU, with average processing times of a few milliseconds per frame.
Since no neural network inference is required, the detector is fast, stable, and interpretable.

## Annotation API and Integration Logic (FastAPI)
This module (app.py) serves as the core integration point that ties together the eye-state and posture detection subsystems.
It provides a single /annotate endpoint that accepts a user’s video file (and optionally a ground truth JSON) and outputs a unified annotation file containing frame-wise posture and eye state labels.
The API is designed to be lightweight, asynchronous, and fully automated — allowing both detectors to run concurrently using Python’s asyncio and thread-based execution.

1. Initialization
The script begins by importing all the necessary dependencies:
FastAPI, UploadFile, and File to handle incoming HTTP requests.
JSONResponse for structured error or success responses.
os, shutil, and json for file handling and serialization.
asyncio for asynchronous concurrency.
The three core modules:
EyeBlinkDetector — for eye state labeling.
PostureDetector — for body posture labeling.
F1ScoreEvaluator — for computing performance metrics if ground truth data is available.
Then, an instance of the FastAPI app is created:
app = FastAPI()
This sets up the foundation for defining API routes and handling requests.

2. The "/annotate" Endpoint
The endpoint is defined as:
@app.post("/annotate")
It accepts:
video: A required video file (.mp4 or .avi).
groundtruth: An optional JSON file containing true frame-level labels.

3. File Validation and Saving
When a request hits the endpoint:
A directory named uploaded_videos is created if it doesn’t already exist.
The API validates that the uploaded video file has a supported extension (.mp4 or .avi).
If not, it returns a 400 Bad Request error with a descriptive message.
The uploaded video is saved locally inside the uploaded_videos directory.
If a ground truth file is included, it’s also saved to the same directory.
This ensures all temporary data is neatly organized for processing.

4. Detector Initialization
Once the video is stored:
Two independent detectors are instantiated one for posture and one for eye state:
eye_state_detector = EyeBlinkDetector(save_path)
posture_detector = PostureDetector(save_path)
Each detector runs its own logic and writes frame-level annotations internally.

5. Parallel Execution with asyncio
To minimize processing time, both detectors are executed concurrently using asynchronous threads.
Two helper coroutines are defined:
async def run_eye_detector(): ...
async def run_posture_detector(): ...
Each runs the respective detector’s .run() method in a background thread and saves its raw output JSON (eye_state_output.json and posture_output.json).
These two detectors are then launched together:
await asyncio.gather(run_eye_detector(), run_posture_detector())
This ensures that while one process is busy (for example, doing CPU work or waiting for I/O), the other continues to execute — effectively cutting total runtime nearly in half.

6. Merging the Outputs and Deleting the Temporary JSON Files
After both detectors finish, another helper coroutine merges their outputs:
async def merge_json_files(): ...
This function:
Reads both JSON files generated by the detectors.
Aligns them frame by frame (based on the smaller of their total frame counts).
Combines the labels into a single JSON structure like this:
{
  "video_filename": "input_video.mp4",
  "total_frames": 450,
  "labels_per_frame": {
    "0": { "eye_state": "Open", "posture": "Good" },
    "1": { "eye_state": "Closed", "posture": "Slouching" }
  }
}
Saves the merged JSON under:
uploaded_videos/<video_name>_output.json
Deletes the temporary detector outputs to keep the workspace clean.
This merged annotation file is the final deliverable from the endpoint.

7. F1 Score Evaluation (Optional)
If a groundtruth file is provided, the API automatically evaluates the detection performance by calling:
evaluator = F1ScoreEvaluator(output_path, groundtruth_path)
f1_scores = evaluator.run()
It then prints both Eye State F1 Score and Posture F1 Score to the console.
If no ground truth file is uploaded, the API simply skips this step and continues.

8. Returning the Response
The endpoint finally returns the merged JSON as a FastAPI JSONResponse, allowing clients (like your frontend or testing scripts) to visualize or store the frame-level results directly.
If any error occurs at any stage, a clean JSON-formatted error response is returned with a 500 status code and an explanatory message.

9. Why This Architecture Works Well
Asynchronous: Both detectors run concurrently, maximizing resource usage.
Modular: The eye, posture, and scoring modules are completely decoupled — easy to update or replace individually.
Explainable: Each phase (detection, merging, scoring) produces traceable outputs.
Lightweight: No GPU or deep model dependencies.
Flexible: Accepts any .mp4 or .avi file, and optionally evaluates accuracy.

## Testing the API via cURL / Api Usage
Endpoint
POST /annotate
# Case 1: Without Ground Truth File
To simply analyze a video and get the combined frame-wise annotations:

curl.exe -X POST "http://127.0.0.1:8000/annotate" -H "accept: application/json" -F "video=@C:\Users\Stylebender07\OneDrive\Desktop\Project\testvideo.mp4"


(For PowerShell users, make sure to use the ^ for line breaks, or put it all on one line.)

# Case 2: With Ground Truth File
If you also have a labeled JSON file for evaluation:

curl.exe -X POST "http://127.0.0.1:8000/annotate" -H "accept: application/json" -F "video=@C:\Users\Stylebender07\OneDrive\Desktop\Project\testvideo.mp4" -F "groundtruth=@C:\Users\Stylebender07\OneDrive\Desktop\Project\groundtruth.json"

This version triggers both:
Frame-wise annotation generation, and
F1 score computation for both modules.

## System Flow

The system follows a modular and asynchronous architecture designed for real-time video annotation with concurrent execution of detection modules. The entire process from video upload to annotated JSON output runs automatically through the FastAPI /annotate endpoint.

1. Video Upload and Validation

The user uploads a video via the FastAPI /annotate endpoint. The API accepts .mp4 or .avi formats only.
Uploaded videos are first saved locally under the uploaded_videos/ directory, as OpenCV cannot efficiently process raw byte streams.
If provided, the optional ground-truth JSON file is also saved in the same directory for later evaluation.

2. Detector Initialization and Asynchronous Execution

Once the files are saved, two detection modules are initialized:
EyeBlinkDetector –> handles blink and eye state detection using Mediapipe’s FaceMesh and EAR thresholds.
PostureDetector –>  detects upper-body posture and computes geometric ratios using Mediapipe’s Pose landmarks.

Both detectors run asynchronously and independently using Python’s asyncio.gather() method.
Internally, each detector processes all video frames using OpenCV and stores frame-level results in memory (frame_logs), which are later dumped as separate JSON files:

# eye_state_output.json

# posture_output.json

3. Frame-Level Merging of Results

After both detectors complete, their outputs are merged into a single structured JSON file.
The merge operation uses a simple linear pass over frames (O(n) complexity), combining both eye and posture predictions for each frame.
The merged output contains:

Video filename
Total number of processed frames
Per-frame annotations with "eye_state" and "posture" labels
This combined file is saved as <video_name>_output.json inside the uploaded_videos/ directory.
Temporary JSONs generated by individual detectors are deleted automatically after merging.

4. Optional F1 Evaluation

If the user provides a ground-truth JSON during upload, the system automatically computes F1 scores for both the eye-state and posture detection outputs. The F1ScoreEvaluator module compares predicted and ground-truth frame labels, providing a concise performance summary.
If no ground-truth is provided, this step is skipped without affecting the annotation process.

5. Output Delivery and Cleanup

The final merged annotations are returned as a JSON API response via FastAPI.
All intermediate files (temporary detector JSONs and unused ground-truths) are safely removed to maintain a clean workspace.
This ensures that only the final merged JSON output remains in the directory.

6. System Characteristics

The workflow is fully asynchronous, minimizing idle time and optimizing CPU utilization.
Each detection task is isolated and failure-tolerant, meaning if one module fails, detailed logs are provided without crashing the full pipeline.
The linear-time merging process ensures near real-time behavior for short videos and smooth scaling for longer inputs.
This architecture supports easy future extension (e.g., emotion detection or gesture recognition modules) with minimal restructuring.

# Visualization Testers
The project also includes tester scripts to visualize landmarks, plot graphs (e.g., EAR over time), and debug posture ratios.
These were built mainly for research and internal validation, helping verify how smoothing, calibration, and ratios behave visually.

## Cost Estimation & Methodology
This project’s cost and effort estimation follows a Bottom-Up Approach, where every part of the system from research to deployment was broken down and individually analyzed. The goal was to build a fully functional, explainable posture and eye state detection system within 5–7 days, using limited CPU-only hardware while maintaining measurable accuracy and interpretability.

# Bottom - UP approach for Cost Estimation

1. Breakdown of Tasks
The project was divided into multiple core modules, each with its own estimated effort range (Optimistic, Most Likely, and Pessimistic).
Initial Research & Model Selection: Compared CNN-, VLM-, and Mediapipe-based methods mathematically for feasibility and interpretability under time and resource constraints. (6–10 hours)
Dataset Exploration & Calibration Design: Reviewed sample videos and designed normalization logic using EAR (Eye Aspect Ratio) and posture ratios. (4–8 hours)
Eye State Detection Module: Implemented blink detection using Mediapipe’s FaceMesh and EAR thresholding approach. (5–10 hours)
Posture Detection Module: Built a geometric-based slouch detection system using Pose landmarks and coordinate ratios. (5–9 hours)
Frame-wise Labeling & JSON Export: Created per-frame annotations and structured JSON outputs for API use. (2–4 hours)
FastAPI Integration: Developed an asynchronous "/annotate" endpoint to enable parallel video analysis and output merging. (3–6 hours)
F1 Evaluation Integration: Added automated F1-score computation using ground-truth labels for validation. (2–5 hours)
Optimization & Logging: Enhanced runtime stability with smoothing filters, calibration fallback, and logging mechanisms. (2–4 hours)
Documentation & README Preparation: Authored setup guides, endpoint usage notes, and system documentation. (3–5 hours)
Testing & Debugging: Conducted multi-video validation, frame mapping checks, and asynchronous stress tests. (3–7 hours)

2. Total Estimated Effort

Based on the breakdown:
Optimistic scenario: Around 35 hours of total development time.
Most likely scenario: Around 50 hours of total development time.
Pessimistic scenario: Around 65 hours in case of extended testing or refactoring needs.

3. Adding a 20% Contingency Buffer

To accommodate potential delays and debugging iterations, a 20% contingency buffer was added:
Optimistic: 42 hours
Most Likely: 60 hours
Pessimistic: 78 hours

4. Development Cost (Assuming $20/hour)

Based on the above estimates:
Optimistic: $840
Most Likely: $1,200
Pessimistic: $1,560
This range represents the full development and testing cost for a single-developer implementation.

5. Cost per Minute of Annotated Video

On a CPU-only setup, the system processes roughly 1 minute of video in 2.5 – 8 minutes, depending on model complexity, lighting, and frame rate.
From empirical tests:
Using model_complexity = 1 (fast mode): 145 frames took about 12 seconds, giving 12 FPS.
Using model_complexity = 2 (accurate mode): 145 frames took about 38 seconds, giving 3.8 FPS.
Extrapolating for a 10-minute video (18,000 frames):
# Complexity 1: 25–30 minutes
# Complexity 2: 75–80 minutes

If we consider the total development and compute cost:
Cost per minute of annotated video = (Total Development + Compute Cost) / Total Minutes Processed
For example, if a 10-minute video takes 50 minutes to process on average,
$1200 (development cost) / 10 (minutes) = $120 per video minute (including development + compute).
With GPU acceleration or batch processing in cloud environments, this can reduce to $40 – $50 per minute.

6. Local vs Cloud Runtime Cost

The approximate runtime and cost comparison across environments:
Local (CPU-only, Intel i7 / 16GB RAM): Takes around 25–80 minutes per 10-minute video, costing roughly $0.01 – $0.05 (mainly electricity).
Cloud (CPU VM, 2 vCPUs / 8GB RAM): Takes about 20–75 minutes, costing roughly $0.25 – $0.95 per run depending on provider rates.
Cloud (GPU VM, NVIDIA T4 / 16GB RAM): Takes 5–12 minutes, costing approximately $0.80 – $1.20 per run.

# F1-Score Evaluation

I implemented an independent script (f1_score_evaluator.py) that compares predicted outputs with provided ground-truth JSON files.
It computes F1-scores for both Eye State and Posture based on frame-level matches.
I chose not to pre-compute or report any fixed F1-scores because testing on a single local video would bias the results heavily.
The evaluator is included so the scores can be fairly computed during the official demo with the real ground-truth dataset.

# Local Setup
git clone https://github.com/<your_username>/WAW_Project.git
cd WAW_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt # installing all the necessary packages 
uvicorn app:app --reload # running the app locally

or

source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt # installing the necessary packages
uvicorn app:app --reload # running the app locally

Then open: http://127.0.0.1:8000/docs
 for an interactive Swagger UI.

## Project Structure
WAW_Project/
│
├── app.py                     # FastAPI main app
├── eye_state_detector.py      # Eye state detection (EAR-based)
├── posture_detector.py        # Posture detection (landmark-based)
├── f1_score_evaluator.py      # Optional F1 score calculator for validation
│
├── visualization_testers/     # Helper scripts for visual landmark debugging 
│                              # Includes Eye State and Posture detector testers 
│                              # with GUI and live plots for understanding model behavior
│
├── Images_For_LandMark_Values/ # Contains FaceMesh and Pose landmark reference maps
│                               # with corresponding numerical landmark indices for calibration
│
├── uploaded_videos/           # Auto-created folder for uploaded video files
│
├── requirements.txt           # Lists all dependencies needed to set up the virtual environment
│
└── README.md                  # Complete documentation for setup, usage, and explanation


# Future Work

1. Replace geometric posture detection with a small CNN trained on annotated posture data to make it more Robust and accurate.
2. Dockerize and deploy to AWS Lambda/ECS for scalability.

# Developer Details

Developed by: Prateek Dwivedi
Email: prateekdaddwivedi@gmail.com