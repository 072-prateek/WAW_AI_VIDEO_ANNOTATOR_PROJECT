import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import os

class EyeBlinkDetector:

    def __init__(self, video_path = None):

        # loading video file or initializing webcam
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.video_filename = os.path.basename(video_path)
        else:
            self.cap = cv2.VideoCapture(0)
            self.video_filename = "webcam_feed"
        # face mesh detector object initialization
        self.detector = FaceMeshDetector(maxFaces=1)
        # landmark ids for both eyes
        self.left_eye_ids = [362, 263, 386, 374, 387, 373]
        self.right_eye_ids = [33, 133, 159, 145, 160, 144]
        self.all_ids = self.left_eye_ids + self.right_eye_ids
        # ear history lists for smoothing
        self.left_ear_list = []
        self.right_ear_list = []
        # thresholds for eye closed detection (will be overwritten by calibration thresholds that will be computed)
        self.left_thresh = 19
        self.right_thresh = 20
        # json data structure for logging frame-wise results
        self.frame_logs = {
            "video_filename": self.video_filename,
            "total_frames": 0,
            "labels_per_frame": {}
        }
        #  calibration parameters
        self.auto_calibrate = True            # for enabling or disabling auto calibration
        self.calib_frames = 150               # number of valid frame samples to collect for calibration
        self.p_low = 20                       # low percentile (closed-ish) 
        self.p_high = 80                      # high percentile (open-ish)
        self.smoothing_window = 4              # moving average window size used  for smoothing               
        self.reliable_count_for_calib = 5      # minimum reliable frames to attempt calibration

    def calc_EAR(self, face):

        # calculating eye aspect ratio for both eyes
        leftUp = face[386]
        leftDown = face[374]
        leftLeft = face[263]
        leftRight = face[362]
        leftleftup = face[387]
        leftleftdown = face[373]
        rightUp = face[159]
        rightDown = face[145]
        rightLeft = face[133]
        rightRight = face[33]
        rightleftup = face[160]
        rightleftdown = face[144]
        leftVer1, _ = self.detector.findDistance(leftUp, leftDown)
        leftVer2, _ = self.detector.findDistance(leftleftup, leftleftdown)
        leftVer = (leftVer1 + leftVer2) / 2.0
        leftHor, _ = self.detector.findDistance(leftLeft, leftRight)
        rightVer1, _ = self.detector.findDistance(rightUp, rightDown)
        rightVer2, _ = self.detector.findDistance(rightleftup, rightleftdown)
        rightVer = (rightVer1 + rightVer2) / 2.0
        rightHor, _ = self.detector.findDistance(rightLeft, rightRight)
        leftEAR = (leftVer / leftHor) * 100 if leftHor > 0 else 0 # so we don't divide by zero
        rightEAR = (rightVer / rightHor) * 100 if rightHor > 0 else 0
        return leftEAR, rightEAR, self.all_ids

    def compute_thresholds_from_samples(self, left_samples, right_samples, p_low = None, p_high = None):

        # percentile-based: closed -> low percentile, open -> high percentile, so we avoid the effect of outliers and then average to get thresholds
        if p_low is None: p_low = self.p_low
        if p_high is None: p_high = self.p_high
        if len(left_samples) < self.reliable_count_for_calib or len(right_samples) < self.reliable_count_for_calib:
            return None
        left_eye_closed = np.percentile(left_samples, p_low)
        left_eye_open = np.percentile(left_samples, p_high)
        right_eye_closed = np.percentile(right_samples, p_low)
        right_eye_open = np.percentile(right_samples, p_high)
        left_thresh = (left_eye_closed + left_eye_open) / 2
        right_thresh = (right_eye_closed + right_eye_open) / 2
        return left_thresh, right_thresh

    def run(self):

        # main loop for reading frames and detecting eyes
        frame_count = 0
        # calibration state for this run
        calibrated = False
        calib_left_samples = []
        calib_right_samples = []
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to read frame")
                break
            img, faces = self.detector.findFaceMesh(img, draw = False)
            if faces:
                face = faces[0]
                # calculating  EARs for this run
                leftEAR, rightEAR, points = self.calc_EAR(face)
                # smoothing values using small buffer here the value we used for smoothing window is 4
                self.left_ear_list.append(leftEAR)
                self.right_ear_list.append(rightEAR)
                if len(self.left_ear_list) > self.smoothing_window:
                    self.left_ear_list.pop(0)
                    self.right_ear_list.pop(0)
                avg_left_ear = sum(self.left_ear_list) / len(self.left_ear_list)
                avg_right_ear = sum(self.right_ear_list) / len(self.right_ear_list)
                # per-run auto calibration logic
                if self.auto_calibrate and (not calibrated):
                    # collecting only valid face samples for calibration
                    if len(calib_left_samples) < self.calib_frames:
                        calib_left_samples.append(leftEAR)
                        calib_right_samples.append(rightEAR)                       
                        # once we've collected enough valid samples computing thresholds
                        if len(calib_left_samples) >= self.calib_frames:
                            result = self.compute_thresholds_from_samples(calib_left_samples, calib_right_samples)
                            if result:
                                self.left_thresh, self.right_thresh = result
                                calibrated = True
                    else:
                        # safety fallback if something odd happens
                        calibrated = True
                # determining eye state and we assume closed if both eyes are below their respective thresholds
                if avg_left_ear < self.left_thresh and avg_right_ear < self.right_thresh:
                    eye_state = "Closed"
                else:
                    eye_state = "Open"
            else:
                self.left_ear_list.clear()
                self.right_ear_list.clear()
                eye_state = None
            # logging JSON data for each frame
            self.frame_logs["labels_per_frame"][str(frame_count)] = {
                "eye_state": eye_state,
            }
            frame_count += 1
            self.frame_logs["total_frames"] = frame_count
        self.cap.release()
        cv2.destroyAllWindows()
