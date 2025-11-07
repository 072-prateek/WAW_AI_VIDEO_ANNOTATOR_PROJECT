import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import math
import os

class PostureDetector:
    def __init__(self, video_path = None):

        # loading video file or initializing webcam
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.video_filename = os.path.basename(video_path)
        else:
            self.cap = cv2.VideoCapture(0)
            self.video_filename = "webcam_feed"
        # pose detector object initialization
        self.detector = PoseDetector(staticMode = False, #In static mode, detection is done on each image best for our usecase but increases latency so I turned it off.
                                     modelComplexity = 2, # 1 = lite, 2 = Medium , 3 = heavy so I used 2 for finding the sweet spot between accuracy and speed.
                                     smoothLandmarks = True, # landmark smoothing, helps reduce jitter
                                     enableSegmentation = False, # disabled segmentation for faster processing
                                     detectionCon = 0.3, # Mininum Detection Confidence Threshold, reduced to 0.3 to detect more frames
                                     trackCon = 0.3)    # Minimum Tracking Confidence Threshold, reduced to 0.3 to detect more frames
        # ratio smoothing buffer
        self.ratio_list = []
        # thresholds for posture detection (overwritten after calibration)
        self.posture_thresh = 18.0
        # json data structure for logging frame wise results
        self.frame_logs = {
            "video_filename": self.video_filename,
            "total_frames": 0,
            "labels_per_frame": {}
        }
        # calibration parameters
        self.auto_calibrate = True # for enabling or disabling auto calibration
        self.calib_frames = 150 # number of valid frame samples to collect for calibration
        self.p_low = 20 # low percentile
        self.p_high = 80 # high percentile
        # additional settings for robustness
        self.min_shoulder_width = 10        # less than this => too small/invalid
        self.max_head_turn_ratio = 0.55        # nose_x offset / shoulder_width -> this -> turned -> unreliable
        self.smoothing_window = 4              # moving average window size used for smoothing
        self.reliable_count_for_calib = 5      # minimum reliable frames to attempt calibration

    def calc_ratio(self, lmList):

        # extract key landmarks manually
        if len(lmList) <= 12:
            return 0.0  # insufficient landmarks detected
        nose_x, nose_y = lmList[0][0:2]
        left_shoulder_x, left_shoulder_y = lmList[11][0:2]
        right_shoulder_x, right_shoulder_y = lmList[12][0:2]
        # calculate shoulder midpoint (average of both shoulders)
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2.0
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2.0
        # compute shoulder width using both x and y 
        shoulder_dx = right_shoulder_x - left_shoulder_x
        shoulder_dy = right_shoulder_y - left_shoulder_y
        shoulder_width = math.hypot(shoulder_dx, shoulder_dy)
        # check reliability too small width = subject too far or poor detection
        if shoulder_width < self.min_shoulder_width:
            return 0.0
        # find vertical distance between nose and shoulder midpoint
        vertical_offset_y = shoulder_mid_y - nose_y 
        normalized_vertical = vertical_offset_y / shoulder_width  # normalized by body scale
        # find horizontal distance to detect head turn (helps ignore side poses)
        horizontal_offset_x = nose_x - shoulder_mid_x
        head_turn_ratio = abs(horizontal_offset_x) / shoulder_width
        # reject frames where head is too turned sideways
        if head_turn_ratio > self.max_head_turn_ratio:
            return 0.0
        # converting normalized vertical ratio to a percentage-like score
        posture_score = max(0.0, normalized_vertical * 100.0)
        return posture_score

    def compute_threshold_from_samples(self, samples, p_low = None, p_high = None):

        # percentile-based: slouched -> low percentile, straight -> high percentile, so we avoid the effect of outliers and then average to get threshold
        if p_low is None: p_low = self.p_low
        if p_high is None: p_high = self.p_high
        if len(samples) < self.reliable_count_for_calib:
            return None
        low_p = np.percentile(samples, p_low)
        high_p = np.percentile(samples, p_high)
        threshold = (low_p + high_p) / 2.0
        return threshold

    def run(self):

        # main loop for reading frames and detecting posture
        frame_count = 0
        # calibration state for this run
        calibrated = False
        calib_samples = []
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to read frame")
                break
            img = self.detector.findPose(img, draw = True)
            lmList, _ = self.detector.findPosition(img, draw = False, bboxWithHands = False)
            posture = None
            ratio_for_frame = None
            if lmList:
                ratio_for_frame = self.calc_ratio(lmList)
                avg_ratio = 0.0
                # calculating ratio for this run
                if ratio_for_frame is not None and ratio_for_frame > 0.0:
                    # smoothing buffer
                    self.ratio_list.append(ratio_for_frame)
                    if len(self.ratio_list) > self.smoothing_window:
                        self.ratio_list.pop(0)
                    avg_ratio = sum(self.ratio_list) / len(self.ratio_list) if self.ratio_list else 0.0
                    # calibration phase
                    if self.auto_calibrate and (not calibrated):
                        # only using reliable frames for calibration
                        if len(calib_samples) < self.calib_frames and ratio_for_frame > 0.0:
                            calib_samples.append(ratio_for_frame)
                            if len(calib_samples) >= self.calib_frames:
                                result = self.compute_threshold_from_samples(calib_samples)
                                if result is not None:
                                    self.posture_thresh = result
                                    calibrated = True
                        else:
                            calibrated = True
                else:
                    # unreliable current frame -> clear smoothing buffer to avoid stale values
                    self.ratio_list.clear()
                    posture = None
                # determining posture based on average ratio and threshold    
                if avg_ratio >= self.posture_thresh:
                    posture = "Straight"
                else:
                    posture = "Hunched"    
            else:
                # no landmarks detected -> clear smoothing buffer
                self.ratio_list.clear()
                posture = None
            # logging JSON data for each frame
            self.frame_logs["labels_per_frame"][str(frame_count)] = {
                "posture": posture
            }
            frame_count += 1
            self.frame_logs["total_frames"] = frame_count
        self.cap.release()
        cv2.destroyAllWindows()
