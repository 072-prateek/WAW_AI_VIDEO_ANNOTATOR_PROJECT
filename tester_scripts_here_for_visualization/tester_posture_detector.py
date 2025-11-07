import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import math
import json
import os

class PostureDetector:
    def __init__(self, video_path=None):
        # load video file or webcam
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.video_filename = os.path.basename(video_path)
            print(f"Loaded video file: {self.video_filename}")
        else:
            self.cap = cv2.VideoCapture(0)
            self.video_filename = "webcam_feed"

        # pose detector
        self.detector = PoseDetector(staticMode=False,
                                     modelComplexity=1,
                                     smoothLandmarks=True,
                                     enableSegmentation=False,
                                     detectionCon=0.5,
                                     trackCon=0.5)

        # live plot for ratio values
        self.plotRatio = LivePlot(640, 360, [0, 100], invert=True)

        # ratio smoothing buffer
        self.ratio_list = []

        # thresholds for posture detection (overwritten after calibration)
        # unit: normalized percent (dy/shoulder_width * 100)
        self.posture_thresh = 18.0

        # json data structure
        self.frame_logs = {
            "video_filename": self.video_filename,
            "total_frames": 0,
            "labels_per_frame": {}
        }

        # calibration parameters
        self.auto_calibrate = True
        self.calib_frames = 150
        self.p_low = 20
        self.p_high = 80

        # additional settings for robustness
        self.min_shoulder_width_px = 10        # less than this => too small/invalid
        self.max_head_turn_ratio = 0.55        # nose_x offset / shoulder_width > this => turned -> unreliable
        self.smoothing_window = 5              # moving average window size used externally (you already had 4)
        self.reliable_count_for_calib = 5      # minimum reliable frames to attempt calibration

    def _safe_get_landmark(self, lmList, idx):
        try:
            p = lmList[idx]
            if len(p) >= 2:
                return float(p[0]), float(p[1])
        except Exception:
            pass
        return None

    def calc_ratio(self, lmList):

        # extract key landmarks manually
        if len(lmList) <= 12:
            return None  # insufficient landmarks detected

        nose_x, nose_y = lmList[0][0:2]
        left_shoulder_x, left_shoulder_y = lmList[11][0:2]
        right_shoulder_x, right_shoulder_y = lmList[12][0:2]

        # calculate shoulder midpoint (average of both shoulders)
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2.0
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2.0

        # compute shoulder width using both x and y (real 2D distance)
        shoulder_dx = right_shoulder_x - left_shoulder_x
        shoulder_dy = right_shoulder_y - left_shoulder_y
        shoulder_width = math.hypot(shoulder_dx, shoulder_dy)

        # check reliability — too small width = subject too far or poor detection
        if shoulder_width < self.min_shoulder_width_px:
            return None

        # find vertical distance between nose and shoulder midpoint
        vertical_offset_y = shoulder_mid_y - nose_y  # positive if nose is above (since y increases downward)
        normalized_vertical = vertical_offset_y / shoulder_width  # normalize by body scale

        # find horizontal distance to detect head turn (helps ignore side poses)
        horizontal_offset_x = nose_x - shoulder_mid_x
        head_turn_ratio = abs(horizontal_offset_x) / shoulder_width

        # reject frames where head is too turned sideways
        if head_turn_ratio > self.max_head_turn_ratio:
            return None

        # convert normalized vertical ratio to a percentage-like score
        posture_score = max(0.0, normalized_vertical * 100.0)

        # cap extreme outliers
        if posture_score > 300:
            posture_score = 300.0

        # higher posture_score → straighter posture
        return posture_score

    def compute_threshold_from_samples(self, samples, p_low=None, p_high=None):
        if p_low is None: p_low = self.p_low
        if p_high is None: p_high = self.p_high
        if len(samples) < self.reliable_count_for_calib:
            return None
        low_p = np.percentile(samples, p_low)
        high_p = np.percentile(samples, p_high)
        threshold = (low_p + high_p) / 2.0
        return threshold

    def show_posture_state(self, img, avg_ratio):
        # display text on frame
        if avg_ratio is None:
            cvzone.putTextRect(img, "Posture: Unknown", (25, 400),
                               scale=2, thickness=2, offset=10, colorR=(0, 0, 200))
            return None

        if avg_ratio > self.posture_thresh:
            posture = "Straight"
            color = (0, 255, 0)
        else:
            posture = "Hunched"
            color = (0, 0, 255)
        cvzone.putTextRect(img, f"Posture: {posture}", (25, 400),
                           scale=2, thickness=2, offset=10, colorR=color)
        return posture

    def run(self):
        frame_count = 0

        calibrated = False
        calib_samples = []

        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to read frame")
                break

            img = self.detector.findPose(img, draw=True)
            lmList, _ = self.detector.findPosition(img, draw=False, bboxWithHands=False)

            posture = None
            ratio_for_frame = None

            if lmList:
                ratio_for_frame = self.calc_ratio(lmList)

                # show visual cues if ratio available
                if ratio_for_frame is not None:
                    # smoothing buffer
                    self.ratio_list.append(ratio_for_frame)
                    if len(self.ratio_list) > self.smoothing_window:
                        self.ratio_list.pop(0)
                    avg_ratio = sum(self.ratio_list) / len(self.ratio_list)

                    # calibration phase
                    if self.auto_calibrate and (not calibrated):
                        # only use reliable frames for calibration
                        if len(calib_samples) < self.calib_frames:
                            calib_samples.append(ratio_for_frame)
                            cvzone.putTextRect(img, f"Calibrating... ({len(calib_samples)}/{self.calib_frames})",
                                               (25, 50), scale=1, thickness=2, offset=6, colorR=(0,150,255))
                            if len(calib_samples) >= self.calib_frames:
                                result = self.compute_threshold_from_samples(calib_samples)
                                if result is not None:
                                    self.posture_thresh = result
                                    calibrated = True
                                    print("Calibration complete - threshold:", self.posture_thresh)
                        else:
                            calibrated = True

                    posture = self.show_posture_state(img, avg_ratio)

                    # draw diagnostic visuals: nose & shoulder mid & line
                    nose = self._safe_get_landmark(lmList, 0)
                    left_sh = self._safe_get_landmark(lmList, 11)
                    right_sh = self._safe_get_landmark(lmList, 12)
                    if nose and left_sh and right_sh:
                        nx, ny = nose
                        lx, ly = left_sh
                        rx, ry = right_sh
                        mid_x = int((lx + rx) / 2)
                        mid_y = int((ly + ry) / 2)
                        cv2.circle(img, (int(nx), int(ny)), 6, (255, 0, 0), -1)
                        cv2.circle(img, (mid_x, mid_y), 6, (0, 255, 0), -1)
                        cv2.line(img, (int(nx), int(ny)), (mid_x, mid_y), (200, 200, 0), 2)
                        cvzone.putTextRect(img, f"Score:{avg_ratio:.1f}", (25, 100),
                                           scale=1, thickness=2, offset=6, colorR=(255,255,0))

                    # live plot update
                    ratio_plot = self.plotRatio.update(avg_ratio)
                    cv2.imshow("Posture Ratio", ratio_plot)

                else:
                    # unreliable current frame -> clear smoothing buffer to avoid stale values
                    self.ratio_list.clear()
                    cvzone.putTextRect(img, "Unreliable view (turned/missing)", (25, 50),
                                       scale=1, thickness=2, offset=6, colorR=(0,0,255))
                    posture = None

            else:
                cvzone.putTextRect(img, "No Body Detected", (25, 400),
                                   scale=2, thickness=2, offset=10, colorR=(0, 0, 255))
                self.ratio_list.clear()
                posture = None

            # store label for this frame
            self.frame_logs["labels_per_frame"][str(frame_count)] = {
                "posture": posture
            }
            frame_count += 1
            self.frame_logs["total_frames"] = frame_count

            cv2.imshow("Posture Detection", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
