import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import json
import os

class EyeBlinkDetector:
    def __init__(self, video_path=None):
        # load video file or webcam
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.video_filename = os.path.basename(video_path)
        else:
            self.cap = cv2.VideoCapture(0)
            self.video_filename = "webcam_feed"

        # face mesh detector
        self.detector = FaceMeshDetector(maxFaces=1)

        # live plot for ear values
        self.plotY = LivePlot(640, 360, [0, 50], invert=True)

        # landmark ids for both eyes
        self.left_eye_ids = [362, 263, 386, 374, 387, 373]
        self.right_eye_ids = [33, 133, 159, 145, 160, 144]
        self.all_ids = self.left_eye_ids + self.right_eye_ids

        # ear history lists for smoothing
        self.left_ear_list = []
        self.right_ear_list = []

        # thresholds for eye closed detection (will be overwritten by calibration)
        self.left_thresh = 19
        self.right_thresh = 20

        # json data structure
        self.frame_logs = {
            "video_filename": self.video_filename,
            "total_frames": 0,
            "labels_per_frame": {}
        }

        #  calibration parameters
        self.auto_calibrate = True            # set False if you want to disable auto-calibration
        self.calib_frames = 150               # number of valid frame samples to collect for calibration
        self.p_low = 20                       # low percentile (closed-ish)
        self.p_high = 80                      # high percentile (open-ish)

    def calc_EAR(self, face):
        # calculate eye aspect ratio for both eyes
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
        leftver2, _ = self.detector.findDistance(leftleftup, leftleftdown)
        leftVer = (leftVer1 + leftver2) / 2
        leftHor, _ = self.detector.findDistance(leftLeft, leftRight)
        rightVer1, _ = self.detector.findDistance(rightUp, rightDown)
        rightver2, _ = self.detector.findDistance(rightleftup, rightleftdown)
        rightver = (rightVer1 + rightver2) / 2
        rightHor, _ = self.detector.findDistance(rightLeft, rightRight)

        leftEAR = (leftVer / leftHor) * 100 if leftHor > 0 else 0 # so we don't divide by zero
        rightEAR = (rightver / rightHor) * 100 if rightHor > 0 else 0

        return leftEAR, rightEAR, [leftUp, leftDown, leftLeft, leftRight, rightUp, rightDown, rightLeft, rightRight, leftleftup, leftleftdown, rightleftup, rightleftdown]

    def draw_eye_lines(self, img, pts):
       # draw green lines connecting eye landmarks
        leftUp, leftDown, leftLeft, leftRight, rightUp, rightDown, rightLeft, rightRight, leftleftup, leftleftdown, rightleftup, rightleftdown = pts
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 2)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 2)
        cv2.line(img, rightUp, rightDown, (0, 200, 0), 2)
        cv2.line(img, rightLeft, rightRight, (0, 200, 0), 2)
        cv2.line(img, leftleftup, leftleftdown, (0, 200, 0), 2)
        cv2.line(img, rightleftup, rightleftdown, (0, 200, 0), 2)

    def show_eye_state(self, img, avg_left_ear, avg_right_ear):
        # put text for left and right eye states
        if avg_left_ear < self.left_thresh:
            cvzone.putTextRect(img, "Left Eye Closed", (25, 400), scale=2, thickness=2, offset=10)
        else:
            cvzone.putTextRect(img, "Left Eye Open", (25, 400), scale=2, thickness=2, offset=10)

        if avg_right_ear < self.right_thresh:
            cvzone.putTextRect(img, "Right Eye Closed", (25, 450), scale=2, thickness=2, offset=10)
        else:
            cvzone.putTextRect(img, "Right Eye Open", (25, 450), scale=2, thickness=2, offset=10)

    def compute_thresholds_from_samples(self, left_samples, right_samples, p_low=None, p_high=None):
        # percentile-based: closed -> low percentile, open -> high percentile
        if p_low is None: p_low = self.p_low
        if p_high is None: p_high = self.p_high
        if len(left_samples) < 5 or len(right_samples) < 5:
            return None
        l_closed = np.percentile(left_samples, p_low)
        l_open = np.percentile(left_samples, p_high)
        r_closed = np.percentile(right_samples, p_low)
        r_open = np.percentile(right_samples, p_high)
        left_thresh = (l_closed + l_open) / 2
        right_thresh = (r_closed + r_open) / 2
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

            img, faces = self.detector.findFaceMesh(img, draw=False)

            if faces:
                face = faces[0]

                # draw small circles on eye landmarks
                for id in self.all_ids:
                    cv2.circle(img, face[id], 3, (255, 0, 255), cv2.FILLED)

                # calculate EARs
                leftEAR, rightEAR, points = self.calc_EAR(face)

                # smoothing values using small buffer
                # length can be anything between 3 - 5 for smoothing gotta find the sweet spot
                self.left_ear_list.append(leftEAR)
                self.right_ear_list.append(rightEAR)
                if len(self.left_ear_list) > 4:
                    self.left_ear_list.pop(0)
                    self.right_ear_list.pop(0)

                avg_left_ear = sum(self.left_ear_list) / len(self.left_ear_list)
                avg_right_ear = sum(self.right_ear_list) / len(self.right_ear_list)

                # draw connecting lines and show text
                self.draw_eye_lines(img, points)

                # per-run auto calibration logic
                if self.auto_calibrate and (not calibrated):
                    # collect only valid face samples (sample-based)
                    if len(calib_left_samples) < self.calib_frames:
                        calib_left_samples.append(leftEAR)
                        calib_right_samples.append(rightEAR)
                        cvzone.putTextRect(img, f"Calibrating... ({len(calib_left_samples)}/{self.calib_frames})", (25, 50), scale=1, thickness=2, offset=6, colorR=(0,150,255))
                        # once we've collected enough valid samples compute thresholds
                        if len(calib_left_samples) >= self.calib_frames:
                            result = self.compute_thresholds_from_samples(calib_left_samples, calib_right_samples)
                            if result:
                                self.left_thresh, self.right_thresh = result
                                calibrated = True
                    else:
                        # safety fallback if something odd happens
                        calibrated = True

                # use calibrated or default thresholds to show state
                self.show_eye_state(img, avg_left_ear, avg_right_ear)

                # determine state
                if avg_left_ear < self.left_thresh and avg_right_ear < self.right_thresh:
                    eye_state = "Closed"
                else:
                    eye_state = "Open"

                # plot graphs for both eyes
                left_plot = self.plotY.update(avg_left_ear)
                right_plot = self.plotY.update(avg_right_ear)
                cv2.imshow("Left Eye EAR", left_plot)
                cv2.imshow("Right Eye EAR", right_plot)
            else:
                # when face not detected in frame
                cvzone.putTextRect(img, "No Face Detected", (25, 400),
                                   scale=2, thickness=2, offset=10, colorR=(0, 0, 255))
                self.left_ear_list.clear()
                self.right_ear_list.clear()
                eye_state = None

            # log JSON data for each frame
            self.frame_logs["labels_per_frame"][str(frame_count)] = {
                "eye_state": eye_state,
            }
            frame_count += 1
            self.frame_logs["total_frames"] = frame_count

            cv2.imshow("Image", img)

            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
