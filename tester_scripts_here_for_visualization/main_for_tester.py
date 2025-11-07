from tester_eye_state_detector import EyeBlinkDetector
from tester_posture_detector import PostureDetector

import json

# Path to your video
video_path = r"C:\Users\Stylebender07\OneDrive\Desktop\WAW_Project\priyavideo.mp4"

# Create detector instance
eye_state_detector = EyeBlinkDetector(video_path)

# Run detection
eye_state_detector.run()
posture_detector = PostureDetector(video_path)

# After run(), JSON log is available in eye_state_detector.frame_logs
print("Detection done")
print(f"Total frames processed: {eye_state_detector.frame_logs['total_frames']}")

# You can also log or modify JSON here:
custom_json_path = "eye_state_output.json"
with open(custom_json_path, "w") as f:
    json.dump(eye_state_detector.frame_logs, f, indent=2)

print(f"Results saved as: {custom_json_path}")

# Run detection
posture_detector.run()

# After run(), JSON log is available in posture_detector.frame_logs
print("Posture detection done")
print(f"Total frames processed: {posture_detector.frame_logs['total_frames']}")

# Save JSON output
custom_json_path = "posture_output.json"
with open(custom_json_path, "w") as f:
    json.dump(posture_detector.frame_logs, f, indent=2)

print(f"Posture results saved as: {custom_json_path}")

