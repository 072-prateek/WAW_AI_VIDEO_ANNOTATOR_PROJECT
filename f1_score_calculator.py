import json

class F1ScoreEvaluator:

    def __init__(self, prediction_path, groundtruth_path):

        # Initializing paths and loading JSON data
        self.prediction_path = prediction_path
        self.groundtruth_path = groundtruth_path
        self.prediction_data = self._load_json(prediction_path)
        self.groundtruth_data = self._load_json(groundtruth_path)

    def _load_json(self, path):

        # Loading JSON data from a file
        with open(path, "r") as f:
            return json.load(f)

    def _compute_f1(self, label_key, positive_label, negative_label):

        # Initializing true positive, false positive, false negative, true negative counters
        tp = fp = fn = tn = 0
        total_frames = min(
            self.prediction_data["total_frames"],
            self.groundtruth_data["total_frames"]
        )
        # Iterating through each frame to compute true positives, false positives, false negatives, true negatives
        for i in range(total_frames):
            frame_key = str(i)
            if (
                frame_key not in self.prediction_data["labels_per_frame"]
                or frame_key not in self.groundtruth_data["labels_per_frame"]
            ):
                continue

            pred_value = self.prediction_data["labels_per_frame"][frame_key][label_key]
            gt_value = self.groundtruth_data["labels_per_frame"][frame_key][label_key]

            if pred_value == positive_label and gt_value == positive_label:
                tp += 1
            elif pred_value == positive_label and gt_value == negative_label:
                fp += 1
            elif pred_value == negative_label and gt_value == positive_label:
                fn += 1
            else:
                tn += 1
        # Calculating precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        return  f1
    
    def run(self):

        # Running F1 score computation for eye_state and posture
        results = {}
        results["eye_state"] = self._compute_f1("eye_state", "Open", "Closed")
        results["posture"] = self._compute_f1("posture", "Straight", "Hunched")
        return results

