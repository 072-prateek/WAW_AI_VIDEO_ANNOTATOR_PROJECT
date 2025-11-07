from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os, shutil, json, asyncio
from eye_state_detector import EyeBlinkDetector
from posture_detector import PostureDetector
from f1_score_calculator import F1ScoreEvaluator  

# Initializing FastApi App Object
app = FastAPI() 

@app.post("/annotate")
async def analyze_video(
    video: UploadFile = File(...), # mandatory video file
    groundtruth: UploadFile = File(None)  # optional GroundTruth json  file
):
    try:
        # Ensuring directory exists
        save_dir = "uploaded_videos"
        os.makedirs(save_dir, exist_ok=True)

        # Validating file type .mp4 or .avi
        if not (video.filename.endswith(".mp4") or video.filename.endswith(".avi")):
            return JSONResponse(
                content={"error": "Invalid file format. Only .mp4 and .avi are supported."},
                status_code=400
            )

        # Saving uploaded video locally
        save_path = os.path.join(save_dir, video.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Saving ground truth JSON locally if provided
        groundtruth_path = None
        if groundtruth:
            groundtruth_path = os.path.join(save_dir, groundtruth.filename)
            with open(groundtruth_path, "wb") as buffer:
                shutil.copyfileobj(groundtruth.file, buffer)

        # Initializing detector objects
        eye_state_detector = EyeBlinkDetector(save_path)
        posture_detector = PostureDetector(save_path)
        # Defining async functions to run both detectors concurrently to optimize processing time
        async def run_eye_detector():
            try:
                await asyncio.to_thread(eye_state_detector.run)
                with open("eye_state_output.json", "w") as f:
                    json.dump(eye_state_detector.frame_logs, f, indent=2)
            except Exception as e:
                raise RuntimeError(f"Eye detector failed: {str(e)}")

        async def run_posture_detector():
            try:
                await asyncio.to_thread(posture_detector.run)
                with open("posture_output.json", "w") as f:
                    json.dump(posture_detector.frame_logs, f, indent=2)
            except Exception as e:
                raise RuntimeError(f"Posture detector failed: {str(e)}")
        # Defining async function to merge both JSON outputs that we get from both the detectors and deleting the individual files after merging
        async def merge_json_files():
            try:
                with open("eye_state_output.json", "r") as f1:
                    eye_data = json.load(f1)
                with open("posture_output.json", "r") as f2:
                    posture_data = json.load(f2)

                total_frames = min(eye_data["total_frames"], posture_data["total_frames"])
                merged = {
                    "video_filename": eye_data["video_filename"],
                    "total_frames": total_frames,
                    "labels_per_frame": {}
                }

                for i in range(total_frames):
                    frame_key = str(i)
                    merged["labels_per_frame"][frame_key] = {
                        "eye_state": eye_data["labels_per_frame"][frame_key]["eye_state"],
                        "posture": posture_data["labels_per_frame"][frame_key]["posture"]
                    }

                output_path = os.path.join(
                    save_dir, f"{os.path.splitext(video.filename)[0]}_output.json"
                )
                with open(output_path, "w") as f:
                    json.dump(merged, f, indent=2)

                try:
                    os.remove("eye_state_output.json")
                    os.remove("posture_output.json")
                except:
                    pass

                return merged, output_path
            except Exception as e:
                raise RuntimeError(f"Failed to merge JSON files: {str(e)}")

        # Running both detectors concurrently
        await asyncio.gather(run_eye_detector(), run_posture_detector())

        # Merged JSON output from both detectors and its output path for f1 evaluation
        merged_result, output_path = await merge_json_files()

        #  calculating F1 scores if ground truth is provided
        if groundtruth_path:
            try:
                evaluator = F1ScoreEvaluator(output_path, groundtruth_path)
                f1_scores = evaluator.run()
                print(f"Eye State F1 Score:   {f1_scores['eye_state']:.4f}")
                print(f"Posture F1 Score:     {f1_scores['posture']:.4f}")
            except Exception as e:
                print(f"F1 Evaluation failed: {e}")
        else:
            print("No ground truth JSON provided. Skipping F1 evaluation.")

        # Returning  JSON annotation output
        return JSONResponse(content=merged_result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
