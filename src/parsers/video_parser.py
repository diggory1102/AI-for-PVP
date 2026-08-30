import os
import cv2

def extract_keyframes(video_path, output_dir=None, threshold=0.3, sample_rate_sec=2):
    """
    Extracts keyframes from a video file based on temporal sampling.
    """
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        
    frame_interval = int(fps * sample_rate_sec)
    
    keyframes = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            
            # Record metadata about frame
            keyframes.append({
                "frame": frame.copy(),
                "timestamp": timestamp_str,
                "seconds": timestamp_sec,
                "video_name": video_name
            })
            
        frame_count += 1
        
    cap.release()
    return keyframes

def parse_video_metadata(video_path):
    """
    Basic video properties extractor.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "file_name": os.path.basename(video_path),
        "duration_seconds": duration,
        "resolution": f"{width}x{height}",
        "fps": fps
    }

if __name__ == "__main__":
    # Standard dummy check
    print("Video parser imported successfully.")
