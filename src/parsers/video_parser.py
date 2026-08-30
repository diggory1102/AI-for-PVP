import os
import cv2
import tempfile
from faster_whisper import WhisperModel

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from video file using moviepy.
    """
    try:
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(video_path)
        if video.audio:
            video.audio.write_audiofile(output_audio_path, logger=None)
            video.close()
            return True
    except Exception as e:
        print(f"Moviepy audio extraction failed: {e}. Trying raw ffmpeg fallback...")
        # Fallback to direct ffmpeg system call
        import subprocess
        try:
            cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception as ex:
            print(f"ffmpeg fallback failed: {ex}")
    return False

def parse_video_audio(video_path, model_size="base"):
    """
    Extracts audio from video and transcribes it using faster-whisper.
    Returns: list of chunks with texts and timestamps.
    """
    file_name = os.path.basename(video_path)
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"{os.path.splitext(file_name)[0]}_temp.wav")
    
    transcripts = []
    
    if extract_audio_from_video(video_path, temp_audio_path):
        try:
            # Initialize local whisper model. Uses CPU and int8 encoding for portability.
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, info = model.transcribe(temp_audio_path, beam_size=5)
            
            for segment in segments:
                # Format timestamps as MM:SS
                start_min = int(segment.start // 60)
                start_sec = int(segment.start % 60)
                ts_str = f"{start_min:02d}:{start_sec:02d}"
                
                transcripts.append({
                    "text": segment.text.strip(),
                    "metadata": {
                        "source_type": "video_file",
                        "file_name": file_name,
                        "timestamp": ts_str
                    }
                })
        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
    return transcripts

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
