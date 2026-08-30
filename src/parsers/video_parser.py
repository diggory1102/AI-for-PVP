import os
import cv2
import tempfile
import whisper
from PIL import Image

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
    Extracts audio from video and transcribes it using openai-whisper.
    Returns: list of chunks with texts and timestamps.
    """
    file_name = os.path.basename(video_path)
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"{os.path.splitext(file_name)[0]}_temp.wav")
    
    transcripts = []
    
    if extract_audio_from_video(video_path, temp_audio_path):
        try:
            # Initialize local whisper model.
            model = whisper.load_model(model_size, device="cpu")
            result = model.transcribe(temp_audio_path)
            
            for segment in result.get("segments", []):
                # Format timestamps as MM:SS
                start_min = int(segment["start"] // 60)
                start_sec = int(segment["start"] % 60)
                ts_str = f"{start_min:02d}:{start_sec:02d}"
                
                transcripts.append({
                    "text": segment["text"].strip(),
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

def parse_video_multimodal(video_path, vision_model=None, whisper_model_size="base", sample_rate_sec=5):
    """
    Extracts both audio (Whisper speech-to-text) and visual content (Qwen-VL keyframe descriptions)
    from a video, returning a unified list of text chunks with metadata for RAG indexing.
    """
    file_name = os.path.basename(video_path)
    combined_chunks = []

    # 1. AUDIO PARSING (Whisper)
    print(f" -> [Audio Ingestion] Running Whisper speech-to-text on '{file_name}'...")
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"{os.path.splitext(file_name)[0]}_temp.wav")
    
    if extract_audio_from_video(video_path, temp_audio_path):
        try:
            model = whisper.load_model(whisper_model_size, device="cpu")
            result = model.transcribe(temp_audio_path)
            
            for segment in result.get("segments", []):
                start_min = int(segment["start"] // 60)
                start_sec = int(segment["start"] % 60)
                ts_str = f"{start_min:02d}:{start_sec:02d}"
                
                combined_chunks.append({
                    "text": f"Video Audio commentary at [{ts_str}]: {segment['text'].strip()}",
                    "metadata": {
                        "source_type": "video_audio",
                        "file_name": file_name,
                        "timestamp": ts_str
                    }
                })
        except Exception as e:
            print(f"    Whisper Transcription failed or skipped: {e}")
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    # 2. VISUAL PARSING (Qwen-VL Keyframe Descriptions)
    if vision_model:
        print(f" -> [Visual Ingestion] Extracting keyframes and generating VLM descriptions...")
        keyframes = extract_keyframes(video_path, sample_rate_sec=sample_rate_sec)
        print(f"    Extracted {len(keyframes)} keyframes to describe.")
        
        for kf in keyframes:
            ts_str = kf["timestamp"]
            # Convert CV2 BGR frame to PIL RGB Image
            frame_rgb = cv2.cvtColor(kf["frame"], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Request description from Vision LLM
            prompt = (
                "Describe the visual state of this game screen at this instant in the video clip. "
                "List visible characters/monsters, active menus, action buttons, draft/pick slots, and health/speed bars. "
                "Be detailed and concise."
            )
            try:
                description = vision_model.analyze_image(pil_img, prompt)
                if description and "Error" not in description:
                    print(f"    [Visual {ts_str}]: {description[:80]}...")
                    combined_chunks.append({
                        "text": f"Video Visual gameplay state at [{ts_str}]: {description.strip()}",
                        "metadata": {
                            "source_type": "video_visual",
                            "file_name": file_name,
                            "timestamp": ts_str
                        }
                    })
            except Exception as e:
                print(f"    VLM failed to analyze keyframe at {ts_str}: {e}")
                
    return combined_chunks
