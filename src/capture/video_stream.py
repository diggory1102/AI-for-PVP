import time
import threading
from PIL import Image
import mss
import mss.tools

class ScreenCaptureStream:
    def __init__(self, bbox=None, interval=1.0, callback=None):
        """
        bbox: dict with keys 'x', 'y', 'width', 'height'. If None, captures primary monitor.
        interval: seconds between captures.
        callback: function that accepts (PIL.Image, timestamp)
        """
        self.bbox = bbox
        self.interval = interval
        self.callback = callback
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

    def capture_single(self):
        """Captures a single frame and returns the PIL Image."""
        with mss.mss() as sct:
            if self.bbox:
                # mss expects top, left, width, height
                monitor = {
                    "top": int(self.bbox["y"]),
                    "left": int(self.bbox["x"]),
                    "width": int(self.bbox["width"]),
                    "height": int(self.bbox["height"])
                }
            else:
                monitor = sct.monitors[1]  # primary monitor
                
            sct_img = sct.grab(monitor)
            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return img

    def _run(self):
        while self.running:
            try:
                img = self.capture_single()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                if self.callback:
                    self.callback(img, timestamp)
            except Exception as e:
                print(f"Error during screen capture: {e}")
            time.sleep(self.interval)

if __name__ == "__main__":
    def print_callback(img, ts):
        print(f"[{ts}] Captured frame size: {img.size}")
        
    print("Starting capture stream of primary monitor for 5 seconds...")
    stream = ScreenCaptureStream(interval=1.0, callback=print_callback)
    stream.start()
    time.sleep(5)
    stream.stop()
    print("Capture stream stopped.")
