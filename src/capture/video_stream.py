import time
import threading
from PIL import Image
import mss
import mss.tools
import pygetwindow as gw

class ScreenCaptureStream:
    def __init__(self, bbox=None, window_title=None, interval=1.0, callback=None):
        """
        bbox: dict with keys 'x', 'y', 'width', 'height'.
        window_title: string representing the title of target window. Takes priority over bbox if valid.
        interval: seconds between captures.
        callback: function that accepts (PIL.Image, timestamp)
        """
        self.bbox = bbox
        self.window_title = window_title
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
        target_bbox = self.bbox
        
        if self.window_title:
            try:
                # Try to locate window by title
                windows = gw.getWindowsWithTitle(self.window_title)
                if windows:
                    win = windows[0]
                    if not win.isMinimized and win.width > 10 and win.height > 10:
                        target_bbox = {
                            "x": win.left,
                            "y": win.top,
                            "width": win.width,
                            "height": win.height
                        }
            except Exception as e:
                print(f"Error fetching window coordinates for '{self.window_title}': {e}")
                
        with mss.mss() as sct:
            if target_bbox:
                # Ensure width/height are positive and non-zero
                w = max(10, int(target_bbox["width"]))
                h = max(10, int(target_bbox["height"]))
                monitor = {
                    "top": int(target_bbox["y"]),
                    "left": int(target_bbox["x"]),
                    "width": w,
                    "height": h
                }
            else:
                monitor = sct.monitors[1]  # primary monitor
                
            try:
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                return img
            except Exception as e:
                print(f"mss grab failed: {e}")
                # Return small dummy image to prevent crash
                return Image.new("RGB", (100, 100), color="black")

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
