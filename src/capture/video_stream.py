import time
import threading
from PIL import Image
import mss
import mss.tools
import pygetwindow as gw

# Import pywin32 modules for GDI window capture
import win32gui
import win32ui
import win32con

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

    def _capture_gdi(self, title):
        """
        Attempts to capture only the target window's drawing buffer using Windows GDI.
        This ignores overlapping windows.
        """
        try:
            # Find window handle
            hwnd = win32gui.FindWindow(None, title)
            if not hwnd or hwnd == 0:
                # Try partial match fallback
                def callback(h, extra):
                    if win32gui.IsWindowVisible(h) and title.lower() in win32gui.GetWindowText(h).lower():
                        extra.append(h)
                    return True
                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                if hwnds:
                    hwnd = hwnds[0]
                else:
                    return None

            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            w = right - left
            h = bottom - top

            if w <= 0 or h <= 0:
                return None

            # Setup DC and Bitmap
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
            saveDC.SelectObject(saveBitMap)

            # Use PrintWindow with PW_RENDERFULLCONTENT (3) flag to grab window buffer
            result = win32gui.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
            
            # Fallback to BitBlt if PrintWindow fails (BitBlt grabs screen pixels, so overlapping window shows)
            if not result:
                saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

            # Export bits to PIL
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )

            # Cleanup GDI Objects to prevent memory leak
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

            return img
        except Exception as e:
            print(f"GDI window capture failed for '{title}': {e}")
            return None

    def capture_single(self):
        """Captures a single frame and returns the PIL Image."""
        # 1. Try GDI buffer capture first if window_title is set
        if self.window_title:
            gdi_img = self._capture_gdi(self.window_title)
            if gdi_img:
                return gdi_img
        
        # 2. Fallback to physical screen-space capture using mss
        target_bbox = self.bbox
        if self.window_title:
            try:
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
