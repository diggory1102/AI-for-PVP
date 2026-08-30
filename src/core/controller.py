import time
import win32gui
import win32con

# Safety Guardrail: List of semantic keywords prohibited from clicking
BANNED_KEYWORDS = ["xóa", "hủy", "release", "delete", "shop", "cửa hàng", "nạp tiền", "nạp thẻ", "settings", "cài đặt"]

def get_window_handle(window_title):
    """Resolves target HWND using exact or partial title matching."""
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd or hwnd == 0:
        # Partial match fallback
        def callback(h, extra):
            if win32gui.IsWindowVisible(h) and window_title.lower() in win32gui.GetWindowText(h).lower():
                extra.append(h)
            return True
        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        if hwnds:
            hwnd = hwnds[0]
        else:
            return None
    return hwnd

def background_click(window_title, x, y, label=None):
    """
    Sends WM_LBUTTONDOWN and WM_LBUTTONUP messages to the target window
    to simulate a click at relative coordinates (x, y) without moving the real mouse cursor.
    Validates the text label against safety blacklists before executing.
    """
    if label:
        label_lower = label.lower()
        for kw in BANNED_KEYWORDS:
            if kw in label_lower:
                print(f"Controller Safety Check: Blocked click at ({x}, {y}) because label '{label}' contains banned keyword '{kw}'.")
                return False

    hwnd = get_window_handle(window_title)
    if not hwnd:
        print(f"Controller Error: Window '{window_title}' not found.")
        return False
        
    # Coordinate encoding: y in high 16 bits, x in low 16 bits
    lParam = (int(y) << 16) | int(x)
    
    try:
        # Send Left Button Down
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
        time.sleep(0.05) # Simulate physical click duration
        # Send Left Button Up
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
        print(f"Controller: Sent background click to '{window_title}' at ({x}, {y}) [Label: {label}]")
        return True
    except Exception as e:
        print(f"Controller Error: Failed to click in background: {e}")
        return False

def background_type(window_title, text):
    """
    Sends WM_CHAR messages to the target window message loop
    to simulate typing characters in a background input field.
    """
    hwnd = get_window_handle(window_title)
    if not hwnd:
        print(f"Controller Error: Window '{window_title}' not found.")
        return False
        
    try:
        for char in text:
            # Send character directly to target window's message queue
            win32gui.PostMessage(hwnd, win32con.WM_CHAR, ord(char), 0)
            time.sleep(0.02)
        print(f"Controller: Sent background text to '{window_title}': '{text}'")
        return True
    except Exception as e:
        print(f"Controller Error: Failed to type in background: {e}")
        return False

if __name__ == "__main__":
    # Standard dummy import check
    print("Action Controller loaded successfully.")
