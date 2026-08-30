import sys
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRect, QPoint

class ScreenPickerOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setWindowOpacity(0.3)
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        # Make full screen across all monitors
        screen = QApplication.primaryScreen()
        geom = screen.geometry()
        self.setGeometry(geom)
        
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_selecting = False
        self.selected_rect = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.end_point = event.pos()
            self.is_selecting = False
            
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)
            
            if w > 5 and h > 5:
                self.selected_rect = {"x": x, "y": y, "width": w, "height": h}
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.selected_rect = None
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        if self.is_selecting:
            pen = QPen(QColor(0, 174, 255), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 174, 255, 50))
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)

def pick_screen_area():
    """
    Opens a full-screen semi-transparent overlay to let the user select a region.
    Returns: dict with 'x', 'y', 'width', 'height' or None.
    """
    app = QApplication.instance()
    created_app = False
    if not app:
        app = QApplication(sys.argv)
        created_app = True
        
    picker = ScreenPickerOverlay()
    picker.show()
    
    if created_app:
        app.exec()
    else:
        # If run inside another Qt app loop
        picker.setWindowModality(Qt.WindowModality.ApplicationModal)
        # Block until closed
        import time
        while picker.isVisible():
            app.processEvents()
            time.sleep(0.01)
            
    return picker.selected_rect

if __name__ == "__main__":
    print("Drag to select an area. Press ESC to cancel.")
    area = pick_screen_area()
    print("Selected area:", area)
