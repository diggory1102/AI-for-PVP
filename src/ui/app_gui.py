import os
import re
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTextEdit, QLineEdit, QLabel, QFileDialog, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor
from src.core.controller import background_click, background_type

class AgentWorker(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, agent, query, latest_frame, bbox, window_title, stream_class, retrieve_func, collection):
        super().__init__()
        self.agent = agent
        self.query = query
        self.latest_frame = latest_frame
        self.bbox = bbox
        self.window_title = window_title
        self.stream_class = stream_class
        self.retrieve_func = retrieve_func
        self.collection = collection
        
    def run(self):
        try:
            # 1. Grab frame in background if not already captured
            frame = self.latest_frame
            if not frame and (self.bbox or self.window_title):
                temp_stream = self.stream_class(bbox=self.bbox, window_title=self.window_title)
                frame = temp_stream.capture_single()
                
            # 2. Run RAG retrieval (local embedding computation) in background
            context = self.retrieve_func(self.collection, self.query, n_results=4)
            
            # 3. Generate agent response in background
            response = self.agent.generate_response(self.query, context, frame)
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Error generating response: {e}")

class IndexWorker(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, file_path, collection, parse_func, add_func):
        super().__init__()
        self.file_path = file_path
        self.collection = collection
        self.parse_func = parse_func
        self.add_func = add_func
        
    def run(self):
        try:
            docs = self.parse_func(self.file_path)
            if docs:
                self.add_func(self.collection, docs)
                self.finished.emit(f"Successfully indexed {len(docs)} chunks from {os.path.basename(self.file_path)}.")
            else:
                self.finished.emit("No text could be extracted from the file.")
        except Exception as e:
            self.finished.emit(f"Error indexing file: {e}")

class AssistantGUI(QMainWindow):
    def __init__(self, picker_func, stream_class, db_collection, agent_instance, add_docs_func, retrieve_func):
        super().__init__()
        self.picker_func = picker_func
        self.stream_class = stream_class
        self.collection = db_collection
        self.agent = agent_instance
        self.add_docs_func = add_docs_func
        self.retrieve_func = retrieve_func
        
        self.selected_bbox = None
        self.selected_window_title = None
        self.capture_stream = None
        self.latest_frame = None
        
        self.setWindowTitle("Windows AI Assistant")
        self.resize(800, 600)
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Header
        header = QLabel("Windows AI Assistant - RAG Dashboard")
        header.setFont(QFont("Outfit", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #FFFFFF; padding-bottom: 5px;")
        main_layout.addWidget(header)
        
        # Capture controls
        capture_layout = QHBoxLayout()
        self.btn_select_area = QPushButton("Select Screen Area")
        self.btn_select_area.clicked.connect(self.select_area)
        self.btn_select_area.setStyleSheet(self.button_style("#00AEFF"))
        capture_layout.addWidget(self.btn_select_area)
        
        self.chk_realtime = QCheckBox("Monitor Live Screen")
        self.chk_realtime.stateChanged.connect(self.toggle_live_monitoring)
        self.chk_realtime.setStyleSheet("color: #E0E0E0;")
        capture_layout.addWidget(self.chk_realtime)
        
        self.btn_clear_captures = QPushButton("Clear Captures")
        self.btn_clear_captures.clicked.connect(self.clear_captures)
        self.btn_clear_captures.setStyleSheet(self.button_style("#E74C3C"))
        capture_layout.addWidget(self.btn_clear_captures)
        
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color: #AAAAAA;")
        capture_layout.addWidget(self.lbl_status)
        capture_layout.addStretch()
        main_layout.addLayout(capture_layout)
        
        # Window selector layout
        window_layout = QHBoxLayout()
        self.lbl_window = QLabel("Or Active Window:")
        self.lbl_window.setStyleSheet("color: #E0E0E0;")
        window_layout.addWidget(self.lbl_window)
        
        self.combo_windows = QComboBox()
        self.combo_windows.setMinimumWidth(300)
        self.combo_windows.currentTextChanged.connect(self.on_window_selected)
        self.combo_windows.setStyleSheet("background-color: #1E1E24; color: #FFFFFF; border: 1px solid #2D2D35; border-radius: 4px; padding: 4px;")
        window_layout.addWidget(self.combo_windows)
        
        self.btn_refresh_windows = QPushButton("Refresh List")
        self.btn_refresh_windows.clicked.connect(self.refresh_window_list)
        self.btn_refresh_windows.setStyleSheet(self.button_style("#9B59B6"))
        window_layout.addWidget(self.btn_refresh_windows)
        window_layout.addStretch()
        main_layout.addLayout(window_layout)
        
        self.refresh_window_list()
        
        # Document management controls
        doc_layout = QHBoxLayout()
        self.btn_upload_doc = QPushButton("Upload Document (PDF/Docx/Txt)")
        self.btn_upload_doc.clicked.connect(self.upload_document)
        self.btn_upload_doc.setStyleSheet(self.button_style("#2ECC71"))
        doc_layout.addWidget(self.btn_upload_doc)
        
        self.lbl_db_info = QLabel("Indexed chunks: 0")
        self.lbl_db_info.setStyleSheet("color: #AAAAAA;")
        doc_layout.addWidget(self.lbl_db_info)
        doc_layout.addStretch()
        main_layout.addLayout(doc_layout)
        
        # Chat area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Inter", 10))
        self.chat_display.setStyleSheet("background-color: #1E1E24; border: 1px solid #2D2D35; border-radius: 6px; color: #E0E0E0; padding: 8px;")
        main_layout.addWidget(self.chat_display)
        
        # Input row
        input_layout = QHBoxLayout()
        self.input_query = QLineEdit()
        self.input_query.setPlaceholderText("Ask a question about your files or current screen area...")
        self.input_query.setFont(QFont("Inter", 10))
        self.input_query.returnPressed.connect(self.ask_question)
        self.input_query.setStyleSheet("background-color: #1E1E24; border: 1px solid #2D2D35; border-radius: 6px; color: #FFFFFF; padding: 8px;")
        input_layout.addWidget(self.input_query)
        
        self.btn_send = QPushButton("Ask")
        self.btn_send.clicked.connect(self.ask_question)
        self.btn_send.setStyleSheet(self.button_style("#3498DB"))
        self.btn_send.setFixedWidth(80)
        input_layout.addWidget(self.btn_send)
        main_layout.addLayout(input_layout)
        
        self.update_db_count()

    def select_area(self):
        self.lbl_status.setText("Selecting area...")
        # Minimize window temporarily to allow selection
        self.showMinimized()
        
        import time
        QThread.msleep(300) # Give UI time to minimize
        
        self.selected_bbox = self.picker_func()
        self.showNormal()
        
        if self.selected_bbox:
            self.lbl_status.setText(f"Area: {self.selected_bbox['width']}x{self.selected_bbox['height']}")
            # Restart stream if live monitoring is active
            if self.chk_realtime.isChecked():
                self.start_stream()
        else:
            self.lbl_status.setText("Selection cancelled.")

    def refresh_window_list(self):
        import pygetwindow as gw
        self.combo_windows.clear()
        self.combo_windows.addItem("--- Choose Active Window (Optional) ---")
        titles = sorted(list(set([w.title for w in gw.getAllWindows() if w.title.strip()])))
        for title in titles:
            self.combo_windows.addItem(title)

    def on_window_selected(self, title):
        if title == "--- Choose Active Window (Optional) ---" or not title:
            self.selected_window_title = None
        else:
            self.selected_window_title = title
            self.selected_bbox = None  # Clear manual selection if window selected
            self.lbl_status.setText(f"Locked on Window: {title[:30]}...")
            if self.chk_realtime.isChecked():
                self.start_stream()

    def start_stream(self):
        if self.capture_stream:
            self.capture_stream.stop()
        self.capture_stream = self.stream_class(
            bbox=self.selected_bbox,
            window_title=self.selected_window_title,
            interval=1.5,
            callback=self.on_frame_captured
        )
        self.capture_stream.start()
        self.lbl_status.setText(f"Monitoring live {'window' if self.selected_window_title else 'area'}...")

    def on_frame_captured(self, img, ts):
        self.latest_frame = img
        try:
            os.makedirs("data/captures", exist_ok=True)
            clean_ts = ts.replace(":", "-").replace(" ", "_")
            file_path = f"data/captures/capture_{clean_ts}.jpg"
            img.save(file_path, "JPEG")
        except Exception as e:
            print(f"Error saving live capture to disk: {e}")

    def clear_captures(self):
        folder = "data/captures"
        if os.path.exists(folder):
            import shutil
            try:
                count = 0
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        count += 1
                self.chat_display.append(f"<font color='#E74C3C'><b>System:</b> Deleted {count} local screen capture files.</font><br>")
            except Exception as e:
                self.chat_display.append(f"<font color='#E74C3C'><b>System:</b> Error deleting captures: {e}</font><br>")
        else:
            self.chat_display.append("<font color='#AAAAAA'><b>System:</b> No captures folder found to delete.</font><br>")

    def toggle_live_monitoring(self, state):
        if state == 2: # checked
            self.start_stream()
        else:
            if self.capture_stream:
                self.capture_stream.stop()
                self.capture_stream = None
            self.lbl_status.setText("Live monitoring stopped.")
            self.latest_frame = None

    def upload_document(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Document", "", "Documents (*.pdf *.docx *.txt *.py *.js *.cpp *.h)"
        )
        if file_path:
            self.lbl_db_info.setText("Parsing and indexing...")
            from src.parsers.doc_parser import parse_document
            self.worker = IndexWorker(file_path, self.collection, parse_document, self.add_docs_func)
            self.worker.finished.connect(self.on_indexing_finished)
            self.worker.start()

    def on_indexing_finished(self, msg):
        self.lbl_db_info.setText(msg)
        self.update_db_count()

    def update_db_count(self):
        try:
            count = self.collection.count()
            self.lbl_db_info.setText(f"Indexed chunks in DB: {count}")
        except Exception:
            pass

    def ask_question(self):
        query = self.input_query.text().strip()
        if not query:
            return
            
        self.chat_display.append(f"<b>You:</b> {query}")
        self.input_query.clear()
        self.chat_display.append("<i>Thinking...</i>")
        
        # Disable input controls to prevent duplicate questions during generation
        self.input_query.setEnabled(False)
        self.btn_send.setEnabled(False)
        
        # Start background worker to run all heavy operations (RAG embedding search, frame capture, Ollama)
        self.agent_worker = AgentWorker(
            agent=self.agent,
            query=query,
            latest_frame=self.latest_frame,
            bbox=self.selected_bbox,
            window_title=self.selected_window_title,
            stream_class=self.stream_class,
            retrieve_func=self.retrieve_func,
            collection=self.collection
        )
        self.agent_worker.finished.connect(self.on_agent_response)
        self.agent_worker.start()

    def on_agent_response(self, response):
        # Remove "Thinking..."
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        
        self.chat_display.append(f"<b>Assistant:</b> {response}<br>")
        
        # Parse for action commands: [CLICK: x, y] and [TYPE: text]
        click_matches = re.findall(r"\[CLICK:\s*(\d+),\s*(\d+)\]", response)
        type_matches = re.findall(r"\[TYPE:\s*([^\]]+)\]", response)
        
        if click_matches or type_matches:
            if not self.selected_window_title:
                self.chat_display.append("<font color='#E74C3C'><b>[System Warning]:</b> AI attempted to execute action commands but no Active Window is selected!</font><br>")
            else:
                # Execute clicks
                for x, y in click_matches:
                    success = background_click(self.selected_window_title, int(x), int(y))
                    if success:
                        self.chat_display.append(f"<font color='#2ECC71'><b>[System Action]:</b> Clicked at ({x}, {y}) inside '{self.selected_window_title[:20]}...'</font><br>")
                    else:
                        self.chat_display.append(f"<font color='#E74C3C'><b>[System Error]:</b> Failed to click at ({x}, {y}) inside '{self.selected_window_title[:20]}...'</font><br>")
                
                # Execute typing
                for text in type_matches:
                    success = background_type(self.selected_window_title, text)
                    if success:
                        self.chat_display.append(f"<font color='#2ECC71'><b>[System Action]:</b> Typed '{text}' inside '{self.selected_window_title[:20]}...'</font><br>")
                    else:
                        self.chat_display.append(f"<font color='#E74C3C'><b>[System Error]:</b> Failed to type '{text}' inside '{self.selected_window_title[:20]}...'</font><br>")

        # Re-enable inputs
        self.input_query.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.input_query.setFocus()

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#121214"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#1E1E24"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#121214"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#1E1E24"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#FF0000"))
        palette.setColor(QPalette.ColorRole.Link, QColor("#00AEFF"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#00AEFF"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
        self.setPalette(palette)

    def button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
        """

    def closeEvent(self, event):
        if self.capture_stream:
            self.capture_stream.stop()
        event.accept()
