from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                               QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QLineEdit, QSizePolicy,
                               QFrame)
from PySide6.QtCore import Qt
from plotting import ScatterWidget
import numpy as np  


MATCH_COLOR = '#fa3737'
BASIC_COLOR = '#7aabfa'

class GUI(QMainWindow):
    """Main GUI class."""
    def __init__(self):
        # ============================================= GENERAL ==========================================
        super().__init__()  
        self.setWindowTitle("SAM-GEM")
        self.setGeometry(100, 100, 1200, 700)

        # Define vars to hold info on selected points / matches
        self.currently_selected = None
        self.first_match = None
        self.second_match = None
        self.third_match = None

        # Fill main window with actual widget
        central = QWidget()
        central.setStyleSheet("background-color: #1f1f1f;")     # TODO: Change color
        self.setCentralWidget(central)

        root = QVBoxLayout(central)      # main layout is vertical
        top = QHBoxLayout()              # top part of vertical layout is arranges widgets horizontally
        bottom = QVBoxLayout()           # so does bottom part


        # ============================================= TOP ==========================================
        # ===LEFT COL=================================================================================
        # WAVEFORMS OF TOP 3 MATCHES
        left = QVBoxLayout()
        left.setSpacing(20)
        left.stretch(1)
        left.setContentsMargins(10, 10, 10, 10)

        first_match = QLabel("FIRST MATCH")
        first_match.setAlignment(Qt.AlignHCenter)

        second_match = QLabel("SECOND MATCH")
        second_match.setAlignment(Qt.AlignHCenter)

        third_match = QLabel("THIRD MATCH")
        third_match.setAlignment(Qt.AlignHCenter)

        frame_style = """QFrame {
                            border: .5px solid #555;
                            border-radius: 4px;
                            border-color: #444444;
                            background-color: #212121;
                            padding: 0px;
                        }"""

        first_frame  = self.make_match_frame("FIRST MATCH", frame_style)
        second_frame = self.make_match_frame("SECOND MATCH", frame_style)
        third_frame  = self.make_match_frame("THIRD MATCH", frame_style)

        # Add space at start
        left.addStretch(1)

        left.addWidget(first_frame, stretch=1)
        left.addWidget(second_frame, stretch=1)
        left.addWidget(third_frame, stretch=1)

        # Add space at bottom
        left.addStretch(3)


        # ===RIGHT COL=================================================================================
        right = QVBoxLayout()
        right.setSpacing(20)
        right.setContentsMargins(10, 10, 10, 10)
        demo_data = {           # TODO: Remove
            'pos': np.random.normal(size=(1000, 2)),
            'size': np.concatenate(([1]*497, [3]*3, [1]*500))*.02,
            'color':        
                                [BASIC_COLOR]*497 +          # convert list of Hex colors to VisPy recognizable RGBA format
                                [MATCH_COLOR]*3 + 
                                [BASIC_COLOR]*500}
        scatter = ScatterWidget(init_data=demo_data,
                                      bg_color='#1f1f1f',
                                      match_color=MATCH_COLOR,
                                      basic_color=BASIC_COLOR)
        scatter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(scatter, stretch=8)
        selected_waveform = self.make_match_frame("CURRENTLY SELECTED", frame_style)
        right.addWidget(selected_waveform, stretch=1)
        right.addStretch(1)

        top.addLayout(left, stretch=4)
        top.addLayout(right, stretch=5)


        # ============================================= BOTTOM ==========================================
        input_line = QLineEdit()
        input_line.setAlignment(Qt.AlignCenter) 
        input_line.setPlaceholderText("")                    # Empty to fix Qt bug 
        input_line.setPlaceholderText("DESCRIBE YOUR DESIRED SOUND HERE..")  # replace with prompt that is centered
        input_line.setMinimumHeight(40)
        input_line.setMaximumWidth(1000)
        input_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        find_sound_btn = QPushButton("Find Your Sound")
        find_sound_btn.setMaximumHeight(30)
        find_sound_btn.setMaximumWidth(1000)
        find_sound_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        find_sound_btn.setStyleSheet("""
            QPushButton {
                background-color: #ac2076;
                color: white;
                border-radius: 8px;
                padding: 8px 24px;
                font-size: 16px;
            }
        """)

        # User input 
        input_row = QHBoxLayout()
        input_row.addStretch(1)
        input_row.addWidget(input_line, stretch=7)
        input_row.addStretch(1)

        # Button to activate search
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(find_sound_btn, stretch=7)
        button_row.addStretch(1)

        bottom.addLayout(input_row)
        bottom.addLayout(button_row)
        bottom.addStretch(1)

        # ADD EVERYTHING TO ROOT
        root.addLayout(top, stretch=17)
        root.addLayout(bottom, stretch=3)


    def make_match_frame(self, text, frame_style):
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(4)
        container_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(text)
        label.setAlignment(Qt.AlignHCenter)

        frame = QFrame()
        frame.setStyleSheet(frame_style)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frame.setMinimumHeight(80)        

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.addStretch()

        container_layout.addWidget(label)
        container_layout.addWidget(frame, 1) 

        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return container


if __name__ == '__main__':
    app = QApplication([])
    gui = GUI()
    gui.show()
    app.exec()