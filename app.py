import sys
import numpy as np  
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                               QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QLineEdit, QSizePolicy,
                               QFrame, QMenuBar)
from PySide6.QtCore import Qt
from src.plotting import ScatterWidget, DraggableWaveform



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
        # TODO: UPDATE PATHS
        self.currently_selected = r"demo_audio\ah_chd120_upstate_B.wav"    # as current placeholder
        self.first_match_pth = r"demo_audio\ah_chd120_upstate_B.wav"
        self.second_match_pth = r"demo_audio\BOS_BRT_Kick_Rumble_One_Shot_Gestalt.wav"
        self.third_match_pth = r"demo_audio\dhg_hat_usg.wav"

        # Fill main window with actual widget
        central = QWidget()
        central.setStyleSheet("background-color: #1f1f1f;")     # TODO: Change color
        self.setCentralWidget(central)

        root = QVBoxLayout(central)      # main layout is vertical
        top = QHBoxLayout()              # top part of vertical layout is arranges widgets horizontally
        bottom = QVBoxLayout()           # so does bottom part

        # ========================================== TOP MENU ========================================
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)
        self.menu = menuBar.addMenu("Menu")
        self.menu.setStyleSheet("""
            QMenu {
                background-color: rgba(43, 43, 43, 180);  
                border-radius: 8px;
                border: none;
                padding: 6px 0px;
            }
            QMenu::item {
                padding: 6px 16px;
                color: white;
                background-color: transparent;
            }
            QMenu::item:selected {
                background-color: rgba(255, 255, 255, 40);
            }
        """)
        self.menu.setAttribute(Qt.WA_TranslucentBackground, True)
        self.menu.setWindowFlags(
            self.menu.windowFlags()
            | Qt.FramelessWindowHint
            | Qt.NoDropShadowWindowHint
        )

        self.menu.addAction("Initialize Backend")
        self.menu.addAction("Set Data Directory")
        self.menu.addAction("Rescan Data Directory & Update DataBase")
        self.menu.addAction("Recluster Data")
        self.menu.addAction("Exit")

        # Imlement menu actions
        self.menu.actions()[4].triggered.connect(lambda: sys.exit())    # close app

        # ============================================= TOP ==========================================
        # ===LEFT COL=================================================================================
        # WAVEFORMS OF TOP 3 MATCHES
        left = QVBoxLayout()
        left.setSpacing(20)
        left.stretch(1)
        left.setContentsMargins(10, 10, 10, 10)

        frame_style = """QFrame {
                            border: .5px solid #555;
                            border-radius: 4px;
                            border-color: #444444;
                            background-color: #212121;
                            padding: 0px;
                        }"""

        first_frame  = self.make_match_frame("FIRST MATCH", frame_style, self.first_match_pth)
        second_frame = self.make_match_frame("SECOND MATCH", frame_style, self.second_match_pth)
        third_frame  = self.make_match_frame("THIRD MATCH", frame_style, self.third_match_pth)

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
        selected_waveform = self.make_match_frame("CURRENTLY SELECTED", frame_style, self.currently_selected) 
        right.addWidget(selected_waveform, stretch=1)
        right.addStretch(1)

        top.addLayout(left, stretch=4)
        top.addLayout(right, stretch=5)


        # ============================================= BOTTOM ==========================================
        input_line = QLineEdit()
        input_line.setAlignment(Qt.AlignCenter) 
        input_line.setPlaceholderText("")                    # empty to fix Qt bug 
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
                background-color: #4786eb;
                color: white;
                border-radius: 8px;
                padding: 8px 24px;
                font-size: 16px;
            }
            QPushButton:pressed {
                background-color: #3a60c9; 
            }
        """)

        "#3a60c9"
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


    def make_match_frame(self, text, frame_style, audio_pth:str) -> QWidget:
        """Creates a frame with label and draggable waveform inside.

        Args:
            text (str): Label text.
            frame_style (str): Style sheet for frame.
            audio_pth (str): Path to audio file to display.
        Returns:
            QWidget: Container widget with label and frame.
        """
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
        frame_layout.addWidget(DraggableWaveform(audio_pth))

        container_layout.addWidget(label)
        container_layout.addWidget(frame, 1) 

        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return container


if __name__ == '__main__':
    app = QApplication([])
    gui = GUI()
    gui.show()
    app.exec()