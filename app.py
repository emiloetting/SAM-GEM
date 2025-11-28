import sys
import os
import numpy as np  
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                               QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QLineEdit, QSizePolicy,
                               QFrame, QMenuBar)
from PySide6.QtCore import Qt, QUrl, QLoggingCategory
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from src.plotting import ScatterWidget, DraggableWaveform
from src.interface import InterFacer


# Verbose = False for audio output
QLoggingCategory.setFilterRules("""
qt.multimedia.ffmpeg.*=false
""")


AUDIO_VOLUME = 0.8
MATCH_COLOR = '#fa3737'
BASIC_COLOR = '#7aabfa'
BACKGROUND_COLOR = '#1f1f1f'
K_MATCHES = 3
CWD = os.getcwd()


class GUI(QMainWindow):
    """Main GUI class."""
    def __init__(self):
        # ============================================= GENERAL ==========================================
        super().__init__()  
        self.setWindowTitle("SAM-GEM")
        self.setGeometry(100, 100, 1200, 700)

        # Define vars to hold info on selected points / matches
        # TODO: UPDATE PATHS
        self.interfacer = InterFacer(cwd=CWD)
        self.currently_selected = None
        self.first_match_pth = None
        self.second_match_pth = None
        self.third_match_pth = None
        self.data_dict = None
        
        # Audio support
        self.audio = QAudioOutput()
        self.audio.setVolume(AUDIO_VOLUME)   # adapt to taste
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio)


        # Fill main window with actual widget
        central = QWidget()
        central.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")     
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

        # Implement menu actions
        self.menu.actions()[0].triggered.connect(lambda: self.interfacer.full_setup(self))
        self.menu.actions()[1].triggered.connect(lambda: self.interfacer.set_sample_dir(self))
        self.menu.actions()[4].triggered.connect(lambda: sys.exit())    # close app


        # ============================================= TOP ==========================================
        # ===LEFT COL=================================================================================
        # WAVEFORMS OF TOP 3 MATCHES
        self.left = QVBoxLayout()
        self.left.setSpacing(20)
        self.left.stretch(1)
        self.left.setContentsMargins(10, 10, 10, 10)


        self.first_frame  = self.make_match_frame("FIRST MATCH", self.first_match_pth)
        self.second_frame = self.make_match_frame("SECOND MATCH", self.second_match_pth)
        self.third_frame  = self.make_match_frame("THIRD MATCH", self.third_match_pth)

        # Add space at start
        self.left.addStretch(1)

        self.left.addWidget(self.first_frame, stretch=1)
        self.left.addWidget(self.second_frame, stretch=1)
        self.left.addWidget(self.third_frame, stretch=1)

        # Add space at bottom
        self.left.addStretch(3)


        # ===RIGHT COL=================================================================================
        self.right = QVBoxLayout()
        self.right.setSpacing(20)
        self.right.setContentsMargins(10, 10, 10, 10)
        demo_data = {           # TODO: Remove
            'pos': np.random.normal(size=(1000, 2)),
            'size': np.concatenate(([1]*497, [3]*3, [1]*500))*.02,
            'color':        
                                [BASIC_COLOR]*497 +          # convert list of Hex colors to VisPy recognizable RGBA format
                                [MATCH_COLOR]*3 + 
                                [BASIC_COLOR]*500}
        self.scatter = ScatterWidget(
                                init_data=None,
                                gui_interfacer=self.interfacer,
                                gui_parent=self,
                                bg_color='#1f1f1f',
                                match_color=MATCH_COLOR,
                                basic_color=BASIC_COLOR)
        
        self.scatter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right.addWidget(self.scatter, stretch=8)
        self.currently_selected = self.make_match_frame("CURRENTLY SELECTED: ", self.scatter.selected_sample) 
        self.right.addWidget(self.currently_selected, stretch=1)
        self.right.addStretch(1)

        top.addLayout(self.left, stretch=4)
        top.addLayout(self.right, stretch=5)


        # ============================================= BOTTOM ==========================================
        self.input_line = QLineEdit()
        self.input_line.setAlignment(Qt.AlignCenter) 
        self.input_line.setPlaceholderText("")                    # empty to fix Qt bug 
        self.input_line.setPlaceholderText("DESCRIBE YOUR DESIRED SOUND HERE..")  # replace with prompt that is centered
        self.input_line.setMinimumHeight(40)
        self.input_line.setMaximumWidth(1000)
        self.input_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.find_sound_btn = QPushButton("Find Your Sound")
        self.find_sound_btn.clicked.connect(self.evaluate)
        self.find_sound_btn.setMaximumHeight(30)
        self.find_sound_btn.setMaximumWidth(1000)
        self.find_sound_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.find_sound_btn.setStyleSheet("""
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
        self.input_row = QHBoxLayout()
        self.input_row.addStretch(1)
        self.input_row.addWidget(self.input_line, stretch=7)
        self.input_row.addStretch(1)

        # Button to activate search
        self.button_row = QHBoxLayout()
        self.button_row.addStretch(1)
        self.button_row.addWidget(self.find_sound_btn, stretch=7)
        self.button_row.addStretch(1)

        bottom.addLayout(self.input_row)
        bottom.addLayout(self.button_row)
        bottom.addStretch(1)

        # ADD EVERYTHING TO ROOT
        root.addLayout(top, stretch=17)
        root.addLayout(bottom, stretch=3)


    def make_match_frame(self, text, audio_pth:str) -> QWidget:
        """Creates a frame with label and draggable waveform inside.

        Args:
            text (str): Label text.
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
        frame.setStyleSheet("""QFrame {
                            border: .5px solid #555;
                            border-radius: 4px;
                            border-color: #444444;
                            background-color: #212121;
                            padding: 0px;}""")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frame.setMinimumHeight(80)       
         

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.addStretch()
        waveform = DraggableWaveform(audio_pth, parent_gui=self)
        container.waveform = waveform
        frame_layout.addWidget(container.waveform)

        container_layout.addWidget(label)
        container_layout.addWidget(frame, 1) 

        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

         
        return container
    

    def evaluate(self) -> None:
        """Executes search for described sample."""
        # Take user input
        user_input = self.input_line.text() 
        print("Search button clicked! Input: ", user_input) # debug

        # Select IDs of top 3 matches
        match1, match2, match3 = self.interfacer.find_top_k_matches(user_input, k=K_MATCHES)
        match_ids = [match1[0], 
                     match2[0], 
                     match3[0]]
        
        # Select all IDs and positions in connected DB
        data = self.interfacer._grab_all_pos_and_id_db()

        # Create new dictionary of data for plotting
        self.data_dict = {
            "ids": [dp[0] for dp in data],
            "pos": [[dp[1], dp[2]] for dp in data]
        }

        # Update scatter using newly defined data-dict
        self.scatter.update_plot(match_ids=match_ids, data=self.data_dict)



if __name__ == '__main__':
    app = QApplication([])
    gui = GUI()
    gui.show()
    app.exec()