from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt
from plotting import ScatterWidget
import numpy as np  




class GUI(QMainWindow):
    """Main GUI class."""
    def __init__(self):
        super().__init__()  
        self.setWindowTitle("SAM-GEM")
        self.setGeometry(100, 100, 800, 600)

        # Fill main window with actual widget
        central = QWidget()
        central.setStyleSheet("background-color: #2c3e50;")     # TODO: Change color
        self.setCentralWidget(central)

        root = QVBoxLayout(central)      # oberste Vertikale
        top = QHBoxLayout()              # oben: zwei Spalten nebeneinander
        bottom = QVBoxLayout()           # unten: ein Bereich über ganze Breite

        # LINKE SPALTE
        left = QVBoxLayout()
        left.addWidget(QLabel("Links"),  alignment=Qt.AlignHCenter)
        left.addWidget(QPushButton("A"), alignment=Qt.AlignHCenter)

        # RECHTE SPALTE
        right = QVBoxLayout()
        demo_data = {           # TODO: Remove
            'pos': np.random.normal(size=(1000, 2)),
            'size': np.concatenate(([1]*497, [3]*3, [1]*500))*.02,
            'color':        
                                ['#7aabfa']*497 +          # Convert list of Hex colors to VisPy recognizable RGBA format
                                ['#fa3737']*3 + 
                                ['#7aabfa']*500}
        
        right.addWidget(ScatterWidget(init_data=demo_data), alignment=Qt.AlignHCenter)
        right.addWidget(QPushButton("B"), alignment=Qt.AlignHCenter)

        top.addLayout(left)
        top.addLayout(right)

        # unterer Bereich
        bottom.addWidget(QLabel("Unten"), alignment=Qt.AlignHCenter)
        bottom.addWidget(QPushButton("C"), alignment=Qt.AlignHCenter)

        root.addLayout(top)
        root.addLayout(bottom)


if __name__ == '__main__':
    app = QApplication([])
    gui = GUI()
    gui.show()
    app.exec()