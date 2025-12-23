import os, sys, json
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput



SAMPLES_DIR = r"E:\Samples" # Parent dir of audio files
ANNOTATIONS_JSON = "annotations.json"  



def list_audio_files(root):
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(r, f))
    return sorted(paths)

class MiniAnnot(QWidget):
    def __init__(self, annotations):
        super().__init__()
        self.setWindowTitle("Mini Annotator (3 Buttons)")
        self.paths = list_audio_files(SAMPLES_DIR)
        self.paths = [path for path in self.paths if (os.path.basename(path) not in annotations) or (annotations[os.path.basename(path)] == "")]     # adapt to only unlabeled files
        self.labels = annotations
        self.idx = 0

        self.audio_out = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)

        self.info = QLabel("Shortcuts: Enter, R (Replay), → (+5s)")
        self.file_lbl = QLabel("")
        self.edit = QLineEdit()
        self.edit.returnPressed.connect(self.save_next)

        row = QHBoxLayout()
        self.btn_enter = QPushButton("Enter (Save & Next)")
        self.btn_again = QPushButton("Play Again")
        self.btn_fwd = QPushButton("+5 s")
        self.btn_enter.clicked.connect(self.save_next)
        self.btn_again.clicked.connect(self.restart)
        self.btn_fwd.clicked.connect(self.forward5)
        for b in (self.btn_enter, self.btn_again, self.btn_fwd):
            row.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.file_lbl)
        lay.addWidget(self.edit)
        lay.addLayout(row)

        self.load_current()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.save_next()
        elif e.key() == Qt.Key.Key_Right:
            self.forward5()
        elif e.key() in (Qt.Key.Key_R, ):
            self.restart()

    def restart(self):
        self.player.setPosition(0)
        self.player.play()

    def forward5(self):
        self.player.setPosition(self.player.position() + 5000)
        self.player.play()

    def save_next(self):
        if not self.paths or self.idx >= len(self.paths):
            return

        text = self.edit.text().strip()
        if not text:             # don't allow empty labels
            self.edit.setFocus()
            self.edit.selectAll()
            return

        fpath = self.paths[self.idx]
        self.labels[os.path.basename(fpath)] = text

        with open(ANNOTATIONS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)

        self.idx += 1
        self.load_current()

    def load_current(self):
        if not self.paths:
            self.file_lbl.setText("No audio files found.")
            self.set_controls(False)
            return
        if self.idx >= len(self.paths):
            self.file_lbl.setText("Everything annotated. Done!")
            self.set_controls(False)
            return

        fpath = self.paths[self.idx]
        self.file_lbl.setText(f"{self.idx+1}/{len(self.paths)}  ·  {os.path.basename(fpath)}")
        self.edit.setText("") 
        self.edit.setEnabled(True); self.btn_enter.setEnabled(True); self.btn_again.setEnabled(True); self.btn_fwd.setEnabled(True)
        self.edit.setFocus()
        self.player.setSource(QUrl.fromLocalFile(fpath))
        self.player.play()

    def set_controls(self, enabled):
        self.edit.setEnabled(enabled)
        self.btn_enter.setEnabled(enabled)
        self.btn_again.setEnabled(enabled)
        self.btn_fwd.setEnabled(enabled)

    def closeEvent(self, e):
        self.player.stop()
        return super().closeEvent(e)

if __name__ == "__main__":
    with open(ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
        try:
            annotations = json.load(f)
        except Exception as e:
            raise IndexError("Error loading existing annotations.json:\n " + str(e))
    
    app = QApplication(sys.argv)
    w = MiniAnnot(annotations=annotations)
    w.resize(650, 140)
    w.show()
    sys.exit(app.exec())
