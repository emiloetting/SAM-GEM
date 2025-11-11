import os
import numpy as np
import soundfile as sf
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtGui import QColor, QDrag, QMouseEvent
from PySide6.QtCore import (QPoint, Qt, QUrl, 
                            QMimeData)



class ScatterWidget(QWidget):
    """Class to create scatterplot of audio features via PyQtGraph."""
    def __init__(self, init_data: dict, 
                 bg_color: str = '#FFFFFF', 
                 match_color: str = '#7aabfa', 
                 basic_color: str = '#fa3737') -> None: 
        # Init parent object
        super().__init__() 

        # Load and format initial data
        self.data = init_data
        self._convert_from_hex()

        # Define colors to update point colors later
        self.match_color = match_color
        self.basic_color = basic_color

        # Create pyQtGraph
        self.plot = pg.PlotWidget(background=bg_color)
        self.plot.hideAxis('bottom')    # rm axes
        self.plot.hideAxis('left')

        # Define Layout
        layout = QVBoxLayout(self)      # Apply layout to this widget
        layout.setContentsMargins(0, 0, 0, 0)   # Make content fill full widget
        layout.addWidget(self.plot)    # Widget now has vertical layout containing VisPy canvas

        # Init scatter plot
        self.scatter = None # will be created in fill_scatter
        self.fill_scatter()


    def _connect_interactions(self) -> None:
        """Connect interaction events to functions."""
        self.scatter.sigClicked.connect(self.on_point_clicked)


    def load_data(self, data: dict) -> None:
        """Load data into scatter plot widget.
        
        Args:
            data (dict): Dictionary containing 'pos', 'size', and 'color' keys.
        """
        # Check passed data
        assert all(key in data for key in ['pos', 'size', 'color']),                     "Data dictionary must contain 'pos', 'size', and 'color' keys."
        assert data['pos'].shape[1] == 2,                                                "Position data must be 2-dimensional." 
        assert len(data['pos']) == len(data['size']) == len(data['color']),              "Length of 'pos', 'size', and 'color' must be the same."
        assert all(isinstance(c, str) and (c.startswith("#")) for c in data['color']),   "Colors are to be passed as strings containing HEX-encoded color."
        assert len(data['pos']) > 0,                                                     "Data arrays must not be empty."
        
        data['color'] = [QColor(clr) for clr in data['color']]  # Convert list of Hex colors to VisPy recognizable RGBA format
        self.data = data    # load into scatter
        self._order_points_in_plot()
        self.fill_scatter()
        self._set_initial_range()

        
    def fill_scatter(self) -> None:
        """Update scatter plot with current data."""
        if self.scatter is not None:
            self.plot.removeItem(self.scatter)  # remove old scatter if existing

        self.scatter = pg.ScatterPlotItem(
            pos=self.data['pos'],
            size=self.data['size'],
            brush=self.data['color'],
            pen=pg.mkPen('w', width=0.1),
            pxMode=False
        )

        self._connect_interactions()
        self.plot.addItem(self.scatter)
        

    def _convert_from_hex(self) -> None:
        """Convert Hex color strings to RGBA format."""
        self.data['color'] = [QColor(clr) for clr in self.data['color']]


    def highlight_matching(self, indices: np.ndarray[int]) -> None:
        """To highlight top 3 matching audio representative points."""
        # Color-adjustment
        colors = np.array([self.basic_color]*len(self.data['color']))
        colors[indices] = self.match_color
        self.data['color'] = colors
        self._convert_from_hex()

        # Size-adjustment
        min_size, max_size = np.min(self.data['size']), np.max(self.data['size'])
        sizes = np.array([min_size]*len(self.data['size']))
        sizes[indices] = max_size
        self.data['size'] = sizes

        self._order_points_in_plot()
        self.fill_scatter()
    

    def _set_initial_range(self):
        """Set initial view range to fit all points."""
        pos = np.asarray(self.data['pos'])
        x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
        y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
        self.vb.setRange(
            xRange=(x_min, x_max),
            yRange=(y_min, y_max),
            padding=0.05
        )


    def _order_points_in_plot(self) -> None:
        """Function to place bigger points behind normal sized points to increase explorability of data close to matching points."""
        sort_idx = np.argsort(self.data['size'])[::-1]  # indices to sort sizes descending
        self.data['size'] = np.array(self.data['size'])[sort_idx]
        self.data['color'] = np.array(self.data['color'])[sort_idx]
        self.data['pos'] = np.array(self.data['pos'])[sort_idx]


    def on_point_clicked(self, scatter, points):
        """Defines action on point click - currently only debug output."""
        for p in points:
            print(f"Clicked on point at: {p.pos()}")
        #TODO: Fill with useful stuff


class DraggableWaveform(QWidget):
    """Widget mit Waveform-Anzeige + Drag&Drop der zugehörigen Audiodatei."""
    def __init__(self, audio_pth: str, wav_color: str = '#fa3737', parent=None) -> None:
        super().__init__(parent)

        self.audio_pth = os.path.abspath(audio_pth)
        self.wav_clr = wav_color
        self._drag_start_pos = QPoint()

        layout = QVBoxLayout(self) # create layout to hold plot widget
        layout.setContentsMargins(0, 0, 0, 0)

        # Create Widget to plot waveform
        self.plot_widget = DraggablePlotWidget(self)
        self.plot_widget.setBackground(None)
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.setMouseEnabled(x=False, y=False)

        layout.addWidget(self.plot_widget)  # add to layout
        self.show_wav()  # draw waveform


    def show_wav(self) -> None:
        """Loads and displays waveform of audio file."""
        data, samplerate = sf.read(self.audio_pth)

        if data.ndim > 1:
            data = data.mean(axis=1)

        t_steps = np.linspace(0, len(data) / samplerate, len(data)) # time steps for x-axis

        pen = pg.mkPen(color=self.wav_clr, width=1.5)   # define line settings
        self.plot_widget.plot(t_steps, data, pen=pen, clear=True)     # update plot


    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Defines behavior on mouse press event for drag & drop."""
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.position().toPoint()   # set start pos to later start drag if mouse moved enough
        super().mousePressEvent(event)


    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Defines behavior on mouse move event for drag & drop."""
        #Check if left button is pressed
        if not (event.buttons() & Qt.LeftButton):
            return

        # Check if mouse moved enough to start drag
        if (event.position().toPoint() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        # Validate path to audio file
        if not os.path.isfile(self.audio_pth):
            print("Traying to drag non-existing file: \n", self.audio_pth)
            return

        drag = QDrag(self)
        mime = QMimeData()  # store information in clipboard style
        mime.setUrls([QUrl.fromLocalFile(self.audio_pth)])  # Define file to be dragged
        drag.setMimeData(mime)  # add path info to drag object
        drag.exec(Qt.CopyAction)    # start drag with copy action


    def _update(self, new_path: str) -> None:
        """Update audio path and redraw waveform.

        Args:
            new_path (str): New path to audio file.
        """
        self.audio_pth = os.path.abspath(new_path)
        self.show_wav()
    
class DraggablePlotWidget(pg.PlotWidget):
    """Subclass of PlotWidget to allow mouse events on parent DraggableWaveform."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def mousePressEvent(self, event) -> None:
        self.parent.mousePressEvent(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self.parent.mouseMoveEvent(event)
        super().mouseMoveEvent(event)