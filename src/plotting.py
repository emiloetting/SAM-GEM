import os
import numpy as np
import soundfile as sf
import pyqtgraph as pg
from random import randint
from src.interface import InterFacer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QApplication,
                               QSizePolicy)
from PySide6.QtGui import QColor, QDrag, QMouseEvent
from PySide6.QtCore import (QPoint, Qt, QUrl, 
                            QMimeData)


NORMAL_SIZE = .05
MATCH_SIZE = 0.15
HARD_DRIVE_PREFIX = "E"
class ScatterWidget(QWidget):
    """Class to create scatterplot of audio features via PyQtGraph."""
    def __init__(self, init_data: dict|None, 
                 gui_interfacer: InterFacer,
                 gui_parent,    # TypeHinting not easily possible due to cross-imports
                 bg_color: str = '#FFFFFF', 
                 match_color: str = '#7aabfa', 
                 basic_color: str = '#fa3737',
                 selected_color: str = "#FAD400") -> None: 
        
        # Init parent object
        super().__init__() 
        self.gui_interfacer = gui_interfacer
        self.gui_parent = gui_parent

        # Load and format initial data
        if init_data is None:
            self.data = {
                "pos": [],
                "color" : [],
                "size" : []
            }
        else:
            self.data = init_data
        self._convert_from_hex()

        # Define colors & sizes to update point colors later
        self.match_color = match_color
        self.normal_color = basic_color
        self.selected_color = selected_color
        self.normal_size = NORMAL_SIZE
        self.match_size = MATCH_SIZE

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

        # Store path of selected audio sample
        self.selected_sample = None
        
        # Last clicked point
        self.last_clicked = None
        self.last_clicked_style = (None, None)    # (brush, size)


    def _connect_interactions(self) -> None:
        """Connect interaction events to functions."""
        self.scatter.sigClicked.connect(self.on_point_clicked)


    def load_data(self, data: dict|None) -> None:
        """Load data into scatter plot widget.
        
        Args:
            data (dict|None): Dictionary containing 'pos', 'size', and 'color' keys. If 'None', self.data will be selected!
        """
        if data is None:
            data = self.data    # Safety mechanism

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
        abs_min = min(x_min, y_min)
        abs_max = max(x_max, y_max)
        self.scatter.getViewBox().setRange(
            xRange=(abs_min, abs_max),
            yRange=(abs_min, abs_max),
            padding=0.05
        )


    def _order_points_in_plot(self) -> None:
        """Method to place bigger points behind normal sized points to increase explorability of data close to matching points."""
        sort_idx = np.argsort(self.data['size'])[::-1]  # indices to sort sizes descending
        self.data['size'] = np.array(self.data['size'])[sort_idx]
        self.data['color'] = np.array(self.data['color'])[sort_idx]
        self.data['pos'] = np.array(self.data['pos'])[sort_idx]


    def update_plot(self, match_ids:list[int], data:dict) -> None:
        """Method to update plot based on user input.
        
        Args:
            ids (list[int]): List of IDs of best matches.
            data (dict): Dictionary containing information on position and IDs
        
        Returns:
            None
        """
        #TODO: set new data dict into self.data based on passed IDs 
        # Validate match-IDs
        assert len(data["pos"]) == len(data["ids"]), f"Unequal amount of IDs and positional information provided!"
        for id in match_ids:
            if not id in data["ids"]:
                raise ValueError("Unknown ID passed as match!")
            
        match_ids = np.array(match_ids)
        data_pos = np.array(data["pos"])
        data_ids = np.array(data["ids"])

        real_matching_idcs = np.argwhere(np.isin(data_ids, match_ids)).flatten()
        sizes = np.array([self.normal_size]*len(data_ids))
        sizes[real_matching_idcs] = self.match_size
        colors = np.array([self.normal_color]*len(data_ids))
        colors[real_matching_idcs] = self.match_color

        self.new_data = {
            'pos': data_pos,
            'size': sizes,
            'color': colors     
        }

        self.load_data(data=self.new_data)

        paths = self.gui_interfacer._grab_paths_from_db(ids=match_ids)
        self.gui_parent.first_frame.waveform._update(new_path=paths[0][0], 
                                                     color=self.match_color)
        self.gui_parent.second_frame.waveform._update(new_path=paths[1][0],
                                                      color=self.match_color)
        self.gui_parent.third_frame.waveform._update(new_path=paths[2][0],
                                                      color=self.match_color)
        
        
    def on_point_clicked(self, scatter, points):
        """Defines action on point click - currently only debug output."""
        for p in points:
            print(f"Clicked on point at: {p.pos()}")    # Debug output to see whether ther were multiple points reached
        self.selected_sample = self.gui_interfacer._grab_path_by_pos((p.pos().x(), p.pos().y()))

        # Set first point in points as the "real" one
        p = points[0]

        # Restore pre-selection-style 
        if self.last_clicked is not None and self.last_clicked_style is not None:
            brush, size = self.last_clicked_style
            self.last_clicked.setBrush(brush)
            self.last_clicked.setSize(size)
        
        # Set tuple containing new style settings
        self.last_clicked_style = (
            p.brush(),
            p.size(),
        )

        # Apply selected-stylings to selected point
        p.setBrush(self.selected_color)
        p.setSize(self.match_size)

        # select latest
        self.last_clicked = p

        # Expect multiple datapoints at same position: load random! TODO: Fix Visualization so that overlaps are unlikely/impossible
        idx = randint(0, len(self.selected_sample)-1)
        print(f"Selected sample path: {self.selected_sample[idx][0]}")
        self.gui_parent.currently_selected.waveform._update(new_path=self.selected_sample[idx][0], 
                                                            color=self.selected_color) 

class DraggableWaveform(QWidget):
    """Widget displaying drag'n'droppable WAV-Form."""
    def __init__(self, audio_pth: str, parent_gui, wav_color: str = '#fa3737') -> None:
        super().__init__()
        self.parent_gui = parent_gui
        self.audio_pth = None
        if not (audio_pth is None):
            self.audio_pth = os.path.abspath(audio_pth)
            

        self.wav_clr = wav_color
        self._drag_start_pos = QPoint()

        layout = QVBoxLayout(self) # create layout to hold plot widget
        layout.setContentsMargins(0, 0, 0, 0)

        # Create Widget to plot waveform
        #TODO:Implement feature to play sound on click
        self.plot_widget = DraggablePlotWidget(self)
        self.plot_widget.setBackground(None)
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.plot_widget)  # add to layout
        self.show_wav()  # draw waveform


    def show_wav(self, color:str|None=None) -> None:
        """Loads and displays waveform of audio file.
        
        Args:
            color (str|None): Color of waveform. If set to None, default color will be used.
            
        Returns:
            None 
        """
        if self.audio_pth is None:
            return  # Early return
        self.audio_pth = str(HARD_DRIVE_PREFIX+self.audio_pth[1:]) 
        data, samplerate = sf.read(self.audio_pth)

        if data.ndim > 1:
            data = data.mean(axis=1)

        t_steps = np.linspace(0, len(data) / samplerate, len(data)) # time steps for x-axis

        pen = pg.mkPen(color=color, width=1.5)   # define line settings
        self.plot_widget.plot(t_steps, data, pen=pen, clear=True)     # update plot

        max_amp = np.max(np.abs(data))
        self.plot_widget.setYRange(-max_amp, max_amp, padding=0)    # Set range to center waveform 


    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Defines behavior on mouse press event for drag & drop."""
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.position().toPoint()   # set start pos to later start drag if mouse moved enough
            self.parent_gui.player.setSource(QUrl.fromLocalFile(self.audio_pth))
            self.parent_gui.player.play()
        super().mousePressEvent(event)


    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Defines behavior on mouse move event for drag & drop."""
        # Check if left button is pressed
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


    def _update(self, new_path: str, color:str|None=None) -> None:
        """Update audio path and redraw waveform.

        Args:
            new_path (str): New path to audio file.
            color (str|None): Color of waveform. If set to None, default color will be used.
        
        Returns:
            None
        """
        self.audio_pth = os.path.abspath(new_path)
        self.show_wav(color=color)
    
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