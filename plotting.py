import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtGui import QColor



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

        
    def fill_scatter(self) -> None:
        """Update scatter plot with current data."""
        self.plot.removeItem(getattr(self, 'scatter', pg.ScatterPlotItem()))  # remove old scatter if existing
        self.scatter = pg.ScatterPlotItem(
            pos=self.data['pos'],
            size=self.data['size'],
            brush=self.data['color'],
            pen=None,
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

        # Size-adjustment
        min_size, max_size = np.min(self.data['size']), np.max(self.data['size'])
        sizes = np.array([min_size]*len(self.data['size']))
        sizes[indices] = max_size
        self.data['size'] = sizes

        self._order_points_in_plot()
        self.fill_scatter()
    

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