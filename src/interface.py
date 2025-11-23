# MODULE TO ADD BACKEND FUNCTIONALITY TO GUI:
#   - RESCAN DATA DIRECTORY
#   - PROCESS INPUT PROMPT
#   - UPDATE WAVEFORM DISPLAYS
#   - PLAY SOUNDS BY CLICKING ON FRAMES
#   - RECLUSTER DATA 

import os
import sqlite3 
import pickle
from PySide6.QtWidgets import QWidget, QFileDialog
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import tqdm
from warnings import warn
from create_faiss_index import audio_embeddings_with_paths, create_faiss, add_audios



class InterFacer():
    """Class to handle backend functionality for GUI."""
    def __init__(self) -> None:
        self.cwd = None
        self.db_path = None
        self.connection = None
        self.cursor = None
        self.pca = None
        self.umap = None
        self.faiss = None
        self.sample_dir = None

    
    def _full_setup(self) -> bool:
        # Define dir where faiss index, DB, PCA and UMAP obejcts are stored
        backend_data_dir = os.path.join(self.cwd,"data")

        # Create backend-dir if non-existent
        if not os.path.exists(backend_data_dir):
            os.makedirs(backend_data_dir, exist_ok=True)
        
        # Make user select sample dir
        self.set_sample_dir()
        
        # Early return if sample_dir was not properly selected!
        if self.sample_dir is None:
            print("WARNING: No directory selected. Backend initialization cancelled!")
            return
        
        # Get mapping containing indices, embeddings and filepaths of all scanned samples
        feature_mappings = create_faiss(sample_dir=self.sample_dir,
                                        dst_dir=backend_data_dir)
        
        
    def set_sample_dir(self, parent:QWidget) -> None:
        """ Sets dir at given URL as parent dir containing all audio samples within subdirectories."""
        dir = QFileDialog.getExistingDirectory(
            parent=parent, 
            caption="Select directory containing all samples (samples can be within subdirectories)"
            )

        # Quit if user didn't choose
        if not dir:
            return

        self.sample_dir = dir   # Set sample_dir
    