# MODULE TO ADD BACKEND FUNCTIONALITY TO GUI:
#   - RESCAN DATA DIRECTORY
#   - PROCESS INPUT PROMPT
#   - UPDATE WAVEFORM DISPLAYS
#   - PLAY SOUNDS BY CLICKING ON FRAMES
#   - RECLUSTER DATA 

import os
import sqlite3 
import pickle
import time
import numpy as np
from PySide6.QtWidgets import QWidget, QFileDialog
from sklearn.decomposition import IncrementalPCA
from umap import UMAP
from tqdm import tqdm
from src.create_faiss_index import create_faiss



class InterFacer():
    """Class to handle backend functionality for GUI."""
    def __init__(self, cwd:str) -> None:
        self.cwd = cwd  # set working dir
        self.backend_data_dir = None
        self.db_con = None  # connection to db
        self.crs = None

        self.sample_dir = None # dir containing samples (in subdirs)

    
    def full_setup(self, parent:QWidget) -> bool:
        """Method to automatically fully create backend structure."""
        start = time.time()
        # Define dir where faiss index, DB, PCA and UMAP obejcts are stored
        backend_data_dir = os.path.join(self.cwd,"data")

        # Create backend-dir if non-existent
        if not os.path.exists(backend_data_dir):
            os.makedirs(backend_data_dir, exist_ok=True)
        
        # Make user select sample dir
        self.set_sample_dir(parent=parent)
        
        # Early return if sample_dir was not properly selected!
        if self.sample_dir is None:
            print("WARNING: No directory selected. Backend initialization canceled!")
            return
        
        # Get mapping containing indices, embeddings and filepaths of all scanned samples
        feature_mappings = create_faiss(sample_dir=self.sample_dir,
                                        dst_dir=backend_data_dir)
        
        # Create (new) DB
        db_name = "storage.db"
        successful_creation = self._create_new_db(dst_dir=backend_data_dir,
                            db_name=db_name)
        if not successful_creation:
            ("Abort backend initialization")
            return
        
        # Connect to newly created db
        db_pth = os.path.join(backend_data_dir, db_name)
        self._connect_db(db_pth)

        # Train PCA object and reduce embeds
        pca_dst_pth = os.path.join(backend_data_dir, "ipca.pkl")
        print("Amount embeds: ",len(feature_mappings["embedding"]))
        trained_ipca = self._train_pca(dst_pth=pca_dst_pth,
                                       embeds=feature_mappings["embedding"])
        pca_embeds = self._reduce_to_20(embeds=feature_mappings["embedding"],
                                        ipca=trained_ipca)
        
        # Train UMAP object and reduce embeddings pre-reduced by pca
        umap_dst_pth = os.path.join(backend_data_dir, "umap.pkl")
        trained_umap = self._train_umap(dst_pth=umap_dst_pth,
                                        embeds=pca_embeds)
        two_dim_embeds = self._reduce_to_2(embeds=pca_embeds, 
                                           umap=trained_umap)
        
        # FILL DATABASE
        # Sanity check
        assert len(feature_mappings["id"]) == len(feature_mappings["path"]) == len(two_dim_embeds), "Incoherent amount of ids, paths and 2d-corrdinates. Database will not be filled!"
        for i in range(len(two_dim_embeds)):
            insertion = []
            insertion.append(int(feature_mappings["id"][i]))
            insertion.append(str(feature_mappings["path"][i]))
            insertion.append(float(two_dim_embeds[i][0]))  # x-coordinate
            insertion.append(float(two_dim_embeds[i][1]))  # y-coordinate

            self._add2db(features=tuple(insertion),
                         auto_commit=False)

        self.db_con.commit()
        
        print(f"Backend initialization finished in {time.time() - start:.2f}s!")
        return
        
        
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
    
    
    def _create_new_db(self, dst_dir:str, db_name:str) -> bool:
        """Method to create and save new database ("storage.db") containing ID, paths, embeddings, x_pos, y_pos.\n
        **WARNING**: Will overwrite existing database if found!
        
        Args:
            dst_dir (str): Directory in which to store database.
            db_name (str): Name of database.
            
        Returns:
            success (bool): Whether or not DB and table were created successfully.
        """
        try:
            # Establish connection (auto-create)
            dst = os.path.join(dst_dir, db_name)

            # Delete pre-existing db 
            if os.path.exists(dst):
                os.remove(dst)
            
            con = sqlite3.connect(dst)
            crs = con.cursor()

            # Create single main table
            query = """
            CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            x_pos FLOAT,
            y_pos FLOAT)
            """

            # Apply action within query
            crs.execute(query)
            con.commit()

            return True     # signal everything executed correctly
        
        except Exception as e:
            print(f"Creation of new databse failed:\n{e}")
            return False    # signal of failure
        

    def _connect_db(self, pth:str) -> None:
        """Connects to database at given path.
        
        Args:
            pth (str): Path to database.
            
        Returns: 
            None
        """
        # Check file existence
        if (not os.path.exists(pth)) or (not os.path.isfile(pth)) or (not pth.endswith("db")):
            raise FileNotFoundError(f"No valid datebase found at {pth}!")
        
        self.db_con = sqlite3.connect(pth)
        self.crs = self.db_con.cursor()
        

    def _train_pca(self, dst_pth:str, embeds: np.array) -> IncrementalPCA:
        """Creates new instance of PCA-dim-reducer, trains and stores it as pkl-file.
        
        Args:
            dst_pth (str): Where to to store PCA-object. Filetype must be pkl!
            embeds (np.array): Numpy-Array containg 512-dimensional CLAP-embeddings.
        
        Returns:
            reducer (IncrementalPCA): Trained reducer to bring embeds down to 20 dims.
        """
        if not dst_pth.endswith("pkl"):
            raise ValueError("Argument <dst_pth> must be filepath leading to .pkl-file!")

        # Use incremental-PCA for progress-display
        ipca = IncrementalPCA(n_components=20)
        
        for i in tqdm(embeds, 
                      desc="Training PCA", 
                      total=len(embeds), 
                      colour="green", 
                      leave=False, 
                      bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}'):
            ipca.partial_fit(embeds)

        # Store as .pkl
        with open(os.path.join(self.backend_data_dir,"ipca.pkl"), "wb") as f:
            pickle.dump(ipca, f)
        
        return ipca
    

    def _reduce_to_20(self, embeds:np.array, ipca:IncrementalPCA) -> np.array:
        """Reduces 512-dim embeddings to 20 dims using pre-trained ipca-object.
        
        Args:
            embeds (np.array): Array containg 512-dimensional CLAP embeddings.
            ipca (IncrementalPCA): Pre-trained PCA object used to reduce dims of given embeddings.
            
        Returns:
            reduced (np.array): Embeddings reduced to 20 dims.
        """
        return ipca.transform(embeds)

    
    def _train_umap(self, dst_pth:str, embeds:np.array) -> UMAP:
        """Trains UMAP-reducer on embeds and stores at dst_pth.
        
        Args:
            dst_pth (str): Path where to store trained UMAP-transformer.
            embeds (np.array): Embeddings to train UMAP-transforer.
        
        Returns:
            umap (UMAP): Trained UMAP-transformer.
        """
        if not dst_pth.endswith("pkl"):
            raise ValueError("Argument <dst_pth> must be filepath leading to .pkl-file!")
        
        umap = UMAP(n_components=2,
                    verbose=True,
                    tqdm_kwds={
                        "bar_format": '[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
                        "colour": "red",
                        "desc": "Training UMAP",
                        "leave": "False"
                        }
                    )   # allow for terminal output to keep track of training
        umap.fit(embeds)

        # Store
        with open(os.path.join(self.backend_data_dir, "umap.pkl"), "wb") as f:
            pickle.dump(umap, f)
        
        return umap
    

    def _reduce_to_2(self, embeds:np.array, umap:UMAP) -> np.array:
        """Reduces20-dim embeddings to 2 dims using pre-trained umap-object.

        Args:
            embeds (np.array): 20-dimensional embeddings.
            umap (UMAP): pre-trained UMAP-transfomer.
        
        Returns:
            reduced (np.array): Embeddings reduced using pre-trained UMAP-transformer.
        """
        return umap.transform(embeds)
    

    def _add2db(self, features:tuple, auto_commit:bool=False) -> None:
        """Adds entry into table "data" of connected database.
        
        Args:
            features (tuple): Tuple containing ID, path, x-pos and y-pos (in that particular order!) for a singular datapoint.
            auto_commit (bool): Whether or not to commit entry-transaction.
            
        Returns:
            None
        """
        # Early return if no connection established
        if (self.db_con is None) or (self.crs is None):
            print("No database connected. Insertion not possible!")
            return  
        
        if len(features) != 4:
            raise ValueError(f"Too many entries in argument <feature>. Expected 4, got {len(features)}")
        
        # Add to db
        query = """INSERT INTO data (id, path, x_pos, y_pos) VALUES (?, ?, ?, ?)"""
        self.crs.execute(query, features)
        if auto_commit:
            self.db_con.commit()
    