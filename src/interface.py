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
import faiss
from PySide6.QtWidgets import QWidget, QFileDialog
from sklearn.decomposition import IncrementalPCA
from umap import UMAP
from tqdm import tqdm
from src.create_faiss_index import create_faiss
from src.model import MODEL, PROCESSOR, TOKENIZER        # Core-model used for embedding gen in whole of project




class InterFacer():
    """Class to handle backend functionality for GUI."""
    def __init__(self, cwd:str) -> None:
        self.cwd = cwd  # set working dir
        self.backend_data_dir = os.path.join(self.cwd, "data")
        self.db_con = None  # connection to db
        self.crs = None
        self.faiss = None
        self.ipca = None
        self.umap = None

        self.sample_dir = None # dir containing samples (in subdirs)
        self.model = MODEL
        self.processor = PROCESSOR
        self.tokenizer = TOKENIZER
        self.try_connections()

    
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
        
        # Get build index & mapping containing indices, embeddings and filepaths of all scanned samples
        index, feature_mappings = create_faiss(sample_dir=self.sample_dir,
                                        dst_dir=backend_data_dir)
        self.index = index


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
        # self.ipca = self._train_pca(dst_pth=pca_dst_pth,
        #                                embeds=feature_mappings["embedding"])
        # pca_embeds = self._reduce_to_20(embeds=feature_mappings["embedding"],
        #                                 ipca=self.ipca)
        
        # Train UMAP object and reduce embeddings pre-reduced by pca
        umap_dst_pth = os.path.join(backend_data_dir, "umap.pkl")
        self.umap = self._train_umap(dst_pth=umap_dst_pth,
                                        embeds=feature_mappings["embedding"])
        two_dim_embeds = self._reduce_to_2(embeds=feature_mappings["embedding"], 
                                           umap=self.umap)
        
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
    

    def _gen_embed(self, prompt:str) -> np.array:
        """Generates CLAP-Embedding from input string. 
        
        Args:
            prompt (str): Phrase or words to compute embedding of.
        
        Returns:
            embed (np.array): Computed embedding.
        """
        assert type(prompt) == str, f"Invalid input. Got <{type(prompt)}>, expected <str>!"
        base_embed = self.tokenizer([prompt.strip()],
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt")
        embed_wrong_dims = self.model.get_text_features(base_embed["input_ids"], base_embed["attention_mask"])     # embed is torch.Tensor
        embed_better = embed_wrong_dims.numpy().squeeze()

        if embed_better.ndim == 1:
            embed = embed_better[np.newaxis, :]                        # make it (1, d) for FAISS
        faiss.normalize_L2(embed)
        return embed    
    

    def find_top_k_matches(self, prompt:str, k:int) -> list[tuple[int,str]]:
        """Find k-best matches for L2-normalized input-embedding from FAISS index at self.index.
        
        Args:
            prompt (str): Phrase or words to compute embedding of.
            k (int): Amount of matches to be found.
        
        Returns:
            None
        """
        # Make sure input is actual string
        assert type(prompt) == str, f"Invalid input. Got <{type(prompt)}>, expected <str>!"

        # Validate connection to database
        self.__check_db_con()
        # Validate connection to FAISS-index
        if self.index is None:
            raise ConnectionError("Instatiated Object of class <InterFacer> has not been connected to any instance of FAISS-Index!")
        
        # Compute embedding and normalize
        input_embed = self._gen_embed(prompt)
        print("Input Embed: ", input_embed)
        print("Shape of input_embed: ", input_embed.shape)

        # Get indices of best matches
        _, indices = self.index.search(input_embed, k)
        indices = indices.flatten()     # reformat for grabbing from db
        print("Found matching IDs: ", indices)  
        # Get respective filepaths
        paths = self._grab_paths_from_db(ids=indices)

        # Validate results based on amount of IDs and paths
        if not (len(indices) == len(paths)):
            raise IndexError(
                f"Got {len(indices)} indices, but found {len(paths)} corresponding filepaths! Check connected database for trouble shooting.\n",
                f"IDs: {indices}\n"
                f"Paths: {paths}"
            )

        return list(zip(indices, paths))


    def __check_db_con(self) -> None:
        """Raises Connection Error if no databse is connected, instantiates cursor is non was set previously"""
        # Check whether connections for finding and indicating matching samples are established 
        if self.db_con is None:
            raise ConnectionError("Instatiated Object of class <InterFacer> has not been connected to a database!")
        if self.crs is None:
            self.crs = self.db_con.cursor()
        

    def _grab_paths_from_db(self, ids: np.ndarray|list) -> list[str]:
        """Grabs filepaths to corresponding IDs from connected database

        Args:
            ids (np.array): Array or list containing IDs to extract respective filepath to. 
        
        Returns:
            paths (list[str]): Corresponding filepaths
        """
        query = "SELECT path FROM data WHERE id=?"
        matches = []
        for i in ids.tolist():
            print("ID: ", i)
            self.crs.execute(query, (i,))  # pass as tuple with single element
            matches.append(self.crs.fetchall()[0])
        return matches
    

    def _grab_all_pos_and_id_db(self) -> list[tuple[int, float, float]]:
        """Extracts x-pos and y-pos for each datapoint in connected database."""
        self.__check_db_con()   # Check con
        query = """SELECT id, x_pos, y_pos FROM data;"""
        self.crs.execute(query)
        return self.crs.fetchall()


    def _grab_path_by_pos(self, pos:tuple) -> list:
        """Get path to audio sample via coordinates of the representing data point within plot.
        Args:
            pos (tuple): x- and y-position of corresponding audio sample.
            
        Return:
            path (list): The respective path(s) to the audio sample(s) as stored in connected database.
        """
        self.__check_db_con()
        query = """
        SELECT path FROM data 
        WHERE x_pos=? AND y_pos=?
        """
        self.crs.execute(query, pos)
        return self.crs.fetchall() 
    
    
    def _grab_id_by_pos(self, pos:tuple) -> int:
        """Fetches ID of point at corresponding position from connected database"""
        self.__check_db_con()
        self.crs.execute("SELECT id from data WHERE x_pos=? AND y_pos=?;", pos)
        return int(self.crs.fetchall()[0])
    
    
    def try_connections(self) -> None:
        """Tries to establish connections to / load UMAP, IPCA, FAISS-index and database."""
        # DB
        try:
            db_path = os.path.join(self.backend_data_dir, "storage.db")
            self._connect_db(pth=db_path)
            print("\nConnection to database successful.")
        except:
            print("\nWARNING: Conncection to database could not be established!")

        # FAISS
        try:
            index_pth = os.path.join(self.backend_data_dir, "audio.faiss")
            self.index = faiss.read_index(index_pth)
            print("Connection to FAISS-index successfull.")
        except:
            print("WARNING: Conncection to FAISS-index could not be established!")

        # UMAP
        try:
            umap_pth = os.path.join(self.backend_data_dir, "umap.pkl")
            with open(umap_pth, "rb") as f:
                self.umap = pickle.load(f)
            print("Connection to UMAP-object successfull.")
        except:
            print("WARNING: Connection to UMAP-object could not be established")

        # IPCA
        try:
            ipca_pth = os.path.join(self.backend_data_dir, "ipca.pkl")
            with open(ipca_pth, "rb") as f:
                self.ipca = pickle.load(f)
            print("Connection to IPCA-object successfull.")
        except:
            print("WARNING: Connection to IPCA-object could not be established")
        