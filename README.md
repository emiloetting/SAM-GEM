<img width="1280" height="498" alt="Logo_new_short" src="https://github.com/user-attachments/assets/512d4a9a-4c4b-4c60-a3b1-4a8d86ea887a" />


> [!CAUTION]
> This repository is currently W.I.P.

<br>

# Interactive Text-Based Semantic Audio Search
SAM-GEM is a python-based application enabling an intuitive and interactive audio-search-process.  
Use SAM-GEM to find your sounds using descriptive texts instead of browsing through your sample-folders.  
Get an instant overview over all of your audio samples and explore them on a visual basis.

<br>

# Core-Features  
   
- 📝**Text-Based Semantic Search**  
Search for what you *hear*, not a filename (e.g. "Big Fat Kick Drum").  
Letter case does not matter.
  
- 📎**Drag-And-Droppable Audio**  
In order to implement the found sample into your workflow, simple drag and drop the respective waveform into your project!

- 🔀**Interactive Visualization**  
Interactively explore your audio-catalog display as a two-dimensional scatterplot.
Within, the matches are highlighted as red square, indicating locations containing more samples that might fit your needs.  
A yellow square indicates your currently selected sample (also drag-and-droppable)

<br>

# Demo 
![SAM-GEM_demo](https://github.com/user-attachments/assets/39659166-bc2b-48c6-badb-0c9ee93bdc47)

<br>

# Getting Started
## 1️⃣ **Clone the repository**   
For example, use 
```
git clone https://github.com/emiloetting/SAM-GEM.git
```  

<br>

## 2️⃣ **Install dependancies**  
Install all required packages into your environment, e.g. using  
```
pip install -r requirements.txt
```

⚠️**Note**:  
The repository contains two separate files listing required packages.  
This is to separate the packages for simply using the app and for finetuning the [underlying CLAP-model by LAION](https://github.com/LAION-AI/CLAP).  
If you want to train, make sure to install all packages!  

<br>

## 3️⃣ **Execute the `app.py`**  
This will open the GUI and enable access to all functionalities.  

<br>

## 4️⃣ **Sync your audio catalog to enable exploration**  
In order to sync your sample library, click the `menu` button in the top-left corner and select `Initialize Backend`. 
This will open up your file-explorer and allow you to set the directory containg all of your samples.  
This is a one-time thing only (see section *outlook* for more).  

⚠️**Important**:  
Of course, all of your actual sample-files may be within several subdirectories, what matters is that no samples you wish to be included lay *outside* of this directory!  

<br>

![SAM-GEM_demo_init](https://github.com/user-attachments/assets/4262ef79-8be7-47e7-95b5-6c2c5f9ca24d)

<br>

## 5️⃣ **Grab a cup of tea and enjoy 🍵 ;)**   
Synchronizing and evaluating your samples may take some minutes (~10-12 minutes for 6k files).  
This will include embedding-creation for all your samples and dimensionality-reduction using UMAP to build the database as foundation for the interactive plot.  

<br>

# Outlook 
Of course, SAM-GEM is not yet finished.  
Currently, we are working on:  
- **Quickly Add New Samples**
We are currently working on a feature to quickly add new samples to the existing database, so that adding a kickdrum to your library does not lead to recalculating your whole library but only the added sounds.  
This feature will be available within the `menu` found within the GUI.  
Look out for `Rescan Data Directory & Update DataBase`!

- **Clustering**
Also, clustering will be implemented, allowing for a better distinction of "sample-groups" (e.g. kicks) within your data, colorizing the interactive plot and making it easier to get a sense of what's wihtin your sample library.

- **Loop vs One-Shot Classifier**  
Last but not least, we are working on a classifier to easily limit the search-results to loops or one-shots if desired!

<br>

# About
Sample selection in music prodution takes a lot of time, can often break the flow and get you out of "the zone".  
To avoid getting stuck in the midst of file-names, we propose a semantic search of desired audio files.  
Using LoRA, we finetuned the [htsat-fused CLAP-model by LAION](https://github.com/LAION-AI/CLAP) to include industry-words like "punchy", "beefy", etc.  
CLAP-based embeddings of the search query will be copmared to those of all your selected audio-samples using [FAISS](https://github.com/facebookresearch/faiss).  
Using UMAP, the connected sample-library will be displayed in an interactive scatterplot.  
**Keep in mind**: This is still a W.I.P and more finetuning is necessary. 
