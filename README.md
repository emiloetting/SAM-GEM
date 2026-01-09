<img width="1280" height="670" alt="Kopie von SAM-GEM Logo-Photoroom - Kopie" src="https://github.com/user-attachments/assets/2debe121-8f15-48d1-a21f-c8defeeb35b5" />


> [!CAUTION]
> This repository is currently W.I.P.

# Interactive Text-Based Semantic Audio Search
SAM-GEM is a python-based application enabling an intuitive and interactive audio-search-process.  
Use SAM-GEM to find your sounds using descriptive texts instead of browsing through your sample-folders.  
Get an instant overview over all of your audio samples and explore them on a visual basis.


# Core-Features  
   
- 📝**Text-Based Semantic Search**  
Search for what you *hear*, not its filename (e.g. "Big Fat Kick Drum").  
Letter case does not matter.
  
- 📎**Drag-And-Droppable Audio**  
In order to implement the found sample into your workflow, simple drag and drop the respective waveform into your project!

- 🔀**Interactive Visualization**  
Interactively explore your audio-catalog display as a two-dimensional scatterplot.
Within, the matches are highlighted as red square, indicating locations containing more samples that might fit your needs.  
A yellow square indicates your currently selected sample (also drag-and-droppable)


# Demo 
![SAM-GEM_demo](https://github.com/user-attachments/assets/39659166-bc2b-48c6-badb-0c9ee93bdc47)


# About
Sample selection in music prodution takes a lot of time, can often break the flow and get you out of "the zone".  
To avoid getting stuck in the midst of file-names, we propose a semantic search of desired audio files.  
Using LoRA, we finetuned the [htsat-fused CLAP-model by LAION](https://github.com/LAION-AI/CLAP) to include industry-words like "punchy", "beefy", etc.  
CLAP-based embeddings of the search query will be copmared to those of all your selected audio-samples using [FAISS](https://github.com/facebookresearch/faiss).  
Using UMAP, the connected sample-library will be displayed in an interactive scatterplot.  
**Keep in mind**: This is still a W.I.P and more finetuning is necessary. 
