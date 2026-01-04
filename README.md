<img width="1280" height="670" alt="Kopie von SAM-GEM Logo-Photoroom - Kopie" src="https://github.com/user-attachments/assets/2debe121-8f15-48d1-a21f-c8defeeb35b5" />


> [!CAUTION]
> This repository is currently W.I.P.

# Interactive Text-Based Semantic Audio Search
SAM-GEM is a python-based application enabling an intuitive and interactive audio-search-process.  
Use SAM-GEM to find your sounds using descriptive texts instead of browsing through your sample-folders.  
Get an instant overview over all of your audio samples and explore them on a visual basis.

# About
Sample selection in music prodution takes a lot of time and can often break the flow and get you out of "the zone".  
To avoid getting stuck in the midst of file-names, we propose a semantic search of desired audio files.  
Using LoRA, we finetuned the [htsat-fused CLAP-model by LAION](https://github.com/LAION-AI/CLAP) to include industry-words like "punchy", "beefy", etc.  
CLAP-based embeddings of the search query will be copmared to those of all your selected audio-samples using [FAISS](https://github.com/facebookresearch/faiss).  
Using UMAP, the connected sample-library will be displayed in an interactive scatterplot.  
**Keep in mind**: This is still a W.I.P and more finetuning is necessary. 

# Core-Features
<img width="1195" height="757" alt="image" src="https://github.com/user-attachments/assets/e1326d7e-8812-4a9e-9cad-612f45f05162" />

- Top-3 matches presented as drag-and-droppable waveforms that play the respective sound on-click
- Matches will be highlighted within scatterplot by bigger red dots.
- Scatterplot displays whole connected audio-library interactively
- Sample selected in plot is highlighted as big yellow dot
- Drag-and-Droppable waveform of selected sample is loaded underneath scatterplot (also plays on-click)
- Input-Field to describe your desired sound (e.g. "fat kick drum")
