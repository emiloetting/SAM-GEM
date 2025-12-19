import os
import torch
import json
import faiss
import numpy as np
import seaborn as sns
from laion_clap import CLAP_Module
from random import sample
import matplotlib.pyplot as plt



"""
KONZEPT: 
Für alle annotierten Sounds:
Einmal VOR und einmal NACH Training:

- Annotation in Embedding wandeln
- Sample in Embedding wandeln
- Cos-Sim zw. Sample & Annot
- Cos-Sim von Top3 Matches zu Annotation Embed
- Cos-Sim von Top3 Matches zu zugehörigem Sample

Für Annotation-Audio Embed Cosine Sim als Beispiel: https://github.com/wandb/wandb/issues/4092 
"""


CWD = os.getcwd()
EVAL_DATA_DIR = os.path.join(CWD, "data", "evaluation")
os.makedirs(EVAL_DATA_DIR, exist_ok=True)   # Create dir to store


# Aktuell: 1499 Samples
# Wir nehmen: 300 Random (Großer Teil wo viele ähnlich sind)
SAMPLE_SIZE = 300
UNTRAINED_FAISS_PTH = "data/audio.faiss"


# Load untuned model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UNTRAINED_MODEL = CLAP_Module(enable_fusion=False, device='cpu')
UNTRAINED_MODEL.load_ckpt(verbose=False)


# Load file with annots
with open("new_annotations.json", "r") as f:
    annots = dict(json.load(f))


# Load faiss index for nn-search
index = faiss.read_index(UNTRAINED_FAISS_PTH)


# Randomly grab keys
og_keys = list(annots.keys())
stochastic_idc = sample(range(0,len(og_keys)), SAMPLE_SIZE)
rnd_keys = [og_keys[idx] for idx in stochastic_idc]


# Gen embeds
print("Generating Audio Embeds")
audio_embeds = UNTRAINED_MODEL.get_audio_embedding_from_filelist([key for key in rnd_keys])    # get all audio embeds and L2-normalize them (len==1)
faiss.normalize_L2(audio_embeds)
audio_embd_pth = os.path.join(EVAL_DATA_DIR, "normalized_audio_embeds.npy")
np.save(audio_embd_pth, audio_embeds)

print("Generating Annotation Embeds")
annot_embeds = UNTRAINED_MODEL.get_text_embedding([annots[key] for key in rnd_keys])   # get all annot embeds and L2-normalize
faiss.normalize_L2(annot_embeds)
annot_embd_pth = os.path.join(EVAL_DATA_DIR, "normalized_annot_embeds.npy")
np.save(annot_embd_pth, annot_embeds)

# Calc Cos-Sim matrix
audio_annot_cos = audio_embeds @ annot_embeds.T


# Create visual
# Keep in mind: similar sounds of course have similar embeddings, hence cosine sims will be somewhat high
# Important: Diagonal should always be highest
fig_pth = os.path.join(EVAL_DATA_DIR, "untuned_cossims_audio_annot.png")
ax = sns.heatmap(audio_annot_cos, annot=False)   # Display data without
ax.set(xlabel="annotation-embeddings", ylabel="audio-embeddings", fontsize=12)
ax.set_title("Cosine Similarites: Audio- & Annotation-Embeddings", fontsize=14)
ax.tick_params(left=False, bottom=False)
fig = ax.get_figure()
fig.suptitle("Untuned Model", fontsize=16)
fig.savefig(fig_pth, dpi=300,bbox_inches="tight")


# # PRETUNE-LOOP
# cos_matches_annot = []
# cos_matches_audio = []

# for i, key in enumerate(rnd_keys):

#     # Get normal embeds (reuse prev. embeddings created using same keys)
#     audio_embed = audio_embeds[i]
#     annot_embed = annot_embeds[i]

#     # Get top 3 matches
#     _, I = index.search([annot_embed], 3)   
#     I = I.flatten() # make flat for easier access

#     flat = faiss.downcast_index(index.index)
#     match_embeds = [flat.reconstruct(match_id) for match_id in I]   # list containing top 3 matching embeds for Text input
#     print([np.linalg.norm(match_embed) for match_embed in match_embeds])    # CHeck whether len of embeds are really ~ 1

#     # Calc Cos-Sims:
#     cos_matches_annot.extend([np.dot(annot_embed[0], match) for match in match_embeds])   # cos-sims betw. annot embed & top 3 matches
#     cos_matches_audio.extend([np.dot(audio_embed[0], match) for match in match_embeds])   # cos-sims betw. audio embed & top 3 matches


# #TODO: Visualize match-sim