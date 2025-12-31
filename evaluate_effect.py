import os
import json
import faiss
import numpy as np
import seaborn as sns
from random import sample
import pandas as pd
from src.interface import InterFacer
from tqdm import tqdm
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


#======== GENERATE EMBEDS OF LABELED AUDIO FILES AND CORRESPONDING ANNOTATION =======================================================
CWD = os.getcwd()
EVAL_DATA_DIR = os.path.join(CWD, "data", "evaluation")
os.makedirs(EVAL_DATA_DIR, exist_ok=True)   # Create dir to store

# Liegt bei mir auf externer festplatte
HARD_DRIVE_PREFIX = "E"

# Aktuell: 1499 Samples
# Wir nehmen: 300 Random (Großer Teil wo viele ähnlich sind)
SAMPLE_SIZE = 100
UNTRAINED_FAISS_PTH = "data/audio.faiss"


# Load file with annots
with open("new_annotations.json", "r") as f:
    annots = dict(json.load(f))


# Load faiss index for nn-search
index = faiss.read_index(UNTRAINED_FAISS_PTH)


# Randomly grab keys
og_keys = list(annots.keys())
stochastic_idc = sample(range(0,len(og_keys)), SAMPLE_SIZE)
rnd_keys = [og_keys[idx] for idx in stochastic_idc]


# # Gen embeds
# print("Generating Audio Embeds")
# audio_embeds = UNTRAINED_MODEL.get_audio_embedding_from_filelist([key for key in rnd_keys])    # get all audio embeds and L2-normalize them (len==1)
# faiss.normalize_L2(audio_embeds)
audio_embd_pth = os.path.join(EVAL_DATA_DIR, "normalized_audio_embeds.npy")
# np.save(audio_embd_pth, audio_embeds)
audio_embeds = np.load(audio_embd_pth)


# print("Generating Annotation Embeds")
# annot_embeds = UNTRAINED_MODEL.get_text_embedding([annots[key] for key in rnd_keys])   # get all annot embeds and L2-normalize
# faiss.normalize_L2(annot_embeds)
annot_embd_pth = os.path.join(EVAL_DATA_DIR, "normalized_annot_embeds.npy")
# np.save(annot_embd_pth, annot_embeds)
annot_embeds = np.load(annot_embd_pth)

# Calc Cos-Sim matrix
audio_annot_cos = audio_embeds @ annot_embeds.T




# Create visual
# Keep in mind: similar sounds of course have similar embeddings, hence cosine sims will be somewhat high
# Important: Diagonal should always be highest
fig_pth = os.path.join(EVAL_DATA_DIR, "untuned_cossims_audio_annot.png")
sns.set_context("paper", font_scale=1.2)
ax = sns.heatmap(audio_annot_cos, annot=False, xticklabels=False, yticklabels=False, vmin=0, vmax=1)   # Display data without
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)
cbar.set_label("Cosine Similarity", fontsize=9)
ax.set_xlabel("annotation-embeddings", fontsize=8)
ax.set_ylabel("audio-embeddings", fontsize=8)
ax.set_title("Cosine Similarites: Audio- & Annotation-Embeddings", fontsize=10)
ax.tick_params(left=False, bottom=False)
fig = ax.get_figure()
fig.suptitle("Untuned Model", fontsize=12)
fig.savefig(fig_pth, dpi=300,bbox_inches="tight")



# PRETUNE-LOOP
cos_matches_annot = []
cos_matches_audio = []
interf = InterFacer(cwd=CWD)

with tqdm(leave=False, total=SAMPLE_SIZE, bar_format='{percentage:3.0f}%|{bar}|') as bar:
    for i, key in enumerate(rnd_keys):

        # Get normal embeds (reuse prev. embeddings created using same keys)
        audio_embed = np.array([audio_embeds[i]])
        annot_embed = np.array([annot_embeds[i]])

        # Get top 3 matches
        _, I_audio = index.search(audio_embed, 3)   # audio 
        _, I_annot = index.search(annot_embed, 3)   # annot

        I_audio = np.squeeze(I_audio).tolist()
        I_annot = np.squeeze(I_annot).tolist()

        match_embeds_audio = interf._grab_embeds_from_db(ids=I_audio)
        match_embeds_annot = interf._grab_embeds_from_db(ids=I_annot)

        # Calc Cos-Sims:
        cos_matches_audio.append([np.dot(audio_embed[0], match) for match in match_embeds_audio])   # cos-sims betw. audio embed & top 3 matches
        cos_matches_annot.append([np.dot(annot_embed[0], match) for match in match_embeds_annot])   # cos-sims betw. annot embed & top 3 matches
        bar.update(1)


# Box-Plot: Mean matches to audio cosines
fig_pth = os.path.join(EVAL_DATA_DIR, "untuned_mean_cossims_audio_matches.png")
data = pd.DataFrame(
    {"best match":    [cos[0] for cos in cos_matches_audio],
    "2nd best match": [cos[1] for cos in cos_matches_audio],
    "3rd best match": [cos[2] for cos in cos_matches_audio]}
)
fig = plt.figure(figsize=(14, 10))
ax = plt.gca()
box = sns.boxplot(fill=True, gap=.1, data=data, ax=ax)
fig = ax.get_figure()
fig.savefig(fig_pth)
plt.close(fig)



# Box-Plot: Mean matches to annot cosines
fig_pth = os.path.join(EVAL_DATA_DIR, "untuned_mean_cossims_annot_matches.png")
data = pd.DataFrame(
    {"best match": [cos[0] for cos in cos_matches_annot],
    "2nd best match": [cos[1] for cos in cos_matches_annot],
    "3rd best match": [cos[2] for cos in cos_matches_annot]}
)
fig = plt.figure(figsize=(14, 10))
ax = plt.gca()
box = sns.boxplot(fill=True, gap=.1, data=data, ax=ax)
fig = ax.get_figure()
fig.savefig(fig_pth)
plt.close(fig)