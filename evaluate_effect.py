import os
import json
import faiss
import numpy as np
import seaborn as sns
from random import sample
import pandas as pd
from src.interface import InterFacer, InterFacerUntuned
from tqdm import tqdm
import matplotlib.pyplot as plt



"""
KONZEPT: 
Für alle annotierten Sounds:
Einmal VOR und einmal NACH Training:

- Annotation in Embedding wandeln
- Sample in Embedding wandeln
- Cos-Sim von Top3 Matches zu Annotation Embed
- Cos-Sim von Top3 Matches zu zugehörigem Sample
- BotPlots mit Ergebnissen der des trainierten und untrainierten Modells

"""

#======== GENERATE EMBEDS OF LABELED AUDIO FILES AND CORRESPONDING ANNOTATION =======================================================
CWD = os.getcwd()
EVAL_DATA_DIR = os.path.join(CWD, "data", "evaluation")
os.makedirs(EVAL_DATA_DIR, exist_ok=True)   # Create dir to store

# Liegt bei mir auf externer festplatte
HARD_DRIVE_PREFIX = "E"

# Aktuell: 1499 Samples
# Wir nehmen: 100 Random (Großer Teil wo viele ähnlich sind)
SAMPLE_SIZE = 100
UNTRAINED_FAISS_PTH = "data/audio_untuned.faiss"
FINETUNED_FAISS_PTH = "data/audio.faiss"


# Load file with annots
with open(os.path.join(EVAL_DATA_DIR, "eval.json"), "r") as f:
    annots = dict(json.load(f))


# Load faiss index for nn-search
index_untrained = faiss.read_index(UNTRAINED_FAISS_PTH)
index_trained = faiss.read_index(FINETUNED_FAISS_PTH)

# Load interfacer 
interf_untuned = InterFacerUntuned(cwd=CWD)
interf_fintuned = InterFacer(cwd=CWD)


# Randomly grab keys
og_keys = list(annots.keys())
stochastic_idc = sample(range(0,len(og_keys)), SAMPLE_SIZE)
rnd_keys = [og_keys[idx] for idx in stochastic_idc]





#====================================================================================================================================
# ALL THE UNTRAINED MODEL STUFF IS HERE, ALL TRAINED MODEL STUFF IS BELOW
#====================================================================================================================================

# Gen embeds
print("Generating untuned Audio Embeds")
untuned_audio_embeds = interf_untuned.untrained_model.get_audio_embedding_from_filelist([key for key in rnd_keys])    # get all audio embeds and L2-normalize them (len==1)
faiss.normalize_L2(untuned_audio_embeds)

print("Generating untuned Annotation Embeds")
untuned_annot_embeds = interf_untuned.untrained_model.get_text_embedding([annots[key] for key in rnd_keys])   # get all annot embeds and L2-normalize
faiss.normalize_L2(untuned_annot_embeds)


# PRETUNE-LOOP
cos_matches_annot_untrained = []
cos_matches_audio_untrained = []

with tqdm(leave=False, total=SAMPLE_SIZE, bar_format='{percentage:3.0f}%|{bar}|', desc="Untrained Cos-Sims") as bar:
    for i, key in enumerate(rnd_keys):

        # Get normal embeds (reuse prev. embeddings created using same keys)
        audio_embed = np.array([untuned_audio_embeds[i]])
        annot_embed = np.array([untuned_annot_embeds[i]])

        # Get top 3 matches
        _, I_audio_untrained = index_untrained.search(audio_embed, 4)   # audio 
        _, I_annot_untrained = index_untrained.search(annot_embed, 4)   # annot

        I_audio_untrained = np.squeeze(I_audio_untrained).tolist()
        I_annot_untrained = np.squeeze(I_annot_untrained).tolist()

        match_embeds_audio = interf_untuned._grab_embeds_from_db(ids=I_audio_untrained[1:])
        match_embeds_annot = interf_untuned._grab_embeds_from_db(ids=I_annot_untrained[1:])

        # Calc Cos-Sims:
        cos_matches_audio_untrained.append([np.dot(audio_embed[0], match) for match in match_embeds_audio])   # cos-sims betw. audio embed & top 3 matches
        cos_matches_annot_untrained.append([np.dot(annot_embed[0], match) for match in match_embeds_annot])   # cos-sims betw. annot embed & top 3 matches
        bar.update(1)




#====================================================================================================================================
# TRAINED MODEL METRICS
#====================================================================================================================================

# Gen embeds
print("Generating tuned Audio Embeds")
finetuned_audio_embeds = np.concatenate([interf_fintuned._gen_embed_from_audio(key) for key in rnd_keys], axis=0)

print("Generating tuned Annot Embeds")
finetuned_annot_embeds = np.concatenate([interf_fintuned._gen_embed(annots[key]) for key in rnd_keys], axis=0)
# Emebds were internally normalized


cos_matches_annot_finetuned = []
cos_matches_audio_finetuned = []

with tqdm(leave=False, total=SAMPLE_SIZE, bar_format='{percentage:3.0f}%|{bar}|', desc="Finetuned Cos-Sims") as bar:
    for i, key in enumerate(rnd_keys):

        # Get normal embeds (reuse prev. embeddings created using same keys)
        audio_embed = np.array([finetuned_audio_embeds[i]])
        annot_embed = np.array([finetuned_annot_embeds[i]])

        # Get top 3 matches
        _, I_audio_finetuned = index_trained.search(audio_embed, 4)   # audio 
        _, I_annot_finetuned = index_trained.search(annot_embed, 4)   # annot

        I_audio_finetuned = np.squeeze(I_audio_finetuned).tolist()
        I_annot_finetuned = np.squeeze(I_annot_finetuned).tolist()

        match_embeds_audio = interf_fintuned._grab_embeds_from_db(ids=I_audio_finetuned[1:])
        match_embeds_annot = interf_fintuned._grab_embeds_from_db(ids=I_annot_finetuned[1:])

        # Calc Cos-Sims:
        cos_matches_audio_finetuned.append([np.dot(audio_embed[0], match) for match in match_embeds_audio])   # cos-sims betw. audio embed & top 3 matches
        cos_matches_annot_finetuned.append([np.dot(annot_embed[0], match) for match in match_embeds_annot])   # cos-sims betw. annot embed & top 3 matches
        bar.update(1)



#==============================================================================
# COMBINED BOX PLOTS
#==============================================================================

# Combined boxplot: Audio matches (untuned vs finetuned)
combined_audio = []
for rank_idx, label in enumerate(["best match", "2nd best match", "3rd best match"]):
    combined_audio += [{"score": cos[rank_idx], "rank": label, "model": "Untuned"} for cos in cos_matches_audio_untrained]
    combined_audio += [{"score": cos[rank_idx], "rank": label, "model": "Finetuned"} for cos in cos_matches_audio_finetuned]
df_combined_audio = pd.DataFrame(combined_audio)

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="rank", y="score", hue="model", data=df_combined_audio,
                 palette={"Untuned": "#67AAF9", "Finetuned": "#e60028"}, saturation=1, legend=None)
ax.tick_params(axis="both", length=10, width=2)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.title("Audio matches: Untuned vs Finetuned", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DATA_DIR, "combined_audio_matches_boxplot.png"), dpi=300)
plt.close()

# Combined boxplot: Annotation matches (untuned vs finetuned)
combined_annot = []
for rank_idx, label in enumerate(["best match", "2nd best match", "3rd best match"]):
    combined_annot += [{"score": cos[rank_idx], "rank": label, "model": "Untuned"} for cos in cos_matches_annot_untrained]
    combined_annot += [{"score": cos[rank_idx], "rank": label, "model": "Finetuned"} for cos in cos_matches_annot_finetuned]
df_combined_annot = pd.DataFrame(combined_annot)

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="rank", y="score", hue="model", data=df_combined_annot,
                 palette={"Untuned": "#67AAF9", "Finetuned": "#e60028"}, saturation=1, legend=None)
ax.tick_params(axis="both", length=10, width=2)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.title("Annotation matches: Untuned vs Finetuned", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DATA_DIR, "combined_annot_matches_boxplot.png"), dpi=300)
plt.close()