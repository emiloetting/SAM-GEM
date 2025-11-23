import numpy as np
from pathlib import Path
import tqdm
import os
import faiss
from laion_clap import CLAP_Module
import torch
import json
import ffmpeg



cwd = os.getcwd()
path_to_database = os.path.join(cwd, "DataBase")



model = CLAP_Module(enable_fusion=False, device='cuda' if torch.cuda.is_available() else 'cpu')
model.load_ckpt()



def audio_embeddings_with_paths(folder_path):
    """
    creates clap embeddings for all wav files in folder

    Args:
        folder_path (str): path to folder containing audios

    Yields:
        tuple: (audio_path, embedding)
    """

    folder = Path(folder_path)
    
    for audio_path in tqdm.tqdm(folder.rglob("*.*")):
        if audio_path.is_file():
            if audio_path.suffix.lower() == ".wav":
                audio_file = audio_path
            else:
                file_name = audio_path.with_suffix(".wav")
                try:
                    ffmpeg.input(str(audio_path)).output(file_name, ac=1, ar=16000).run()
                    audio_file = file_name
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue

            try:
                audio_embedding = model.get_audio_embedding_from_filelist([str(audio_file)])
                if audio_embedding is None:
                    print(f"None Error processing {audio_path}")
                else: 
                    yield audio_path, audio_embedding
            except Exception as e:
                print(f"Error processing2 {audio_path}: {e}")
        else:
            continue


def create_faiss(folder_path):
    """
    creates annoy index and json file mapping index to audio path
    
    Args:
        folder_path (str): path to folder containing audios

    """

    path_mapping = {}

    ids = []
    embeds = []
    for i, (path, embedding) in enumerate(audio_embeddings_with_paths(folder_path)):
        ids.append(i)

        embeds.append(embedding)
        path_mapping[i] = str(path)

    embeds = np.vstack([e.squeeze() for e in embeds]).astype(np.float32)
    faiss.normalize_L2(embeds)
    ids = np.array(ids).astype(np.int64)

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeds[0].shape[-1]))
    index.add_with_ids(embeds, ids)
    faiss.write_index(index, "audios.faiss")

    with open("audiopath_mapping.json", "w") as f:
        json.dump(path_mapping, f)
        

def add_audios(folder_path):
    """
    adds audios to existing faiss index and mapping

    Args:
        folder_path (str): path to folder containing audios

    """

    path_mapping = json.load(open("audiopath_mapping.json"))
    length = len(path_mapping)


    ids = []
    embeds = []
    for i, (path, embedding) in enumerate(audio_embeddings_with_paths(folder_path)):
        if str(path) in path_mapping.values():
            continue
        else:
            ids.append(i + length)

            embeds.append(embedding)
            path_mapping[i + length] = str(path)

    embeds = np.vstack([e.squeeze() for e in embeds]).astype(np.float32)#
    faiss.normalize_L2(embeds)
    ids = np.array(ids).astype(np.int64)

    index =faiss.read_index("audios.faiss")
    index.add_with_ids(embeds, ids)
    faiss.write_index(index, "audios.faiss")

    with open("audiopath_mapping.json", "w") as f:
        json.dump(path_mapping, f)
    

if __name__ == "__main__":
    cwd = os.getcwd()
    path_to_database = r"C:\Users\joche\Music\Samples"
    create_faiss(path_to_database)