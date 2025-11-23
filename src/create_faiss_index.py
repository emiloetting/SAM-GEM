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



model = CLAP_Module(enable_fusion=False, 
                    device='cuda' if torch.cuda.is_available() else 'cpu')
model.load_ckpt(verbose=False)



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
                audio_embedding = model.get_audio_embedding_from_filelist([str(audio_file)])[0] # grab first element of list of single element
                if audio_embedding is None:
                    print(f"None Error processing {audio_path}")
                else: 
                    yield audio_path, audio_embedding
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        else:
            continue


def create_faiss(sample_dir:str, dst_dir:str) -> dict:
    """
    Creates annoy index and json file mapping index to audio path
    
    Args:
        folder_path (str): Path to folder containing audios
        dst_dir (str): Diretory where faiss-index & mapping-json are stored.

    Returns:
        path_mapping (dict): Dictionary containing faiss-IDs, paths and embeddings.
    """
    # Init dict to store as json later
    path_mapping = {
        "id" : None,
        "path" : [],
        "embedding" : []
    }
    
    # Iteratively fill mapping dir
    for i, (path, embedding) in enumerate(audio_embeddings_with_paths(sample_dir)):
        path_mapping["path"].append(str(path))
        path_mapping["embedding"].append(embedding)

    # reformat list of embeds to faiss-accepted form
    embeds = np.vstack([e.squeeze() for e in path_mapping["embedding"]]).astype(np.float32)
    faiss.normalize_L2(embeds)

    

    # Create IDs for faiss and json
    if len(path_mapping["path"]) == len(path_mapping["embedding"]):
        path_mapping["id"] = np.arange(start=0,
                                       stop=len(path_mapping["path"]),
                                       step=1,
                                       dtype=np.int64)
    else:
        raise IndexError("Paths and created embeddings to not correspond: unequal length of lists. (Either path(s) or embedding(s) missing")

    # Build index
    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeds[0].shape[-1]))
    index.add_with_ids(embeds, path_mapping["id"])
    faiss.write_index(index, os.path.join(dst_dir, "audio.faiss"))

    return path_mapping
        

# def add_audios(folder_path):
#     """
#     adds audios to existing faiss index and mapping

#     Args:
#         folder_path (str): path to folder containing audios

#     """

#     path_mapping = json.load(open("audiopath_mapping.json"))
#     length = len(path_mapping)


#     ids = []
#     embeds = []
#     for i, (path, embedding) in enumerate(audio_embeddings_with_paths(folder_path)):
#         if str(path) in path_mapping.values():
#             continue
#         else:
#             ids.append(i + length)

#             embeds.append(embedding)
#             path_mapping[i + length] = str(path)

#     embeds = np.vstack([e.squeeze() for e in embeds]).astype(np.float32)#
#     faiss.normalize_L2(embeds)
#     ids = np.array(ids).astype(np.int64)

#     index =faiss.read_index("audios.faiss")
#     index.add_with_ids(embeds, ids)
#     faiss.write_index(index, "audios.faiss")

#     with open("audiopath_mapping.json", "w") as f:
#         json.dump(path_mapping, f)
    