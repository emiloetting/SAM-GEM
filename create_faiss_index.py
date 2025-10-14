import numpy as np
from pathlib import Path
import tqdm
import os
import create_faiss_index
from msclap import CLAP
import json
import ffmpeg

cwd = os.getcwd()
path_to_database = os.path.join(cwd, "DataBase")

model = CLAP(version = '2023', use_cuda=False)

def image_embeddings_with_paths(folder_path):
    """
    creates clap embeddings for all wav files in folder

    Args:
        folder_path (str): path to folder containing audios

    Yields:
        tuple: (audio_path, embedding)
    """

    folder = Path(folder_path)
    

    for audio_path in tqdm.tqdm(folder.rglob("*.*")):
        if audio_path.suffix.lower() == ".wav":
            audio_file = audio_path
        else:
            try:
                file_name = audio_path.with_suffix(".wav")
                ffmpeg.input(audio_path).output(file_name, ac=1, ar=16000).run()
                audio_file = file_name
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

        try:
            audio_embedding = model.get_audio_embeddings(audio_file)
            yield audio_path, audio_embedding
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")



def create_faiss(folder_path):
    """
    creates annoy index and json file mapping index to audio path
    
    Args:
        folder_path (str): path to folder containing audios

    """

    path_mapping = {}

    ids = []
    embeds = []
    for i, (path, embedding) in enumerate(image_embeddings_with_paths(folder_path)):
        ids.append(i)
        embeds.append(embedding)
        path_mapping[i] = path

    embeds = np.vstack(embeds).astype(np.float32)
    ids = np.array(ids).astype(np.int64)

    index = create_faiss_index.IndexIDMap(create_faiss_index.IndexFlatIP(1024))
    index.add_with_ids(embeds, ids)
    create_faiss_index.write_index(index, "audios.faiss")

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
    for i, (path, embedding) in enumerate(image_embeddings_with_paths(folder_path)):
        if path in path_mapping:
            continue
        else:
            ids.append(i + length)
            embeds.append(embedding)
            path_mapping[i + length] = path

    embeds = np.vstack(embeds).astype(np.float32)
    ids = np.array(ids).astype(np.int64)

    index = create_faiss_index.read_index("audios.faiss")
    index.add_with_ids(embeds, ids)
    create_faiss_index.write_index(index, "audios.faiss")

    with open("audiopath_mapping.json", "w") as f:
        json.dump(path_mapping, f)
    

if __name__ == "__main__":
    cwd = os.getcwd()
    path_to_database = os.path.join(cwd, "DataBase")
    create_faiss(path_to_database)