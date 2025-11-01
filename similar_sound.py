
import faiss
import json
from laion_clap import CLAP_Module
import torch
from typing import List, Union
import numpy as np

model = CLAP_Module(enable_fusion=False, device = 'cuda' if torch.cuda.is_available() else 'cpu')
model.load_ckpt()

def get_text_embedding(text: str):
    """
    creates clap embedding for text
    
    Args:
        text(str): input text

    Returns:
        embedding
    """

    try:
        text_embedding = model.get_text_embedding([text])
        text_embedding = text_embedding.squeeze().astype(np.float32)
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        if text_embedding.ndim == 1:
            text_embedding = text_embedding[np.newaxis, :]
        return text_embedding
    
    except Exception as e:
        print(f"Error processing {text}: {e}")


def best_audio(input_text: str, mapping_json: str, faissfile: str, num_results: int = 10):
    """
    gets the best matching audio for given text
    based on cosine similarity of clap embeddings via faiss

    Args:
        input_text (str): input text
        mapping_json (str): path to mapping json file
        faissfile (str): path to faiss index
        num_results (int, optional): number of results to return. Defaults to 1

    Returns:
        list: list of audio paths
    """
    
   
    index = faiss.read_index(faissfile)

    with open(mapping_json, 'r') as f:
        index_to_path = json.load(f)

    embedding = get_text_embedding(text = input_text)
    D, I = index.search(embedding, num_results)
    
    similar_indices = I[0]
    similar_audio_paths = []
    for idx in similar_indices:
        similar_audio_paths.append(index_to_path[str(idx)])
    
    return similar_audio_paths
    
if __name__ == "__main__":
    input_text = input('text: ')
    similar_audio_paths = best_audio(input_text = input_text, mapping_json = 'audiopath_mapping.json', faissfile = 'audios.faiss')
    print(similar_audio_paths)
        