from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoProcessor
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
audio_sample = dataset["train"]["audio"][0]["array"]

#print(dataset["train"]["audio"][0]["array"])
model = AutoModel.from_pretrained("laion/clap-htsat-unfused", dtype=torch.float16, device_map="auto")
model_id="laion/clap-htsat-unfused"

processor = AutoProcessor.from_pretrained(model_id)
 
inputs2 = processor(audios = audio_sample, return_tensors="pt", padding=True)

print(inputs2)


# audio daten müssen am besten torch tensoren sein. siehe chatty code