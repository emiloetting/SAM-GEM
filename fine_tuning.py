from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoProcessor
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")

print(dataset["train"]["audio"][0]["array"])
# model = AutoModel.from_pretrained("laion/clap-htsat-unfused", dtype=torch.float16, device_map="auto")
# model_id="laion/clap-htsat-unfused"

# processor = AutoProcessor.from_pretrained(model_id)
 
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# texts = ["hihat open loop hat top house disco tech 124bpm drum", "kick deep house tech techno tech house thump midlong"]

# inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
# inputs2 = processor(text=texts, return_tensors="pt", padding=True)

# print(inputs)
# print(inputs2)