from transformers import AutoModel, ClapProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio
import json


class CLAPContrastiveModel(nn.Module):

    def __init__(self, pretrained_name: str = "laion/clap-htsat-unfused"):
        """
        Initialize the CLAP contrastive model.

        Args:
            pretrained_name (str):
                Name of the pretrained CLAP model from Hugging Face.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_name)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, audio: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, **kwargs) -> dict:
        """
        Forward pass for audio-text learning.

        Computes audio and text embeddings using the pretrained model,
        normalizes them for cosine similarity, and calculates a symmetric
        contrastive loss (audio-to-text and text-to-audio).

        Necessary for Hugging Face Trainer.
        Loss must be returned in dict.

        Args:
            audio (torch.Tensor):
                Batch of audio inputs as expected by the 
                get_audio_features method from the model.

            input_ids (torch.LongTensor):
                Tokenized text input IDs.

            attention_mask (torch.LongTensor):
                Attention mask for input ids.

        Returns:
            dict:
                dictionary containing:
                    - loss 
                    - logits
            
        """

        # get text and audio embedding
        audio_emb = self.model.get_audio_features(audio)
        text_emb = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # normalise for cosine-similarity
        audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # matrix with dot products of audio and text embeddings
        logits = audio_emb @ text_emb.T
        # diagonal of logits contains "true" value of matching audio text pairs
        targets = torch.arange(audio_emb.size(0), device=audio_emb.device)

        # symmetric contrastive loss (Audi->Text + Tex->Audio)
        loss = (self.loss_fn(logits, targets) + self.loss_fn(logits.T, targets)) / 2
        return {"loss": loss, "logits": logits}

class DrumJSONDataset(Dataset):
    """
    Dataset for audio-text pairs stored in a JSON file.

    The JSON file is expected to map audio file paths to descriptions:
    {
    "path/to/audio.wav": "description",
    ...
    }
    """

    def __init__(self, json_path: str, target_sr: int = 48000):
        """
        Initialize the dataset.

        Args:
            json_path (str):
                Path to the JSON file containing the dataset.

            target_sr (int):
                Target sampling rate for all audio files.
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.audio_paths = list(self.data.keys())
        self.target_sr = target_sr

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        Necessary for Hugging Face Trainer.
        """
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single dataset sample.
        Necessary for Hugging Face Trainer.

        Args:
            idx (int):
                Index of the requested sample.

        Returns:
            dict:
                Dictionary containing:
                - audio (torch.Tensor): mono audio waveform
                - text (str): text description
        """
        path = self.audio_paths[idx]
        text = self.data[path]

        # load audio
        waveform, sr = torchaudio.load(path)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # if stereo audio: mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        target_seconds = 10
        target_samples = self.target_sr * target_seconds

        num_samples = waveform.shape[1]

        # if audio longer than 10 seconds, get 10 seconds from the middle
        if num_samples > target_samples:
            center = num_samples // 2
            half = target_samples // 2
            start = center - half
            end = center + half
            waveform = waveform[:, start:end]

        return {
            "audio": waveform.squeeze(0),      # Mel-Spectrum
            "text": text,       # Text-Token
        }
    

class DataCollatorClap:
    """
    Custom data collator for CLAP audio-text training.

    This collator tokenizes text inputs and converts raw audio
    waveforms into model-ready audio features.
    """
    def __init__(self, processor: ClapProcessor, target_sr: int = 48000):
        """
        Initialize the data collator.

        Args:
            processor (ClapProcessor):
                CLAP processor providing tokenizer and feature extractor.
            target_sr (int):
                Sampling rate expected by the audio feature extractor.
        """
        self.processor = processor
        self.target_sr = target_sr
        self.fe = processor.feature_extractor
        self.tok = processor.tokenizer

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate a batch of dataset samples.
        Necessary for Hugging Face Trainer.

        Args:
            features (list[dict]):
                List of samples produced by the dataset.

        Returns:
            dict:
                Batch dictionary containing:
                - audio (torch.Tensor)
                - input_ids (torch.LongTensor)
                - attention_mask (torch.LongTensor)
        """
        audios = [f["audio"] for f in features]  # List[1D torch.Tensor]
        texts  = [f["text"]  for f in features]

        text_batch = self.tok(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        audio_feats = []
        for audio in audios:
            audio_np = audio.detach().cpu().numpy()  # 1D
            out = self.fe(
                audio_np,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding="repeatpad"
            )
            audio_feats.append(out["input_features"][0])

        audio_batch = torch.stack(audio_feats, dim=0)  

        return {
            "audio": audio_batch,
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
        }



base_model = CLAPContrastiveModel("laion/clap-htsat-unfused")

lora_cfg = LoraConfig(
    r=4, # rank of low ranked matrices
    lora_alpha=32, # parameter determines how much the LoRA updates impact the model’s outputs
    target_modules=['text_model.encoder.layer.0.attention.self.query', 'text_model.encoder.layer.1.attention.self.query', 'text_model.encoder.layer.2.attention.self.query', 'text_model.encoder.layer.3.attention.self.query', 'text_model.encoder.layer.4.attention.self.query', 'text_model.encoder.layer.5.attention.self.query', 'text_model.encoder.layer.6.attention.self.query', 'text_model.encoder.layer.7.attention.self.query', 'text_model.encoder.layer.8.attention.self.query', 'text_model.encoder.layer.9.attention.self.query', 'text_model.encoder.layer.10.attention.self.query', 'text_model.encoder.layer.11.attention.self.query', 'audio_model.audio_encoder.layers.0.blocks.0.attention.self.query', 'audio_model.audio_encoder.layers.0.blocks.1.attention.self.query', 'audio_model.audio_encoder.layers.1.blocks.0.attention.self.query', 'audio_model.audio_encoder.layers.1.blocks.1.attention.self.query', 'audio_model.audio_encoder.layers.2.blocks.0.attention.self.query', 'audio_model.audio_encoder.layers.2.blocks.1.attention.self.query', 'audio_model.audio_encoder.layers.2.blocks.2.attention.self.query', 'audio_model.audio_encoder.layers.2.blocks.3.attention.self.query', 'audio_model.audio_encoder.layers.2.blocks.4.attention.self.query', 'audio_model.audio_encoder.layers.2.blocks.5.attention.self.query', 'audio_model.audio_encoder.layers.3.blocks.0.attention.self.query', 'audio_model.audio_encoder.layers.3.blocks.1.attention.self.query','text_model.encoder.layer.0.attention.self.value', 'text_model.encoder.layer.1.attention.self.value', 'text_model.encoder.layer.2.attention.self.value', 'text_model.encoder.layer.3.attention.self.value', 'text_model.encoder.layer.4.attention.self.value', 'text_model.encoder.layer.5.attention.self.value', 'text_model.encoder.layer.6.attention.self.value', 'text_model.encoder.layer.7.attention.self.value', 'text_model.encoder.layer.8.attention.self.value', 'text_model.encoder.layer.9.attention.self.value', 'text_model.encoder.layer.10.attention.self.value', 'text_model.encoder.layer.11.attention.self.value', 'audio_model.audio_encoder.layers.0.blocks.0.attention.self.value', 'audio_model.audio_encoder.layers.0.blocks.1.attention.self.value', 'audio_model.audio_encoder.layers.1.blocks.0.attention.self.value', 'audio_model.audio_encoder.layers.1.blocks.1.attention.self.value', 'audio_model.audio_encoder.layers.2.blocks.0.attention.self.value', 'audio_model.audio_encoder.layers.2.blocks.1.attention.self.value', 'audio_model.audio_encoder.layers.2.blocks.2.attention.self.value', 'audio_model.audio_encoder.layers.2.blocks.3.attention.self.value', 'audio_model.audio_encoder.layers.2.blocks.4.attention.self.value', 'audio_model.audio_encoder.layers.2.blocks.5.attention.self.value', 'audio_model.audio_encoder.layers.3.blocks.0.attention.self.value', 'audio_model.audio_encoder.layers.3.blocks.1.attention.self.value'],  # ggf. anpassen je nach Layernamen, irgendwie komisch aber passt wahrscheinlich so wie es ist
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION", 
)

model = get_peft_model(base_model, lora_cfg)
model.to('cuda')

processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

# train and eval json necessary in cwd
train_ds = DrumJSONDataset("train.json")
eval_ds = DrumJSONDataset("eval.json")

training_args = TrainingArguments(
    output_dir="./clap_lora_checkpoints",
    per_device_train_batch_size=4,
    learning_rate=1e-3,
    num_train_epochs=3, 
    logging_steps=10, 
    logging_strategy="steps",
    fp16=True,
    bf16=False,
    save_strategy="epoch",
    remove_unused_columns=False
)

trainer = Trainer(
    data_collator=DataCollatorClap(processor),
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()

model.save_pretrained("./clap_lora_adapter")

print("finished Training. Lora Adapter saved at ./clap_lora_adapter")
