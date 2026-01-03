# Script to instantiate singular model used for all gen-embedding purposes of whole project
import os
from transformers import AutoModel
from peft import PeftModel, PeftConfig
from transformers import ClapProcessor

# Check whether config is available
cwd = os.getcwd()
model_dir = os.path.join(cwd, "data", "model")
config_pth = os.path.join(model_dir, "adapter_config.json")
if not os.path.exists(config_pth):
    raise FileNotFoundError(f"No file found at: {config_pth}")

# Load model
config = PeftConfig.from_pretrained(model_dir)
model = AutoModel.from_pretrained(config.base_model_name_or_path)

MODEL = PeftModel.from_pretrained(model, model_dir)
PROCESSOR = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
TOKENIZER = PROCESSOR.tokenizer
FEAT_EXTR = PROCESSOR.feature_extractor
TARGET_SR = 48000
TARGET_AUDIO_LEN_SEC = 5   # FML

