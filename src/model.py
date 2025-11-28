# Script to instantiate singular model used for all gen-embedding purposes of whole project


import torch
from laion_clap import CLAP_Module


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = CLAP_Module(enable_fusion=False, device=DEVICE)
MODEL.load_ckpt(verbose=False)