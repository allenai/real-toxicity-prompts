import torch

from utils.generation import GPT2Generator
from utils.constants import OUTPUT_DIR, PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.datasets import create_affect_vector
from models.affect_lm import AffectGPT2LMHeadModel

model_dir = OUTPUT_DIR / 'affect' / 'affect_5k_1layer_randomsample_upsample_toxicity_5k' / 'finetune_output'
model = AffectGPT2LMHeadModel.from_pretrained(model_dir)
model.affect.beta = -1

generator = GPT2Generator(model)
device = generator.device

affects = [create_affect_vector(**{attribute: 1}) for attribute in PERSPECTIVE_API_ATTRIBUTES_LOWER]
affects.append(create_affect_vector())
affects = torch.tensor(affects).unsqueeze(dim=1).to(device)

# prompt = GPT2Generator.STOP_TOKEN
prompt = "Hello there"
out = generator([prompt] * len(affects), max_len=50, affect_labels=affects)

for attribute, gen in zip(PERSPECTIVE_API_ATTRIBUTES_LOWER + ('neutral',), out):
    print(f'Generation with {attribute}:'.upper(), prompt + gen, sep='\n', end='\n\n')
