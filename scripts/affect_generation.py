import torch

from utils.generation import GPT2Generator
from utils.constants import OUTPUT_DIR, PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.datasets import create_affect_vector
from models.affect_lm import AffectGPT2LMHeadModel

model_dir = OUTPUT_DIR / 'affect' / 'affectv1' / 'finetune_output' / 'checkpoint-12500'
model = AffectGPT2LMHeadModel.from_pretrained(model_dir)
model.affect.beta = 3

generator = GPT2Generator(model)
device = generator.device

affects = [create_affect_vector(**{attribute: 1})
           for attribute in PERSPECTIVE_API_ATTRIBUTES_LOWER]
affects.append(create_affect_vector())
affects = torch.stack(affects).unsqueeze(dim=1).to(device)

prompt = "Hello there I am a"
out = generator([prompt] * len(affects), max_len=200, affect_labels=affects)

for attribute, gen in zip(PERSPECTIVE_API_ATTRIBUTES_LOWER + ('neutral',), out):
    print(f'Generation with {attribute}:'.upper(), prompt + gen, sep='\n', end='\n\n')
