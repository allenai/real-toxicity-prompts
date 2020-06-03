import pandas as pd
from pathlib import Path

from tqdm import tqdm

from utils.constants import SHARED_DIR

GENERATIONS_DIR = Path(SHARED_DIR / 'generations/prompt_n_50percent_toxicity_quartiles_25k_subsample')
gpt2_df = pd.read_json(GENERATIONS_DIR / 'prompt_n_50percent_toxicity_quartiles_25k_subsample_gpt2.jsonl',
                       orient='records', lines=True)

old_dir = GENERATIONS_DIR / 'old'
new_dir = GENERATIONS_DIR / 'new'

for file in tqdm(old_dir.iterdir(), total=9):
    new_df = pd.read_json(file, orient='records', lines=True)
    new_df['prompt'] = gpt2_df['prompt']
    new_df['continuation'] = gpt2_df['continuation']
    new_df = new_df[gpt2_df.columns]
    print(new_df.columns)

    print("Writing new json file")
    new_df.to_json(new_dir / file.name, orient='records', lines=True)
