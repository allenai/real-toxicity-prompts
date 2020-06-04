import pandas as pd
from pathlib import Path

from utils.constants import SHARED_DIR

GENERATIONS_DIR = Path(SHARED_DIR / 'generations/prompt_n_50percent_toxicity_quartiles_25k_subsample')
gpt2_df = pd.read_json(GENERATIONS_DIR / 'prompt_n_50percent_toxicity_quartiles_25k_subsample_gpt2.jsonl',
                       orient='records', lines=True)


def fix_generations_file(in_file, out_file):
    new_df = pd.read_json(in_file, orient='records', lines=True)
    new_df['prompt'] = gpt2_df['prompt']
    new_df['continuation'] = gpt2_df['continuation']
    new_df = new_df[gpt2_df.columns]
    print(new_df.columns)

    print("Writing new json file")
    new_df.to_json(out_file, orient='records', lines=True)
