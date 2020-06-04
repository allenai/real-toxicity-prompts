import os
import pandas as pd
from tqdm import tqdm
import re

dfs = []
files = os.listdir('../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/')
for file in tqdm(files):
     df = pd.read_json(os.path.join('../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/', file), lines=True)
     df['model'] = re.match('prompt_n_50percent_toxicity_quartiles_25k_subsample_(.*)\.jsonl', file).group(1)
     dfs.append(df)
master = pd.concat(dfs)

master.to_json("master_generations.jsonl", lines=True, orient='records')